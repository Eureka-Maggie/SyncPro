"""ProactiveStreamRunner: per-query wrapper around MaskedStreamingRunner.

Owns the model+template+tokenizer at module level; each query gets its own
MaskedStreamingRunner instance (fresh KV cache + step-mask state) plus a
running chat history that feeds the prefill-then-decode loop.

Frame source is decoupled: the caller hands in a PIL.Image per tick.
"""

from __future__ import annotations

import hashlib
import json
import math
import numpy as np
import os
import re
import sys
import time
import torch
import types
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'examples/infer/infer_test'))

from infer_stream_masked import common_prefix_len  # noqa: E402  (path injection above)
from infer_stream_masked import MaskedStreamingRunner, slice_pixel_values_for_image  # noqa: E402

from examples.infer.proactive_eval.checkpoint_utils import checkpoint_has_dual_weights, find_checkpoint_weight  # noqa: E402
from examples.infer.proactive_eval.checkpoint_utils import is_dual_weight_key as _is_dual_weight_key  # noqa: E402
from examples.infer.proactive_eval.runner_dual import _cache_duplicate_batch  # noqa: E402
from examples.infer.proactive_eval.runner_dual import (_cache_append_trail, _cache_select_batch,  # noqa: E402
                                                       _cache_select_batch_rows, _extract_packed_cache_row,
                                                       _is_streaming_cache, _merge_linear_state_from_branches,
                                                       _pack_dynamic_caches, _pack_query_prefix_caches)
from swift import get_processor, get_template  # noqa: E402
from swift.rl.dual_sampler import DualBranchSampler  # noqa: E402
from swift.template.step_mask import enable_flex_attn_support  # noqa: E402
from swift.template.step_mask import dual_lora_enabled, enable_dual_lora, patch_qwen3_5_for_step_mask  # noqa: E402
from swift.template.step_mask.dual_lora import _dual_lora_ctx, clear_ctx  # noqa: E402
from swift.template.step_mask.inference import build_batched_streaming_block_mask  # noqa: E402
from swift.template.step_mask.inference import (build_streaming_block_mask, reset_ctx, streaming_ctx_token,  # noqa: E402
                                                warmup_flex_attention)
from swift.template.step_mask.meta import SPECIAL_TOKEN_IDS  # noqa: E402

DEFAULT_SYSTEM_PROMPT_PATH = str(REPO_ROOT / 'data/system_prompt.txt')
DUAL_SYSTEM_PROMPT_PATH = str(REPO_ROOT / 'data/system_prompt_dual.txt')
NON_THINKING_PREFIX = '<think>\n\n</think>\n\n'
DUAL_BRANCH_SEPARATOR = '\n'
DUAL_TURN_TRAILER = '\n'
_IM_END = SPECIAL_TOKEN_IDS['im_end']

# Keep the compact per-query mask builder separate from the public module
# binding.  The GPU drift diagnostic wraps the latter to count the one packed
# streaming mask built per model forward; internal per-query masks must not
# advance that forward counter.
_build_query_group_block_mask = build_batched_streaming_block_mask


@dataclass(frozen=True)
class PreparedImageFeatures:
    """One frame's deterministic processor and frozen-vision outputs.

    RL samples several trajectories for the same query. Treating these
    tensors as read-only lets those trajectories share the expensive visual
    tower without sharing any language-model state or cache.
    """

    source_frame: Image.Image
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    image_embeds: torch.Tensor


def _number_from_run_tag(raw: str) -> float:
    return float(raw.replace('p', '.'))


def _fixed_gamma_from_policy(policy: str | None) -> float | None:
    if not policy or not str(policy).startswith('fixed'):
        return None
    raw = str(policy)[len('fixed'):]
    try:
        gamma = _number_from_run_tag(raw)
    except ValueError as exc:
        raise RuntimeError(f'invalid fixed merge policy {policy!r}') from exc
    if not math.isfinite(gamma) or not 0 < gamma < 1:
        raise RuntimeError(f'fixed merge gamma must be finite and in (0, 1), got {gamma!r}')
    return gamma


def dual_checkpoint_runtime_spec(model_path: str) -> dict:
    """Recover the custom dual-module constructor contract before loading.

    ``args.json`` records Swift's generic LoRA fields, not the custom
    ``MSS_LORA_*`` values used by :mod:`dual_lora`.  Rank is therefore read
    directly from a safetensors shape; alpha/dropout are recovered from the
    canonical ``-r64-a128-d0p05-`` run tag (or an optional future
    ``dual_infer_contract.json`` sidecar).  A wrong rank fails during weight
    load, but a wrong alpha silently rescales every adapter, so this pre-load
    guard is part of the production contract.
    """
    model_dir = Path(model_path)
    spec: dict = {}
    for manifest_path in (
            model_dir / 'dual_infer_contract.json',
            model_dir.parent / 'dual_infer_contract.json',
    ):
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            for key in (
                    'rank',
                    'alpha',
                    'dropout',
                    'merge_mode',
                    'merge_policy',
                    'merge_gamma',
                    'merge_gamma_init',
                    'merge_gate_mode',
                    'merge_alpha_init',
                    'merge_beta_init',
                    'position_overlap',
                    'source_fork',
                    'system_prompt_sha256',
                    'system_prompt_bytes',
                    'system_prompt_terminal_newline',
                    'loss_scale',
                    'add_non_thinking_prefix',
                    'step_mask_window',
                    'action_loss_mode',
                    'action_space',
                    'requires_constrained_decode',
            ):
                if key in manifest:
                    spec[key] = manifest[key]
            spec['manifest_path'] = str(manifest_path)
            break

    rank_weight = find_checkpoint_weight(model_dir, lambda key: '.lora_A_down.weight' in key)
    if rank_weight is not None:
        rank_key, tensor_path = rank_weight
        from safetensors import safe_open
        try:
            with safe_open(tensor_path, framework='pt', device='cpu') as f:
                tensor_rank = int(f.get_slice(rank_key).get_shape()[0])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f'cannot read dual rank tensor {rank_key!r} from '
                               f'{tensor_path}') from e
        if 'rank' in spec and int(spec['rank']) != tensor_rank:
            raise RuntimeError(f'dual manifest rank={spec["rank"]} disagrees with '
                               f'tensor shape rank={tensor_rank}')
        spec.setdefault('rank', tensor_rank)

    run_tag = str(model_dir)
    tag_match = re.search(
        r'(?:^|[-_/])r(?P<rank>\d+)-a(?P<alpha>\d+(?:p\d+)?)'
        r'(?:-d(?P<dropout>\d+(?:p\d+)?))?',
        run_tag,
    )
    if tag_match:
        tag_rank = int(tag_match.group('rank'))
        if 'rank' in spec and int(spec['rank']) != tag_rank:
            raise RuntimeError(f'dual checkpoint rank tag r{tag_rank} disagrees with tensor '
                               f'shape rank={spec["rank"]}')
        spec.setdefault('rank', tag_rank)
        spec.setdefault('alpha', _number_from_run_tag(tag_match.group('alpha')))
        if tag_match.group('dropout') is not None:
            spec.setdefault('dropout', _number_from_run_tag(tag_match.group('dropout')))

    run_tag_lower = run_tag.lower()
    fixed_match = re.search(
        r'(?:^|[-_/])fixed(?P<gamma>\d+(?:p\d+)?)(?:$|[-_/])',
        run_tag_lower,
    )
    learned_match = re.search(
        r'(?:^|[-_/])learned(?:[_-]?gamma)(?:$|[-_/])',
        run_tag_lower,
    )
    split_match = re.search(
        r'(?:^|[-_/])learned[-_]?alpha[-_]?beta'
        r'(?:[-_]?init(?P<alpha>\d+(?:p\d+)?)[-_](?P<beta>\d+(?:p\d+)?))?'
        r'(?:$|[-_/])',
        run_tag_lower,
    )
    tagged_policies = []
    tagged_gamma = None
    if fixed_match:
        gamma_tag = fixed_match.group('gamma')
        tagged_policies.append(f'fixed{gamma_tag}')
        tagged_gamma = _number_from_run_tag(gamma_tag)
    if learned_match:
        tagged_policies.append('learned_gamma')
    if split_match:
        tagged_policies.append('learned_alpha_beta')
        spec.setdefault('merge_gate_mode', 'split')
        if split_match.group('alpha') is not None:
            tagged_split_weights = {
                'merge_alpha_init': _number_from_run_tag(split_match.group('alpha')),
                'merge_beta_init': _number_from_run_tag(split_match.group('beta')),
            }
            for key, value in tagged_split_weights.items():
                if (key in spec and not math.isclose(
                        float(spec[key]), value, rel_tol=0.0, abs_tol=1e-12)):
                    raise RuntimeError(f'dual manifest {key}={spec[key]!r} conflicts with '
                                       f'checkpoint path tag={value!r}')
                spec.setdefault(key, value)
    if len(tagged_policies) > 1:
        raise RuntimeError(f'dual checkpoint path contains conflicting merge-policy tags: '
                           f'{tagged_policies}')
    tagged_policy = tagged_policies[0] if tagged_policies else None

    gamma_policy = None
    args_path = model_dir / 'args.json'
    if args_path.is_file():
        train_args = json.loads(args_path.read_text(encoding='utf-8'))
        for key in (
                'loss_scale',
                'add_non_thinking_prefix',
                'step_mask_window',
        ):
            if key not in train_args:
                continue
            value = train_args[key]
            if key in spec and spec[key] != value:
                raise RuntimeError(f'dual manifest {key}={spec[key]!r} conflicts with '
                                   f'checkpoint args.json value={value!r}')
            spec.setdefault(key, value)
        trainable = str(train_args.get('trainable_parameters_regex') or '')
        gate_names = ('gamma_logit', 'alpha_logit', 'beta_logit')
        all_merge_gates_trainable = ('dual_merge' in trainable
                                     and not any(name in trainable for name in gate_names))
        learned_gamma = all_merge_gates_trainable or 'gamma_logit' in trainable
        learned_alpha = all_merge_gates_trainable or 'alpha_logit' in trainable
        learned_beta = all_merge_gates_trainable or 'beta_logit' in trainable
        if learned_alpha != learned_beta:
            raise RuntimeError('split merge checkpoint must train both alpha_logit and beta_logit')
        if learned_alpha:
            gamma_policy = 'learned_alpha_beta'
            spec.setdefault('merge_gate_mode', 'split')
        elif learned_gamma:
            gamma_policy = 'learned_gamma'
        else:
            manifest_policy = str(spec.get('merge_policy') or '')
            gamma_policy = (manifest_policy if manifest_policy.startswith('fixed') else
                            (tagged_policy if tagged_policy and tagged_policy.startswith('fixed')
                             else 'fixed0p5'))

    tagged_modes = [mode for mode in ('additive', 'sequential') if mode in run_tag_lower]
    if len(tagged_modes) > 1:
        raise RuntimeError(f'dual checkpoint path contains conflicting merge tags: '
                           f'{tagged_modes}')
    tagged_mode = tagged_modes[0] if tagged_modes else None
    if (tagged_mode is not None and 'merge_mode' in spec and str(spec['merge_mode']) != tagged_mode):
        raise RuntimeError(f'dual manifest merge_mode={spec["merge_mode"]!r} conflicts '
                           f'with checkpoint path tag={tagged_mode!r}')
    if tagged_mode is not None:
        spec.setdefault('merge_mode', tagged_mode)

    if (tagged_policy is not None and gamma_policy is not None and tagged_policy != gamma_policy):
        raise RuntimeError(f'dual checkpoint merge-policy tag={tagged_policy!r} conflicts '
                           f'with args.json-derived policy={gamma_policy!r}')
    checkpoint_policy = gamma_policy or tagged_policy

    expected_policy = None
    if spec.get('merge_mode') == 'sequential':
        expected_policy = 'sequential'
        if checkpoint_policy is not None:
            raise RuntimeError(f'sequential checkpoint cannot carry additive merge-policy '
                               f'{checkpoint_policy!r}')
    elif (spec.get('merge_mode') == 'additive' and checkpoint_policy is not None):
        expected_policy = checkpoint_policy
    if (expected_policy is not None and 'merge_policy' in spec and str(spec['merge_policy']) != expected_policy):
        raise RuntimeError(f'dual manifest merge_policy={spec["merge_policy"]!r} conflicts '
                           f'with checkpoint-derived policy={expected_policy!r}')
    if expected_policy is not None:
        spec.setdefault('merge_policy', expected_policy)
    inferred_gate_mode = ('split' if spec.get('merge_policy') == 'learned_alpha_beta' else 'shared')
    if ('merge_gate_mode' in spec and str(spec['merge_gate_mode']) != inferred_gate_mode):
        raise RuntimeError(f'dual manifest merge_gate_mode={spec["merge_gate_mode"]!r} conflicts '
                           f'with merge_policy={spec.get("merge_policy")!r}')
    spec.setdefault('merge_gate_mode', inferred_gate_mode)
    fixed_gamma = _fixed_gamma_from_policy(spec.get('merge_policy'))
    if fixed_gamma is not None:
        if 'merge_gamma' in spec and not math.isclose(
                float(spec['merge_gamma']), fixed_gamma, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f'dual manifest merge_gamma={spec["merge_gamma"]!r} conflicts '
                               f'with merge_policy={spec["merge_policy"]!r}')
        if tagged_gamma is not None and not math.isclose(
                tagged_gamma, fixed_gamma, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f'dual checkpoint fixed-gamma tag={tagged_gamma!r} conflicts '
                               f'with merge_policy={spec["merge_policy"]!r}')
        spec.setdefault('merge_gamma', fixed_gamma)
    if spec.get('merge_policy') == 'learned_alpha_beta':
        for key in ('merge_alpha_init', 'merge_beta_init'):
            if key in spec:
                value = float(spec[key])
                if not math.isfinite(value) or not 0 < value < 1:
                    raise RuntimeError(f'{key} must be finite and in (0, 1), got {value!r}')
    if 'overlap' in run_tag:
        spec.setdefault('position_overlap', True)
    if 'sourcefork' in run_tag:
        spec.setdefault('source_fork', True)
    # Historical production dual checkpoints predate the sidecar field, but
    # every source-fork + overlap training entry used W=1 ("see once, then
    # discard"). Preserve that one documented legacy contract here; new
    # checkpoints must persist the value explicitly in the sidecar.
    if ('step_mask_window' not in spec and spec.get('source_fork') is True and spec.get('position_overlap') is True):
        spec['step_mask_window'] = 1
        spec['step_mask_window_source'] = 'legacy_dual_sourcefork_overlap'
    if 'step_mask_window' in spec:
        try:
            spec['step_mask_window'] = int(spec['step_mask_window'])
        except (TypeError, ValueError) as exc:
            raise RuntimeError('dual checkpoint step_mask_window must be a positive integer') from exc
        if spec['step_mask_window'] <= 0:
            raise RuntimeError('dual checkpoint step_mask_window must be a positive integer')
    action_loss_mode = spec.get('action_loss_mode', 'full_vocab')
    if action_loss_mode not in {'full_vocab', 'restricted_binary'}:
        raise RuntimeError(f'unsupported dual action_loss_mode={action_loss_mode!r}')
    if action_loss_mode == 'restricted_binary':
        if spec.get('action_space') != 'dual_grammar_v1':
            raise RuntimeError('restricted_binary checkpoint requires action_space=dual_grammar_v1')
        if spec.get('requires_constrained_decode') is not True:
            raise RuntimeError('restricted_binary checkpoint requires constrained decoding')
    spec.setdefault('dropout', 0.05)
    return spec


def _set_or_validate_dual_env(name: str, expected, *, normalizer=str) -> None:
    expected_norm = normalizer(expected)
    current = os.environ.get(name)
    if current is not None and normalizer(current) != expected_norm:
        raise RuntimeError(f'{name}={current!r} conflicts with checkpoint contract '
                           f'{expected_norm!r}')
    os.environ[name] = str(expected)


def configure_dual_runtime(model_path: str) -> dict:
    spec = dual_checkpoint_runtime_spec(model_path)
    required = (
        'rank',
        'alpha',
        'merge_mode',
        'merge_policy',
        'position_overlap',
        'source_fork',
        'step_mask_window',
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise RuntimeError(f'incomplete dual checkpoint runtime contract; missing {missing}. '
                           'Use a canonical run tag or dual_infer_contract.json.')
    if spec['position_overlap'] is not True or spec['source_fork'] is not True:
        raise RuntimeError('the production dual runner requires position_overlap=true and '
                           'source_fork=true')
    if spec['merge_mode'] != 'additive':
        raise RuntimeError('the production independent-branch runner currently supports '
                           'only merge_mode=additive')
    _set_or_validate_dual_env('MSS_DUAL_BRANCH', 1, normalizer=int)
    if 'rank' in spec:
        _set_or_validate_dual_env('MSS_LORA_RANK', spec['rank'], normalizer=int)
    if 'alpha' in spec:
        _set_or_validate_dual_env('MSS_LORA_ALPHA', spec['alpha'], normalizer=float)
    if 'dropout' in spec:
        _set_or_validate_dual_env('MSS_LORA_DROPOUT', spec['dropout'], normalizer=float)
    if 'merge_mode' in spec:
        _set_or_validate_dual_env('MSS_MERGE_MODE', spec['merge_mode'])
    _set_or_validate_dual_env('MSS_MERGE_GATE_MODE', spec.get('merge_gate_mode', 'shared'))
    merge_gamma = spec.get('merge_gamma', spec.get('merge_gamma_init'))
    if merge_gamma is not None:
        _set_or_validate_dual_env('MSS_MERGE_GAMMA', merge_gamma, normalizer=float)
    if 'merge_alpha_init' in spec:
        _set_or_validate_dual_env('MSS_MERGE_ALPHA', spec['merge_alpha_init'], normalizer=float)
    if 'merge_beta_init' in spec:
        _set_or_validate_dual_env('MSS_MERGE_BETA', spec['merge_beta_init'], normalizer=float)
    if spec.get('position_overlap'):
        _set_or_validate_dual_env('MSS_POSITION_ID_OVERLAP', 1, normalizer=int)
    if spec.get('source_fork'):
        _set_or_validate_dual_env('MSS_DUAL_BRANCH_FORK_SOURCE', 1, normalizer=int)
    _set_or_validate_dual_env('STEP_MASK_WINDOW', spec['step_mask_window'], normalizer=int)
    return spec


def assert_dual_system_prompt(model_path: str, prompt_path: str, spec: dict) -> None:
    prompt = Path(prompt_path).read_text(encoding='utf-8')
    args_path = Path(model_path) / 'args.json'
    if args_path.is_file():
        train_prompt = json.loads(args_path.read_text(encoding='utf-8')).get('system')
        if train_prompt is not None and train_prompt != prompt:
            raise RuntimeError(f'inference system prompt differs from checkpoint args.json: '
                               f'{prompt_path}')
    prompt_bytes = prompt.encode('utf-8')
    if 'system_prompt_sha256' in spec:
        digest = hashlib.sha256(prompt_bytes).hexdigest()
        if digest != spec['system_prompt_sha256']:
            raise RuntimeError('inference system prompt SHA256 conflicts with dual manifest: '
                               f'{digest} != {spec["system_prompt_sha256"]}')
    if 'system_prompt_bytes' in spec and len(prompt_bytes) != int(spec['system_prompt_bytes']):
        raise RuntimeError('inference system prompt byte length conflicts with dual manifest')
    if (spec.get('system_prompt_terminal_newline') is True and not prompt.endswith('\n')):
        raise RuntimeError('dual training prompt ended with newline but inference prompt does not')


def assert_dual_weights_loaded(loading_info: dict, *, expected: bool) -> None:
    if not expected or not loading_info:
        return
    unexpected = [k for k in loading_info.get('unexpected_keys', []) if _is_dual_weight_key(k)]
    missing = [k for k in loading_info.get('missing_keys', []) if _is_dual_weight_key(k)]
    if unexpected or missing:
        raise RuntimeError('dual checkpoint did not load cleanly: '
                           f'unexpected_dual_keys={unexpected[:8]} '
                           f'missing_dual_keys={missing[:8]}')


def assert_dual_merge_policy(model, spec: dict) -> None:
    """Fail closed when a fixed-gamma checkpoint carries different gates."""
    merge_modules = [module for module in model.modules() if type(module).__name__ == 'GatedAdditiveMerge']
    if not merge_modules:
        raise RuntimeError('dual model has no GatedAdditiveMerge modules')
    policy = spec.get('merge_policy')
    if policy == 'learned_alpha_beta':
        mismatched = [
            idx for idx, module in enumerate(merge_modules)
            if not (hasattr(module, 'alpha_logit') and hasattr(module, 'beta_logit'))
            or hasattr(module, 'gamma_logit')
        ]
        if mismatched:
            raise RuntimeError('checkpoint requires independent alpha/beta merge gates, but '
                               f'modules {mismatched[:8]} use a different gate layout')
        return
    mismatched_layout = [
        idx for idx, module in enumerate(merge_modules)
        if not hasattr(module, 'gamma_logit')
        or hasattr(module, 'alpha_logit')
        or hasattr(module, 'beta_logit')
    ]
    if mismatched_layout:
        raise RuntimeError(f'checkpoint requires a shared gamma merge gate, but modules '
                           f'{mismatched_layout[:8]} use a different gate layout')
    expected_gamma = _fixed_gamma_from_policy(policy)
    if expected_gamma is None:
        return
    mismatched = [
        idx for idx, module in enumerate(merge_modules)
        if not torch.allclose(
            torch.sigmoid(module.gamma_logit.detach().float()),
            torch.full_like(module.gamma_logit.detach().float(), expected_gamma),
            rtol=0.0,
            atol=1e-6,
        )
    ]
    if mismatched:
        raise RuntimeError(f'checkpoint training fixed merge gamma at {expected_gamma:g}, but '
                           f'loaded different gates at merge modules {mismatched[:8]}')


_DUAL_ROUTE_TENSOR_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def set_dual_routes(routes,
                    seq_len: int,
                    device,
                    *,
                    query_row_group_sizes=None,
                    query_full_attention_groups=None) -> None:
    """Set LoRA routes and optional per-query physical-row boundaries."""
    routes = tuple(routes)
    if not routes:
        raise ValueError('routes must contain at least one batch row')
    batch = len(routes)
    clear_ctx()
    unknown = [route for route in routes if route not in ('A', 'B', 'none')]
    if unknown:
        raise ValueError(f'unknown dual LoRA route: {unknown[0]!r}')
    has_A = 'A' in routes
    has_B = 'B' in routes
    # A base-only shared/trailer forward leaves the LoRA context empty, which
    # lets every DualLoraLinear return immediately without inspecting a CUDA
    # route tensor. Branch forwards cache the two booleans once here instead
    # of synchronizing on route.any().item() in all 96 wrapped projections.
    # The handful of immutable [batch, seq] route patterns are cached as well,
    # avoiding two CUDA allocations on every generated token.
    if has_A or has_B:
        key = (routes, int(seq_len), str(torch.device(device)))
        cached = _DUAL_ROUTE_TENSOR_CACHE.get(key)
        if cached is None:
            route_A = torch.tensor(
                [route == 'A' for route in routes],
                dtype=torch.bool,
                device=device,
            ).view(batch, 1).expand(batch, seq_len)
            route_B = torch.tensor(
                [route == 'B' for route in routes],
                dtype=torch.bool,
                device=device,
            ).view(batch, 1).expand(batch, seq_len)
            cached = (route_A, route_B)
            _DUAL_ROUTE_TENSOR_CACHE[key] = cached
        route_A, route_B = cached
        _dual_lora_ctx['route_A'] = route_A
        _dual_lora_ctx['route_B'] = route_B
        _dual_lora_ctx['has_A'] = has_A
        _dual_lora_ctx['has_B'] = has_B
        if os.environ.get('MSS_DUAL_ROUTE_SYNC_REFERENCE', '0') == '1':
            _dual_lora_ctx['sync_route_presence_reference'] = True
    if query_row_group_sizes is not None:
        group_sizes = tuple(int(size) for size in query_row_group_sizes)
        if (not group_sizes or any(size not in (1, 2) for size in group_sizes) or sum(group_sizes) != batch):
            raise ValueError('query row groups must contain 1/2 active dual-branch rows '
                             f'and sum to batch={batch}, got {group_sizes}')
        _dual_lora_ctx['query_row_group_sizes'] = group_sizes
    if query_full_attention_groups is not None:
        groups = tuple(query_full_attention_groups)
        if query_row_group_sizes is None:
            raise ValueError('full-attention query groups require row-group sizes')
        if len(groups) != len(group_sizes):
            raise ValueError('full-attention/query-row group count mismatch: '
                             f'{len(groups)} != {len(group_sizes)}')
        expected_start = 0
        for group, size in zip(groups, group_sizes):
            row_start, row_end = group['row_range']
            if (int(row_start), int(row_end)) != (expected_start, expected_start + size):
                raise ValueError('full-attention query rows must be contiguous and match '
                                 f'row groups: got {(row_start, row_end)}, expected '
                                 f'{(expected_start, expected_start + size)}')
            expected_start += size
        _dual_lora_ctx['query_full_attention_groups'] = groups
    # Training always supplies a one-sample linear-fork context, even before
    # the first B/A span. Besides recurrent segmentation, that context makes
    # causal-conv receive an all-zero ``seq_idx``. Cached single-token updates
    # use the cache update kernel, so a local Q-length context is also safe for
    # the real-batch branch decoder.
    from swift.template.step_mask.linear_fork import set_linear_fork_ctx
    set_linear_fork_ctx(
        frame_starts=[],
        b_spans=[],
        a_spans=[],
        seq_len=seq_len,
        device=device,
    )


def set_dual_route(route: str, batch: int, seq_len: int, device) -> None:
    """Set one LoRA route for every row of an inference forward."""
    set_dual_routes([route] * batch, seq_len, device)


def _query_row_group_sizes(branches) -> tuple[int, ...]:
    """Return contiguous active B/A row counts for each query."""
    query_indices = [int(branch.get('query_index', 0)) for branch in branches]
    if not query_indices:
        raise ValueError('branches must not be empty')
    groups = []
    previous = query_indices[0]
    count = 0
    seen = set()
    for query_index in query_indices:
        if query_index != previous:
            if query_index in seen or query_index < previous:
                raise ValueError('active query rows must stay contiguous and ordered: '
                                 f'{query_indices}')
            seen.add(previous)
            groups.append(count)
            previous = query_index
            count = 0
        count += 1
    groups.append(count)
    if any(size not in (1, 2) for size in groups):
        raise ValueError(f'each active query must have one or two branch rows: {groups}')
    return tuple(groups)


def patch_qwen3_5_query_group_rmsnorm(model) -> int:
    """Keep decode RMSNorm reductions at canonical per-query row shapes.

    PyTorch's eager Qwen3.5 RMSNorm reduction is not bit-invariant to the
    number of leading rows for every BF16 input.  True query batching can
    therefore change one rounded element even when a query's own inputs are
    identical, after which later transformer layers amplify the difference.

    During fused query decode, active rows remain contiguous by query and
    each query owns either both B/A rows or its one unfinished branch.  Run
    the unmodified module once per such 1/2-row group and concatenate the
    results.  A single-query forward takes the original path unchanged.
    Training and shared multimodal prefills never set the grouping context.
    """
    patched = 0
    for module in model.modules():
        if type(module).__name__ != 'Qwen3_5RMSNorm':
            continue
        if getattr(module, '_mss_query_group_patched', False):
            continue
        original_forward = module.forward

        def grouped_forward(this, hidden_states, _original=original_forward):
            group_sizes = _dual_lora_ctx.get('query_row_group_sizes')
            if not group_sizes or len(group_sizes) == 1:
                return _original(hidden_states)
            if int(hidden_states.shape[0]) != sum(group_sizes):
                raise RuntimeError('RMSNorm/query-group row mismatch: '
                                   f'input={tuple(hidden_states.shape)} groups={group_sizes}')
            outputs = []
            start = 0
            for size in group_sizes:
                end = start + size
                outputs.append(_original(hidden_states[start:end]))
                start = end
            return torch.cat(outputs, dim=0)

        module.forward = types.MethodType(grouped_forward, module)
        module._mss_query_group_patched = True
        module._mss_original_forward = original_forward
        patched += 1
    return patched


def patch_qwen3_5_query_group_shape_sensitive_projections(model) -> int:
    """Keep the two shape-sensitive language projection families exact."""
    patched = 0
    for module in model.modules():
        module_type = type(module).__name__
        if module_type == 'Qwen3_5MLP':
            projection = module.down_proj
            base = getattr(projection, 'base', projection)
        elif module_type in ('Qwen3_5Attention', 'Qwen3_5GatedDeltaNet'):
            base = module.o_proj if module_type == 'Qwen3_5Attention' \
                else module.out_proj
        else:
            continue
        if not isinstance(base, torch.nn.Linear):
            raise RuntimeError('shape-sensitive Qwen3.5 projection has no Linear base')
        if getattr(base, '_mss_query_group_patched', False):
            continue
        original_forward = base.forward

        def grouped_forward(this, hidden_states, _original=original_forward):
            group_sizes = _dual_lora_ctx.get('query_row_group_sizes')
            if not group_sizes or len(group_sizes) == 1:
                return _original(hidden_states)
            if int(hidden_states.shape[0]) != sum(group_sizes):
                raise RuntimeError('shape-sensitive projection/query-group row mismatch: '
                                   f'input={tuple(hidden_states.shape)} '
                                   f'groups={group_sizes}')
            outputs = []
            start = 0
            for size in group_sizes:
                end = start + size
                outputs.append(_original(hidden_states[start:end]))
                start = end
            return torch.cat(outputs, dim=0)

        base.forward = types.MethodType(grouped_forward, base)
        base._mss_query_group_patched = True
        base._mss_original_forward = original_forward
        patched += 1
    return patched


_QUERY_GROUP_FLEX_PATCH = {'installed': False, 'original': None}


def _query_group_flex_forward(original_forward, module, query, key, value, attention_mask, *args, **kwargs):
    """Dispatch one packed full-attention call in canonical query groups."""
    groups = _dual_lora_ctx.get('query_full_attention_groups')
    if not groups:
        return original_forward(module, query, key, value, attention_mask, *args, **kwargs)
    if len(groups) == 1 and groups[0]['uses_all_kv_slots']:
        # The active rows now have the canonical one-query shape and no
        # padding holes. Avoid an unnecessary K/V gather while preserving
        # exactly the same interface call as independent decoding.
        return original_forward(module, query, key, value, attention_mask, *args, **kwargs)

    batch = int(query.shape[0])
    if int(key.shape[0]) != batch or int(value.shape[0]) != batch:
        raise RuntimeError('query-group attention batch mismatch: '
                           f'q={tuple(query.shape)} k={tuple(key.shape)} '
                           f'v={tuple(value.shape)}')

    outputs = []
    lses = []
    saw_lse = None
    for group in groups:
        row_start, row_end = map(int, group['row_range'])
        slot_rows = group['kv_slot_rows'].to(device=key.device)
        group_size = row_end - row_start
        if tuple(slot_rows.shape[:1]) != (group_size, ):
            raise RuntimeError('query-group K/V slot rows do not match batch rows: '
                               f'slots={tuple(slot_rows.shape)} rows='
                               f'{(row_start, row_end)}')
        if slot_rows.ndim != 2 or int(slot_rows.shape[1]) == 0:
            raise RuntimeError('query-group K/V slots must be non-empty [B, L], got '
                               f'{tuple(slot_rows.shape)}')

        key_rows = key[row_start:row_end]
        value_rows = value[row_start:row_end]
        key_indices = slot_rows[:, None, :, None].expand(
            group_size,
            int(key_rows.shape[1]),
            int(slot_rows.shape[1]),
            int(key_rows.shape[3]),
        )
        value_indices = slot_rows[:, None, :, None].expand(
            group_size,
            int(value_rows.shape[1]),
            int(slot_rows.shape[1]),
            int(value_rows.shape[3]),
        )
        compact_key = torch.gather(key_rows, dim=2, index=key_indices).contiguous()
        compact_value = torch.gather(value_rows, dim=2, index=value_indices).contiguous()
        output, lse = original_forward(
            module,
            query[row_start:row_end],
            compact_key,
            compact_value,
            group['attention_mask'],
            *args,
            **kwargs,
        )
        outputs.append(output)
        has_lse = lse is not None
        if saw_lse is None:
            saw_lse = has_lse
        elif saw_lse != has_lse:
            raise RuntimeError('query-group FlexAttention returned inconsistent LSEs')
        if has_lse:
            lses.append(lse)

    output = torch.cat(outputs, dim=0)
    lse = torch.cat(lses, dim=0) if saw_lse else None
    return output, lse


def patch_qwen3_5_query_group_flex_attention() -> bool:
    """Keep full attention numerically identical to per-query decoding.

    A packed query batch can contain padding holes because its independent
    prefixes have different lengths.  Although the batched BlockMask rejects
    those holes exactly, FlexAttention's BF16 reduction is not bit-invariant
    to the leading batch size / padded KV layout.  The resulting one-element
    drift can be amplified by later layers and change a near-tied greedy
    token.

    During query-batched branch decode only, run the *attention kernel* once
    per query using that query's compact physical K/V sequence and its own
    dual-branch BlockMask.  QKV projections and cache updates remain fused;
    the shape-sensitive output/down projections are grouped separately by
    ``patch_qwen3_5_query_group_shape_sensitive_projections``.  The packed
    cache is read but never mutated here.  Single-query inference and training
    take the registered Transformers implementation unchanged.
    """
    if _QUERY_GROUP_FLEX_PATCH['installed']:
        return False

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original_forward = ALL_ATTENTION_FUNCTIONS.get_interface('flex_attention', None)
    if original_forward is None:
        raise RuntimeError('Transformers has no flex_attention interface')

    def grouped_flex_forward(module, query, key, value, attention_mask, *args, **kwargs):
        return _query_group_flex_forward(
            original_forward,
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            **kwargs,
        )

    grouped_flex_forward._mss_query_group_flex = True
    ALL_ATTENTION_FUNCTIONS['flex_attention'] = grouped_flex_forward
    _QUERY_GROUP_FLEX_PATCH.update(
        installed=True,
        original=original_forward,
    )
    return True


def _build_query_full_attention_groups(branches, group_sizes, q_pos, kv_rows, valid_after):
    """Build compact, training-equivalent masks for each active query."""
    groups = []
    row_start = 0
    for size in group_sizes:
        row_end = row_start + size
        group_branches = branches[row_start:row_end]
        kept_lengths = [len(branch['kept']) for branch in group_branches]
        if len(set(kept_lengths)) != 1:
            raise RuntimeError('active branches of one query have different K/V lengths: '
                               f'{kept_lengths}')
        slot_rows = torch.stack(
            [torch.nonzero(valid_after[row], as_tuple=False).flatten() for row in range(row_start, row_end)], dim=0)
        if int(slot_rows.shape[1]) != kept_lengths[0]:
            raise RuntimeError('query-group compact K/V length mismatch: '
                               f'slots={slot_rows.shape[1]} kept={kept_lengths[0]}')
        compact_kv_rows = torch.gather(kv_rows[row_start:row_end], dim=1, index=slot_rows)
        group_mask = _build_query_group_block_mask(
            [branch['state'] for branch in group_branches],
            q_pos[row_start:row_end],
            compact_kv_rows,
            num_heads=1,
        )
        groups.append({
            'row_range': (row_start, row_end),
            'kv_slot_rows': slot_rows,
            'attention_mask': group_mask,
            'uses_all_kv_slots': (kept_lengths[0] == int(valid_after.shape[1])),
        })
        row_start = row_end
    if row_start != len(branches):
        raise RuntimeError(f'query groups cover {row_start} rows, expected {len(branches)}')
    return tuple(groups)


def _inference_flex_kernel_options(q_len: int) -> dict:
    """Use FlexAttention's fast decode kernel only for Q=1 when enabled."""
    fast_decode = (int(q_len) == 1 and os.environ.get('MSS_FAST_DECODE', '0') == '1')
    return {
        'BLOCK_M': 32,
        'BLOCK_N': 32,
        'FORCE_USE_FLEX_ATTENTION': not fast_decode,
    }


def classify_response(resp: str) -> str:
    if '<silent>' in resp:
        return 'SILENT'
    parts = []
    if '<inner>' in resp:
        parts.append('INNER')
    if '<speak>' in resp:
        parts.append('SPEAK')
    return '+'.join(parts) if parts else 'UNKNOWN'


def extract_speak(resp: str) -> Optional[str]:
    """Return text inside <speak>...</speak> if present, else None."""
    if '<speak>' not in resp:
        return None
    s = resp.find('<speak>') + len('<speak>')
    e = resp.find('</speak>', s)
    if e < 0:
        return resp[s:].strip()
    return resp[s:e].strip()


class CachedImageProcessor:
    """Transparent per-image cache around an HF image_processor.

    Swift's ``Qwen3_5Template._encode`` calls
    ``processor.image_processor(images=mm_data, return_tensors='pt',
    do_resize=False)`` on each frame with ``mm_data`` being the FULL list of
    PIL images accumulated so far. Without a cache this is O(N) per frame ->
    O(N²) per query (PVQA's 360-frame queries make this dominant).

    We wrap the processor: split the batched call into per-image lookups,
    serve hits from cache, compute misses one image at a time, then concat.

    Cache key: **content-based** = ``sha256(PIL.tobytes())`` (plus size+mode
    in the dict to disambiguate). We cannot use ``id(PIL)`` because swift's
    upstream ``fetch_image`` rebuilds a fresh PIL on every encode call (see
    qwen.py:_add_default_tags) -> id() is unstable. Content hash is stable
    and gives O(N) (not O(N²)) total image_processor calls per query.

    Per-image processing is deterministic (per-image min/max_pixels, no
    cross-image normalization), so cache returns are byte-exact equivalent
    to the original batched call. Verified by ``verify_cache.py``.

    Toggle ``enabled=False`` to bypass the cache (for verification).
    """

    def __init__(self, inner):
        self._inner = inner
        self._cache = {}
        self.stats = {'hits': 0, 'misses': 0}
        self.enabled = True

    @staticmethod
    def _key(img):
        """Content-derived key. ~1ms for a 360x640 RGB frame (tobytes+sha256);
        amortised over a 50-200ms image_processor call this is negligible."""
        import hashlib
        return (img.size, img.mode, hashlib.sha256(img.tobytes()).digest())

    def __call__(self, images=None, **kwargs):
        if not self.enabled or images is None:
            return self._inner(images=images, **kwargs)
        single = not isinstance(images, list)
        img_list = [images] if single else images

        per_image_outputs = []
        for img in img_list:
            key = self._key(img)
            cached = self._cache.get(key)
            if cached is not None:
                per_image_outputs.append(cached)
                self.stats['hits'] += 1
            else:
                out = self._inner(images=[img], **kwargs)
                self._cache[key] = out
                per_image_outputs.append(out)
                self.stats['misses'] += 1

        if single:
            return per_image_outputs[0]

        # Concat batched outputs across images (each key is a stackable tensor).
        result = {}
        for k in per_image_outputs[0].keys():
            result[k] = torch.cat([o[k] for o in per_image_outputs], dim=0)
        try:
            from transformers.feature_extraction_utils import BatchFeature
            return BatchFeature(data=result, tensor_type='pt')
        except ImportError:
            return result

    def __getattr__(self, name):
        # Delegate any other attribute (patch_size, merge_size, ...).
        return getattr(self._inner, name)

    def clear(self):
        self._cache.clear()
        self.stats['hits'] = 0
        self.stats['misses'] = 0


def _inputs_embeds_with_precomputed_image(
    model,
    input_ids: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    """Repeat Qwen3.5's native image-placeholder replacement exactly."""
    qwen_model = getattr(model, 'model', None)
    if qwen_model is None or not hasattr(qwen_model, 'get_placeholder_mask'):
        raise TypeError('precomputed image features require Qwen3.5 conditional generation')
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_embeds = image_embeds.to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    image_mask, _ = qwen_model.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_embeds,
    )
    return inputs_embeds.masked_scatter(image_mask, image_embeds)


def patch_runner_prebuild_blockmask():
    """Monkey-patch ``MaskedStreamingRunner._forward_with_mask`` to pre-build
    the flex_attention ``BlockMask`` ONCE per forward and reuse it across
    all 8 full-attention layers via ``ctx['precomputed_block_mask']``.

    Why this exists
    ---------------
    The original ``_forward_with_mask`` only sets ``state``, ``q_token_positions``,
    ``kv_token_positions`` in the patch context. When each full-attention
    layer's flex_attention call enters ``patched_create_causal_mask``, it
    falls through to ``build_streaming_block_mask(...)`` and rebuilds the
    mask from scratch. ``Qwen3_5`` has 8 full-attention layers, so each
    forward triggers 8 redundant rebuilds.

    Profile (PVQA frame 20-25, 30k cumulative tokens):
      build_streaming_block_mask = ~7.4s over 148 calls (50ms each)
      → per-frame CPU cost ~400ms just from the redundant 7/8 rebuilds.

    The mask is layer-independent (visibility derives only from position,
    sequence metadata, and window size — not from anything per-layer), so
    sharing one BlockMask across all attention layers in the same forward
    is identity-equivalent to rebuilding it each time. GraphDecoder already
    uses this short-circuit (it stores a static_bm and passes via ctx);
    this patch extends the same trick to the eager (non-graph) path.

    Verified byte-exact by ``verify_cache.py`` (response strings unchanged
    across all frames).

    Idempotent.
    """
    if getattr(MaskedStreamingRunner._forward_with_mask, '_prebuild_bm_patched', False):
        return
    orig = MaskedStreamingRunner._forward_with_mask

    def patched(self,
                input_ids,
                pixel_values,
                image_grid_thw,
                q_token_positions,
                kv_token_positions,
                valid_kv_len=None,
                cache_position=None,
                position_ids=None,
                route='none',
                return_logits=True,
                image_embeds=None):
        compute_logits = (return_logits or os.environ.get('MSS_KEEP_UNUSED_LOGITS', '0') == '1')
        # Build the BlockMask ONCE; layer-independent under the step+window
        # visibility rule. All 8 full-attention layers reuse via ctx
        # short-circuit (see patch.py:265-267).
        bm = build_streaming_block_mask(
            state=self.state,
            q_token_positions=q_token_positions,
            kv_token_positions=kv_token_positions,
            num_heads=1,
            valid_kv_len=valid_kv_len,
        )
        token = streaming_ctx_token(
            self.state,
            q_token_positions,
            kv_token_positions,
            valid_kv_len=valid_kv_len,
            precomputed_block_mask=bm,
        )
        try:
            set_dual_route(route, input_ids.size(0), input_ids.size(1), input_ids.device)
            kwargs = dict(
                past_key_values=self.past_kv,
                use_cache=True,
                # A zero-length tensor makes HF run the transformer/cache
                # update but apply lm_head to an empty sequence. This is safe
                # for shared/trailer/terminal-control forwards whose logits
                # are not consumed; capture/oracle paths keep the old value 1.
                logits_to_keep=(1 if compute_logits else torch.empty(0, device=input_ids.device, dtype=torch.long)),
                cache_position=q_token_positions if cache_position is None else cache_position,
                kernel_options=_inference_flex_kernel_options(input_ids.size(1)),
            )
            if image_embeds is None:
                kwargs['input_ids'] = input_ids
            else:
                if pixel_values is not None:
                    raise ValueError('pixel_values and precomputed image_embeds are mutually exclusive')
                kwargs['inputs_embeds'] = _inputs_embeds_with_precomputed_image(
                    self.model,
                    input_ids,
                    image_embeds,
                )
            if position_ids is not None:
                kwargs['position_ids'] = position_ids
            if pixel_values is not None:
                kwargs['pixel_values'] = pixel_values
            if image_grid_thw is not None:
                kwargs['image_grid_thw'] = image_grid_thw
            out = self.model(**kwargs)
        finally:
            clear_ctx()
            reset_ctx(token)
        # StreamingCache lives in-place on ``self.past_kv`` (update() does
        # not rebind); the legacy DynamicCache may return a new instance
        # via ``out.past_key_values`` in edge cases, so keep the rebind
        # for that path only.
        if not _is_streaming_cache(self.past_kv):
            self.past_kv = out.past_key_values
        return {'logits': out.logits}

    patched._prebuild_bm_patched = True
    patched._orig = orig
    MaskedStreamingRunner._forward_with_mask = patched


def patch_qwen3_5_cached_chunk_linear_attention(model) -> int:
    """Preserve linear-attention state for cached prefill chunks with Q>1.

    Upstream Qwen3.5 uses cached recurrent/conv state only when ``seq_len==1``.
    A later streaming frame is an 80-ish-token prefill chunk, so the stock
    path silently resets both states for that chunk.  This instance patch uses
    the stateful causal-conv update kernel and the chunk delta-rule's
    ``initial_state`` while retaining one model-level forward (full-attention
    layers still process the chunk together).
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    patched = 0
    for module in model.modules():
        if type(module).__name__ != 'Qwen3_5GatedDeltaNet':
            continue
        if getattr(module, '_mss_cached_chunk_patched', False):
            continue
        original_forward = module.forward

        def cached_chunk_forward(
            this,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
            _original=original_forward,
        ):
            batch_size, seq_len, _ = hidden_states.shape
            use_cached_chunk = (
                cache_params is not None and cache_params.has_previous_state and seq_len > 1
                and cache_position is not None)
            if not use_cached_chunk:
                return _original(
                    hidden_states=hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
            conv_state = cache_params.conv_states[this.layer_idx]
            recurrent_state = cache_params.recurrent_states[this.layer_idx]
            if conv_state is None or recurrent_state is None:
                raise RuntimeError(f'linear layer {this.layer_idx} has an incomplete cached '
                                   'state before multi-token streaming prefill')

            mixed_qkv = this.in_proj_qkv(hidden_states).transpose(1, 2)
            z = this.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, this.head_v_dim)
            b = this.in_proj_b(hidden_states)
            a = this.in_proj_a(hidden_states)

            if this.causal_conv1d_update is None:
                raise RuntimeError('cached Q>1 prefill requires causal_conv1d_update')
            # The installed update kernel accepts either one token or an
            # entire chunk and mutates Qwen's width-sized raw-input state in
            # place.  Using it here avoids the incompatible width-1/layout
            # convention of causal_conv1d_fn(initial_states=...).
            mixed_qkv = this.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                this.conv1d.weight.squeeze(1),
                this.conv1d.bias,
                this.activation,
            )
            cache_params.conv_states[this.layer_idx] = conv_state

            mixed_qkv = mixed_qkv.transpose(1, 2)
            query, key, value = torch.split(
                mixed_qkv,
                [this.key_dim, this.key_dim, this.value_dim],
                dim=-1,
            )
            query = query.reshape(batch_size, seq_len, -1, this.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, this.head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, this.head_v_dim)
            beta = b.sigmoid()
            g = -this.A_log.float().exp() * torch.nn.functional.softplus(a.float() + this.dt_bias)
            if this.num_v_heads // this.num_k_heads > 1:
                repeats = this.num_v_heads // this.num_k_heads
                query = query.repeat_interleave(repeats, dim=2)
                key = key.repeat_interleave(repeats, dim=2)

            core_attn_out, final_recurrent_state = this.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            cache_params.recurrent_states[this.layer_idx] = (final_recurrent_state)

            core_attn_out = core_attn_out.reshape(-1, this.head_v_dim)
            z = z.reshape(-1, this.head_v_dim)
            core_attn_out = this.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
            return this.out_proj(core_attn_out)

        module.forward = types.MethodType(cached_chunk_forward, module)
        module._mss_cached_chunk_patched = True
        module._mss_original_forward = original_forward
        patched += 1
    return patched


def _stream_state_clone(state, *, pending_dual_turn: bool = False):
    from swift.template.step_mask.inference import StreamingStepMaskState
    cloned = StreamingStepMaskState(
        window_frames=state.window_frames,
        assistant_role_id=state.assistant_role_id,
    )
    cloned.input_ids_np = state.input_ids_np.copy()
    cloned.n_evicted = int(state.n_evicted)
    cloned._meta_cache_len = 0
    if pending_dual_turn:
        cloned.begin_pending_dual_turn()
    return cloned


def _cache_seq_len(cache) -> int:
    if cache is None:
        return 0
    # Fixed-capacity StreamingCache tensors have shape ``max_kv_len``; only
    # ``cur_len`` is valid. Checking raw K/V shape first would mistake the
    # uninitialised tail for real history at a dual fork.
    if _is_streaming_cache(cache):
        return int(cache.cur_len)
    if hasattr(cache, 'get_seq_length'):
        try:
            return int(cache.get_seq_length())
        except Exception:
            pass
    if hasattr(cache, 'key_cache'):
        for k in cache.key_cache:
            if k is not None:
                return int(k.shape[2])
    return 0


def _qwen_mrope_model(model):
    """Return the object that owns Qwen3.5's ``get_rope_index`` helper."""
    candidates = (
        getattr(model, 'model', None),
        model,
        getattr(getattr(model, 'model', None), 'model', None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, 'get_rope_index'):
            return candidate
    return None


def _build_dual_mrope_position_ids(model, input_ids, image_grid_thw, *, start_position: int):
    """Build the exact model-facing T/H/W positions used by dual training.

    Training first asks Qwen3.5 for native multimodal positions, then applies
    a constant all-axis shift after every source-fork pair.  At inference we
    already keep the post-overlap next position, so a new shared chunk is the
    native *local* layout plus that scalar offset.  Branch text positions are
    handled separately and both start from the same scalar.

    Returns ``(position_ids, next_position)`` where ``position_ids`` is
    ``[3, 1, Q]`` and ``next_position`` is the first T/H/W text position after
    the chunk.  It is intentionally independent of physical history indices
    and DynamicCache slot indices.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError('dual production inference currently requires input_ids [1, Q], '
                         f'got {tuple(input_ids.shape)}')
    q_len = int(input_ids.shape[1])
    if q_len <= 0:
        raise ValueError('cannot build MRoPE positions for an empty chunk')

    config = getattr(model, 'config', None)
    image_token_id = getattr(config, 'image_token_id', None)
    video_token_id = getattr(config, 'video_token_id', None)
    has_image = (image_token_id is not None and bool((input_ids == int(image_token_id)).any().item()))
    has_video = (video_token_id is not None and bool((input_ids == int(video_token_id)).any().item()))

    if has_video:
        raise NotImplementedError('the proactive benchmark runner accepts image frames, not video '
                                  'placeholder groups; explicit video MRoPE is not wired here')

    if has_image:
        if image_grid_thw is None:
            raise ValueError('image tokens require image_grid_thw for exact MRoPE')
        rope_model = _qwen_mrope_model(model)
        if rope_model is None:
            raise TypeError('model does not expose Qwen3.5 get_rope_index')
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
        mm_token_type_ids[input_ids == int(image_token_id)] = 1
        local_position_ids, _ = rope_model.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=None,
        )
    else:
        local_position_ids = torch.arange(
            q_len,
            device=input_ids.device,
            dtype=torch.long,
        ).view(1, 1, q_len).expand(3, 1, q_len)

    local_position_ids = local_position_ids.to(device=input_ids.device, dtype=torch.long)
    position_ids = local_position_ids + int(start_position)
    local_advance = int(local_position_ids.max().item()) + 1
    return position_ids, int(start_position) + local_advance


def _branch_mrope_position(position: int, *, device) -> torch.Tensor:
    """Return one text position on all three Qwen3.5 MRoPE axes."""
    return _branch_mrope_positions([position], device=device)


def _branch_mrope_positions(positions, *, device) -> torch.Tensor:
    """Return per-row text positions as Qwen3.5 [3, B, 1] MRoPE."""
    values = torch.as_tensor(
        tuple(int(position) for position in positions),
        device=device,
        dtype=torch.long,
    ).view(1, -1, 1)
    return values.expand(3, -1, -1)


def _runner_forward_main_prefill(self,
                                 input_ids,
                                 pixel_values,
                                 image_grid_thw,
                                 *,
                                 image_embeds=None,
                                 route='none',
                                 return_logits=True):
    input_ids = input_ids.to(self.device)
    N = int(input_ids.shape[1])
    cache_slot_start = _cache_seq_len(self.past_kv)
    mrope_start = int(getattr(self, 'mrope_pos', 0))
    position_ids, mrope_next = _build_dual_mrope_position_ids(
        self.model,
        input_ids,
        image_grid_thw,
        start_position=mrope_start,
    )
    start_pos = self.state.extend(input_ids[0])
    self.kept_positions.extend(range(start_pos, start_pos + N))
    q_pos = torch.arange(start_pos, start_pos + N, device=self.device, dtype=torch.long)
    cache_pos = torch.arange(cache_slot_start, cache_slot_start + N, device=self.device, dtype=torch.long)
    kv_pos, valid_kv_len = self._build_kv_pos()
    out = self._forward_with_mask(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        q_token_positions=q_pos,
        kv_token_positions=kv_pos,
        valid_kv_len=valid_kv_len,
        cache_position=cache_pos,
        position_ids=position_ids,
        route=route,
        return_logits=return_logits,
        image_embeds=image_embeds,
    )
    if self.use_streaming_cache:
        self.past_kv.advance(N)
    self.mrope_pos = mrope_next
    # Compatibility/debug alias.  New code must use ``mrope_pos`` so this is
    # never confused with physical token indices or cache slots.
    self.logical_pos = mrope_next
    return out


def _runner_forward_branch_token(self,
                                 cache,
                                 branch_state,
                                 branch_kept,
                                 token_id: int,
                                 *,
                                 route: str,
                                 mrope_position: int,
                                 return_logits: bool = True):
    orig_state = self.state
    orig_kept = self.kept_positions
    orig_cache = self.past_kv
    try:
        self.state = branch_state
        self.kept_positions = branch_kept
        self.past_kv = cache

        input_ids = torch.tensor([[int(token_id)]], device=self.device, dtype=torch.long)
        cache_slot = _cache_seq_len(cache)
        physical_pos = self.state.extend([int(token_id)])
        self.kept_positions.append(physical_pos)
        q_pos = torch.tensor([physical_pos], device=self.device, dtype=torch.long)
        cache_pos = torch.tensor([cache_slot], device=self.device, dtype=torch.long)
        # Branch caches are compact DynamicCaches even when the shared main
        # cache is preallocated.  Build the exact valid KV positions instead
        # of padding to the main StreamingCache capacity.
        kv_pos = torch.tensor(self.kept_positions, device=self.device, dtype=torch.long)
        valid_kv_len = None
        out = self._forward_with_mask(
            input_ids=input_ids,
            pixel_values=None,
            image_grid_thw=None,
            q_token_positions=q_pos,
            kv_token_positions=kv_pos,
            valid_kv_len=valid_kv_len,
            cache_position=cache_pos,
            position_ids=_branch_mrope_position(mrope_position, device=self.device),
            route=route,
            return_logits=return_logits,
        )
        logits = out['logits'][:, -1, :] if return_logits else None
        return self.past_kv, logits
    finally:
        self.state = orig_state
        self.kept_positions = orig_kept
        self.past_kv = orig_cache


def _runner_decode_one_dual_branch(self,
                                   *,
                                   branch_source_id: int,
                                   route: str,
                                   include_b_separator: bool,
                                   branch_start_mrope: int,
                                   separator_ids,
                                   forced_tokens=None,
                                   capture_logits: bool = False):
    branch_cache = _cache_duplicate_batch(self.past_kv, factor=1)
    branch_state = _stream_state_clone(self.state, pending_dual_turn=True)
    branch_kept = list(self.kept_positions)

    generated = []
    extra = []
    forced_stop = False
    pending = int(branch_source_id)
    offset = 0
    forced = (None if forced_tokens is None else [int(token) for token in forced_tokens])
    forced_idx = 0
    step_logits = []
    token_logps = []
    if forced is not None:
        if not forced or forced[-1] != _IM_END:
            raise ValueError(f'forced {route} tokens must end with im_end={_IM_END}: '
                             f'{forced[-8:]}')
        if len(forced) > self.max_response_tokens + 1:
            raise ValueError(f'forced {route} sequence is longer than max_response_tokens: '
                             f'{len(forced)} > {self.max_response_tokens + 1}')

    while True:
        need_logits = capture_logits or pending != _IM_END
        branch_cache, logits = _runner_forward_branch_token(
            self,
            branch_cache,
            branch_state,
            branch_kept,
            pending,
            route=route,
            mrope_position=branch_start_mrope + offset,
            return_logits=need_logits,
        )
        offset += 1
        if capture_logits:
            step_logits.append(logits.detach().clone())

        if pending == _IM_END:
            if forced is not None and forced_idx != len(forced):
                raise ValueError(f'forced {route} sequence has unused tokens at stop: '
                                 f'{forced[forced_idx:]}')
            break
        if forced is not None:
            if forced_idx >= len(forced):
                raise ValueError(f'forced {route} sequence ended before im_end was forwarded')
            next_token = forced[forced_idx]
            forced_idx += 1
            next_logp = None
        elif len(generated) >= self.max_response_tokens:
            generated.append(_IM_END)
            token_logps.append(None)
            pending = _IM_END
            forced_stop = True
            continue
        else:
            next_token, next_logp = _sample_dual_token(self, logits[0], route=route, history=generated)
        generated.append(next_token)
        token_logps.append(next_logp)
        pending = next_token

    if include_b_separator and generated and generated[-1] == _IM_END:
        if len(separator_ids) != 1:
            raise ValueError(f'expected one-token B separator, got {separator_ids}')
        sep_id = int(separator_ids[0])
        branch_cache, sep_logits = _runner_forward_branch_token(
            self,
            branch_cache,
            branch_state,
            branch_kept,
            sep_id,
            route=route,
            mrope_position=branch_start_mrope + offset,
            return_logits=capture_logits,
        )
        offset += 1
        if capture_logits:
            step_logits.append(sep_logits.detach().clone())
        extra.append(sep_id)

    return {
        'source_id': int(branch_source_id),
        'tokens': generated,
        'extra': extra,
        'cache': branch_cache,
        'span_len': offset,
        'forced_stop': forced_stop,
        'token_logps': token_logps,
        'step_logits': step_logits,
    }


def _sample_dual_token(runner, logits_1d: torch.Tensor, *, route: str, history=None) -> tuple[int, Optional[float]]:
    """Select one branch token without changing the greedy default path.

    ``Sampler.sample(..., stream=route)`` gives A and B independent RNG
    streams, so serial, synchronous dual-branch, and ragged query scheduling
    consume identical randomness.  Third-party samplers implementing only
    the historical one-argument callable remain supported.
    """
    sampler = getattr(runner, 'sampler', None)
    if sampler is None:
        return int(logits_1d.argmax(dim=-1).item()), None
    sample = getattr(sampler, 'sample', None)
    if sample is not None:
        token, logp = sample(logits_1d, stream=route, history=list(history or ()))
    else:
        token, logp = sampler(logits_1d)
    return int(token), float(logp)


class GreedyDualBranchSampler(DualBranchSampler):
    """Deterministic inference policy on the RL dual-grammar support.

    The first action is a route-specific binary argmax with ties resolved to
    silence.  Content remains a full lexical-vocabulary argmax after masking
    template/control tokens, and structural close tokens are forced by the
    same grammar used during dual-stream RL rollout and actor replay.
    """

    def __init__(self, tokenizer, *, max_response_tokens: int):
        super().__init__(
            tokenizer,
            max_response_tokens=max_response_tokens,
            temperature=1.0,
            top_p=1.0,
            seed=None,
        )

    @torch.no_grad()
    def sample(self,
               logits_1d: torch.Tensor,
               *,
               stream: str = 'default',
               history=None) -> tuple[int, float]:
        history = list(history or ())
        policy_logits = self.policy_logits(logits_1d.detach(), stream=stream, history=history)
        if not history:
            positive_id = int(self._routes[stream]['positive'])
            token = (positive_id
                     if policy_logits[positive_id] > policy_logits[self.silent_id]
                     else self.silent_id)
        else:
            token = int(policy_logits.argmax(dim=-1).item())
        logp = torch.log_softmax(policy_logits, dim=-1)[token]
        return int(token), float(logp.item())


def _runner_forward_dual_batch_token(self, cache, branches, *, return_logits=True):
    """Forward one token for every active A/B row in a real tensor batch."""
    if not branches:
        raise ValueError('branches must contain at least one active row')

    cache_slot = _cache_seq_len(cache)
    input_ids = torch.tensor(
        [[int(branch['pending'])] for branch in branches],
        device=self.device,
        dtype=torch.long,
    )
    q_rows = []
    kv_rows = []
    mrope_positions = []
    for branch in branches:
        physical_pos = branch['state'].extend([int(branch['pending'])])
        branch['kept'].append(physical_pos)
        if len(branch['kept']) != cache_slot + 1:
            raise RuntimeError('batched branch cache/state length mismatch: '
                               f'kept={len(branch["kept"])} cache_next={cache_slot + 1}')
        q_rows.append([physical_pos])
        kv_rows.append(list(branch['kept']))
        mrope_positions.append(int(branch['branch_start_mrope']) + int(branch['offset']))

    q_pos = torch.tensor(q_rows, device=self.device, dtype=torch.long)
    kv_pos = torch.tensor(kv_rows, device=self.device, dtype=torch.long)
    block_mask = build_batched_streaming_block_mask(
        [branch['state'] for branch in branches],
        q_pos,
        kv_pos,
        num_heads=1,
    )
    token = streaming_ctx_token(
        branches[0]['state'],
        q_pos[0],
        kv_pos[0],
        precomputed_block_mask=block_mask,
    )
    compute_logits = (return_logits or os.environ.get('MSS_KEEP_UNUSED_LOGITS', '0') == '1')
    try:
        set_dual_routes(
            [branch['route'] for branch in branches],
            input_ids.size(1),
            input_ids.device,
            query_row_group_sizes=_query_row_group_sizes(branches),
        )
        out = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=(1 if compute_logits else torch.empty(0, device=input_ids.device, dtype=torch.long)),
            cache_position=torch.tensor([cache_slot], device=self.device, dtype=torch.long),
            position_ids=_branch_mrope_positions(mrope_positions, device=self.device),
            kernel_options=_inference_flex_kernel_options(input_ids.size(1)),
        )
    finally:
        clear_ctx()
        reset_ctx(token)
    logits = out.logits[:, -1, :] if return_logits else None
    return out.past_key_values, logits


def _runner_forward_ragged_query_branches(runner, cache, valid_mask, branches, *, return_logits=True):
    """Forward one token for branch rows with different prefix lengths."""
    if not branches:
        raise ValueError('branches must contain at least one active row')
    batch_size = len(branches)
    if int(valid_mask.shape[0]) != batch_size:
        raise ValueError(f'valid-mask batch {valid_mask.shape[0]} != {batch_size}')
    cache_width = int(valid_mask.shape[1])
    input_ids = torch.tensor(
        [[int(branch['pending'])] for branch in branches],
        device=runner.device,
        dtype=torch.long,
    )
    q_rows = []
    kv_rows = torch.zeros(
        batch_size,
        cache_width + 1,
        device=runner.device,
        dtype=torch.long,
    )
    mrope_positions = []
    for row, branch in enumerate(branches):
        old_slots = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        if len(branch['kept']) != int(old_slots.numel()):
            raise RuntimeError('ragged branch slot map mismatch before append: '
                               f'kept={len(branch["kept"])} '
                               f'valid={old_slots.numel()}')
        if old_slots.numel():
            kv_rows[row, old_slots] = torch.tensor(branch['kept'], device=runner.device, dtype=torch.long)
        physical_pos = branch['state'].extend([int(branch['pending'])])
        branch['kept'].append(physical_pos)
        q_rows.append([physical_pos])
        kv_rows[row, cache_width] = physical_pos
        mrope_positions.append(int(branch['branch_start_mrope']) + int(branch['offset']))

    q_pos = torch.tensor(q_rows, device=runner.device, dtype=torch.long)
    valid_after = torch.cat([
        valid_mask,
        torch.ones(batch_size, 1, device=runner.device, dtype=torch.bool),
    ], dim=1)
    block_mask = build_batched_streaming_block_mask(
        [branch['state'] for branch in branches],
        q_pos,
        kv_rows,
        num_heads=1,
        valid_kv_mask=valid_after,
    )
    token = streaming_ctx_token(
        branches[0]['state'],
        q_pos[0],
        kv_rows[0],
        precomputed_block_mask=block_mask,
    )
    compute_logits = (return_logits or os.environ.get('MSS_KEEP_UNUSED_LOGITS', '0') == '1')
    group_sizes = _query_row_group_sizes(branches)
    query_attention_groups = _build_query_full_attention_groups(
        branches,
        group_sizes,
        q_pos,
        kv_rows,
        valid_after,
    )
    try:
        set_dual_routes(
            [branch['route'] for branch in branches],
            input_ids.size(1),
            input_ids.device,
            query_row_group_sizes=group_sizes,
            query_full_attention_groups=query_attention_groups,
        )
        out = runner.model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=(1 if compute_logits else torch.empty(0, device=input_ids.device, dtype=torch.long)),
            cache_position=torch.tensor([cache_width], device=runner.device, dtype=torch.long),
            position_ids=_branch_mrope_positions(mrope_positions, device=runner.device),
            kernel_options=_inference_flex_kernel_options(1),
        )
    finally:
        clear_ctx()
        reset_ctx(token)
    logits = out.logits[:, -1, :] if return_logits else None
    return out.past_key_values, logits, valid_after


def _runner_decode_ragged_queries(runners,
                                  begins,
                                  *,
                                  branch_source_ids,
                                  separator_ids,
                                  forced_B_tokens=None,
                                  forced_A_tokens=None,
                                  capture_logits=False):
    """Decode every query's A/B branches in one shrinking tensor batch."""
    runners = list(runners)
    begins = list(begins)
    query_count = len(runners)
    if not query_count or len(begins) != query_count:
        raise ValueError('runners/begins must have the same non-zero length')
    if any(runner.model is not runners[0].model for runner in runners):
        raise ValueError('all query runners must share one model instance')
    if any(str(runner.device) != str(runners[0].device) for runner in runners):
        raise ValueError('all query runners must share one device')

    def per_query(value, name):
        if len(value) != query_count:
            raise ValueError(f'{name} must contain {query_count} entries')
        return list(value)

    branch_source_ids = per_query(branch_source_ids, 'branch_source_ids')
    separator_ids = per_query(separator_ids, 'separator_ids')
    forced_B_tokens = ([None]
                       * query_count if forced_B_tokens is None else per_query(forced_B_tokens, 'forced_B_tokens'))
    forced_A_tokens = ([None]
                       * query_count if forced_A_tokens is None else per_query(forced_A_tokens, 'forced_A_tokens'))

    pairs = []
    active = []
    query_caches = []
    for query_index, (runner, begin) in enumerate(zip(runners, begins)):
        branch_b = _make_dual_branch(
            runner,
            route='B',
            include_separator=True,
            branch_source_id=branch_source_ids[query_index],
            branch_start_mrope=begin['branch_start_mrope'],
            forced_tokens=forced_B_tokens[query_index],
        )
        branch_a = _make_dual_branch(
            runner,
            route='A',
            include_separator=False,
            branch_source_id=branch_source_ids[query_index],
            branch_start_mrope=begin['branch_start_mrope'],
            forced_tokens=forced_A_tokens[query_index],
        )
        for branch in (branch_b, branch_a):
            branch['query_index'] = query_index
            branch['separator_ids'] = list(separator_ids[query_index])
            active.append(branch)
        pairs.append((branch_b, branch_a))
        query_caches.append(runner.past_kv)

    if os.environ.get('MSS_QUERY_PACK_REFERENCE', '0') == '1':
        branch_cache, valid_mask = _pack_dynamic_caches(
            [_cache_duplicate_batch(cache, factor=1) for cache in query_caches for _ in range(2)])
    else:
        branch_cache, valid_mask = _pack_query_prefix_caches(query_caches, repeats=2)
    while active:
        need_logits = capture_logits or any(
            int(branch['pending']) != _IM_END and not branch['pending_is_separator'] for branch in active)
        branch_cache, logits, valid_mask = (
            _runner_forward_ragged_query_branches(
                runners[0],
                branch_cache,
                valid_mask,
                active,
                return_logits=need_logits,
            ))
        continuing = []
        for row, branch in enumerate(active):
            pending = int(branch['pending'])
            branch['offset'] += 1
            if capture_logits:
                branch['step_logits'].append(logits[row:row + 1].detach().clone())

            if branch['pending_is_separator']:
                branch['cache'] = _extract_packed_cache_row(branch_cache, row, valid_mask[row])
                continue
            if pending == _IM_END:
                forced = branch['forced']
                if forced is not None and branch['forced_idx'] != len(forced):
                    raise ValueError(f'forced {branch["route"]} sequence has unused '
                                     f'tokens: {forced[branch["forced_idx"]:]}')
                if branch['include_separator']:
                    if len(branch['separator_ids']) != 1:
                        raise ValueError('expected one-token B separator, got '
                                         f'{branch["separator_ids"]}')
                    separator = int(branch['separator_ids'][0])
                    branch['extra'].append(separator)
                    branch['pending'] = separator
                    branch['pending_is_separator'] = True
                    continuing.append((row, branch))
                else:
                    branch['cache'] = _extract_packed_cache_row(branch_cache, row, valid_mask[row])
                continue

            forced = branch['forced']
            if forced is not None:
                if branch['forced_idx'] >= len(forced):
                    raise ValueError(f'forced {branch["route"]} sequence ended before '
                                     'im_end was forwarded')
                next_token = forced[branch['forced_idx']]
                branch['forced_idx'] += 1
                next_logp = None
            elif len(branch['generated']) >= runners[branch['query_index']].max_response_tokens:
                next_token = _IM_END
                next_logp = None
                branch['forced_stop'] = True
            else:
                next_token, next_logp = _sample_dual_token(
                    runners[branch['query_index']], logits[row], route=branch['route'], history=branch['generated'])
            branch['generated'].append(next_token)
            branch['token_logps'].append(next_logp)
            branch['pending'] = next_token
            continuing.append((row, branch))

        if continuing:
            rows = [row for row, _ in continuing]
            branch_cache = _cache_select_batch_rows(branch_cache, rows)
            row_indices = torch.as_tensor(rows, device=valid_mask.device, dtype=torch.long)
            valid_mask = valid_mask.index_select(0, row_indices).contiguous()
            active = [branch for _, branch in continuing]
        else:
            active = []

    def finish(branch):
        if branch['cache'] is None:
            raise RuntimeError(f'{branch["route"]} branch has no final cache')
        return {
            'source_id': int(branch['source_id']),
            'tokens': list(branch['generated']),
            'extra': list(branch['extra']),
            'cache': branch['cache'],
            'span_len': int(branch['offset']),
            'forced_stop': bool(branch['forced_stop']),
            'token_logps': list(branch['token_logps']),
            'step_logits': list(branch['step_logits']),
        }

    return [(finish(branch_b), finish(branch_a)) for branch_b, branch_a in pairs]


def _make_dual_branch(self,
                      *,
                      route: str,
                      include_separator: bool,
                      branch_source_id: int,
                      branch_start_mrope: int,
                      forced_tokens=None):
    forced = (None if forced_tokens is None else [int(token) for token in forced_tokens])
    if forced is not None:
        if not forced or forced[-1] != _IM_END:
            raise ValueError(f'forced {route} tokens must end with im_end={_IM_END}: '
                             f'{forced[-8:]}')
        if len(forced) > self.max_response_tokens + 1:
            raise ValueError(f'forced {route} sequence is longer than '
                             f'max_response_tokens: {len(forced)} > '
                             f'{self.max_response_tokens + 1}')
    return {
        'source_id': int(branch_source_id),
        'route': route,
        'include_separator': include_separator,
        'branch_start_mrope': int(branch_start_mrope),
        'state': _stream_state_clone(self.state, pending_dual_turn=True),
        'kept': list(self.kept_positions),
        'pending': int(branch_source_id),
        'pending_is_separator': False,
        'offset': 0,
        'generated': [],
        'extra': [],
        'forced': forced,
        'forced_idx': 0,
        'forced_stop': False,
        # One entry per generated token. ``None`` marks deterministic tokens
        # supplied by a forced replay or the max-length safety close.
        'token_logps': [],
        'step_logits': [],
        'cache': None,
    }


def _runner_decode_dual_batched(self,
                                *,
                                branch_source_id: int,
                                branch_start_mrope: int,
                                separator_ids,
                                forced_B_tokens=None,
                                forced_A_tokens=None,
                                capture_logits: bool = False):
    """Decode B/A together, dropping the finished dual-branch row."""

    branch_B = _make_dual_branch(
        self,
        route='B',
        include_separator=True,
        branch_source_id=branch_source_id,
        branch_start_mrope=branch_start_mrope,
        forced_tokens=forced_B_tokens,
    )
    branch_A = _make_dual_branch(
        self,
        route='A',
        include_separator=False,
        branch_source_id=branch_source_id,
        branch_start_mrope=branch_start_mrope,
        forced_tokens=forced_A_tokens,
    )
    active = [branch_B, branch_A]
    branch_cache = _cache_duplicate_batch(self.past_kv, factor=2)

    while active:
        need_logits = capture_logits or any(
            int(branch['pending']) != _IM_END and not branch['pending_is_separator'] for branch in active)
        branch_cache, logits = _runner_forward_dual_batch_token(self, branch_cache, active, return_logits=need_logits)
        continuing = []

        for row, branch in enumerate(active):
            pending = int(branch['pending'])
            branch['offset'] += 1
            if capture_logits:
                branch['step_logits'].append(logits[row:row + 1].detach().clone())

            if branch['pending_is_separator']:
                branch['cache'] = _cache_select_batch(branch_cache, row)
                continue

            if pending == _IM_END:
                forced = branch['forced']
                if forced is not None and branch['forced_idx'] != len(forced):
                    raise ValueError(f'forced {branch["route"]} sequence has unused '
                                     f'tokens at stop: {forced[branch["forced_idx"]:]}')
                if branch['include_separator']:
                    if len(separator_ids) != 1:
                        raise ValueError('expected one-token B separator, got '
                                         f'{separator_ids}')
                    separator = int(separator_ids[0])
                    branch['extra'].append(separator)
                    branch['pending'] = separator
                    branch['pending_is_separator'] = True
                    continuing.append((row, branch))
                else:
                    branch['cache'] = _cache_select_batch(branch_cache, row)
                continue

            forced = branch['forced']
            if forced is not None:
                if branch['forced_idx'] >= len(forced):
                    raise ValueError(f'forced {branch["route"]} sequence ended before '
                                     'im_end was forwarded')
                next_token = forced[branch['forced_idx']]
                branch['forced_idx'] += 1
                next_logp = None
            elif len(branch['generated']) >= self.max_response_tokens:
                next_token = _IM_END
                next_logp = None
                branch['forced_stop'] = True
            else:
                next_token, next_logp = _sample_dual_token(
                    self, logits[row], route=branch['route'], history=branch['generated'])
            branch['generated'].append(next_token)
            branch['token_logps'].append(next_logp)
            branch['pending'] = next_token
            continuing.append((row, branch))

        if not continuing:
            active = []
            branch_cache = None
        elif len(continuing) == len(active):
            active = [branch for _, branch in continuing]
        elif len(continuing) == 1:
            row, branch = continuing[0]
            branch_cache = _cache_select_batch(branch_cache, row)
            active = [branch]
        else:
            raise RuntimeError('dual batch shrink produced an unsupported active set')

    def finish(branch):
        if branch['cache'] is None:
            raise RuntimeError(f'{branch["route"]} branch has no final cache')
        return {
            'source_id': int(branch['source_id']),
            'tokens': list(branch['generated']),
            'extra': list(branch['extra']),
            'cache': branch['cache'],
            'span_len': int(branch['offset']),
            'forced_stop': bool(branch['forced_stop']),
            'token_logps': list(branch['token_logps']),
            'step_logits': list(branch['step_logits']),
        }

    return finish(branch_B), finish(branch_A)


def _runner_begin_dual(self,
                       new_input_ids,
                       new_pixel_values,
                       new_image_grid_thw,
                       *,
                       new_image_embeds=None,
                       capture_branch_logits=False):
    shared_out = None
    if new_input_ids.numel() > 0:
        shared_out = _runner_forward_main_prefill(
            self,
            new_input_ids,
            new_pixel_values,
            new_image_grid_thw,
            image_embeds=new_image_embeds,
            route='none',
            return_logits=capture_branch_logits,
        )
    return {
        'shared_out': shared_out,
        'prefix_cache_len': _cache_seq_len(self.past_kv),
        'branch_start_mrope': int(self.mrope_pos),
    }


def _runner_finish_dual(self,
                        *,
                        branch_B,
                        branch_A,
                        begin,
                        turn_trailer_ids,
                        model_for_merge=None,
                        capture_branch_logits=False,
                        decode_mode='batched'):
    prefix_cache_len = int(begin['prefix_cache_len'])
    branch_start_mrope = int(begin['branch_start_mrope'])
    _cache_append_trail(self.past_kv, branch_B['cache'], prefix_cache_len)
    _cache_append_trail(self.past_kv, branch_A['cache'], prefix_cache_len)
    _merge_linear_state_from_branches(
        self.past_kv,
        cache_A=branch_A['cache'],
        cache_B=branch_B['cache'],
        model=model_for_merge or self.model,
        mode=os.environ.get('MSS_MERGE_MODE', 'additive'),
    )

    physical_branch_ids = ([int(branch_B['source_id'])] + list(branch_B['tokens']) + list(branch_B['extra'])
                           + [int(branch_A['source_id'])] + list(branch_A['tokens']))
    expected_step_close = None
    physical_start = None
    if physical_branch_ids:
        start_pos = self.state.extend(physical_branch_ids)
        physical_start = int(start_pos)
        self.kept_positions.extend(range(start_pos, start_pos + len(physical_branch_ids)))
        expected_step_close = start_pos + len(physical_branch_ids) - 1

    self.mrope_pos = branch_start_mrope + max(
        int(branch_B['span_len']),
        int(branch_A['span_len']),
    )
    self.logical_pos = self.mrope_pos

    trailer = np.asarray(turn_trailer_ids, dtype=np.int64)
    trailer_out = None
    if trailer.size:
        trailer_ids = torch.tensor(trailer, dtype=torch.long).unsqueeze(0)
        trailer_out = _runner_forward_main_prefill(
            self,
            trailer_ids,
            None,
            None,
            route='none',
            return_logits=capture_branch_logits,
        )

    if expected_step_close is not None:
        actual_step_close = self.state.last_step_close_pos()
        if actual_step_close != expected_step_close:
            self.state._refresh_meta()
            close_positions = np.flatnonzero(self.state._step_close)[-8:].tolist()
            physical_end = physical_start + len(physical_branch_ids)
            physical_suffix = self.state.input_ids_np[physical_start:physical_end + len(trailer)].tolist()
            raise RuntimeError('merged dual turn must close exactly at final A im_end: '
                               f'actual={actual_step_close} expected={expected_step_close} '
                               f'close_positions={close_positions} physical_start={physical_start} '
                               f'B={branch_B["tokens"]} B_extra={branch_B["extra"]} '
                               f'A={branch_A["tokens"]} physical_suffix={physical_suffix}')

    if self.evict:
        self._evict_unused_kv()

    self._frame_idx += 1
    result = {
        'decode_mode': decode_mode,
        'B_tokens': list(branch_B['tokens']),
        'A_tokens': list(branch_A['tokens']),
        'B_extra': list(branch_B['extra']),
        'B_forced_stop': bool(branch_B['forced_stop']),
        'A_forced_stop': bool(branch_A['forced_stop']),
        'B_text': self.tokenizer.decode(branch_B['tokens'], skip_special_tokens=False),
        'A_text': self.tokenizer.decode(branch_A['tokens'], skip_special_tokens=False),
        # Aligned one-for-one with B_tokens/A_tokens. None means the token
        # was deterministically supplied (forced replay or safety close).
        'B_token_logps': list(branch_B.get('token_logps', [None] * len(branch_B['tokens']))),
        'A_token_logps': list(branch_A.get('token_logps', [None] * len(branch_A['tokens']))),
    }
    if physical_start is not None:
        b_token_start = physical_start + 1
        a_source_pos = (b_token_start + len(branch_B['tokens']) + len(branch_B['extra']))
        a_token_start = a_source_pos + 1
        for route, branch, token_start in (
            ('B', branch_B, b_token_start),
            ('A', branch_A, a_token_start),
        ):
            token_logps = list(branch.get('token_logps', [None] * len(branch['tokens'])))
            if len(token_logps) != len(branch['tokens']):
                raise RuntimeError(f'{route} token/logp length mismatch: '
                                   f'{len(branch["tokens"])} != {len(token_logps)}')
            result[f'{route}_token_positions'] = list(range(token_start, token_start + len(branch['tokens'])))
            sampled = [(int(token), token_start + offset, float(logp))
                       for offset, (token, logp) in enumerate(zip(branch['tokens'], token_logps)) if logp is not None]
            result[f'{route}_sampled_tokens'] = [x[0] for x in sampled]
            result[f'{route}_sampled_positions'] = [x[1] for x in sampled]
            result[f'{route}_sampled_logps'] = [x[2] for x in sampled]
    if capture_branch_logits:
        shared_out = begin['shared_out']
        result.update({
            'shared_logits': (None if shared_out is None else shared_out['logits'].detach()),
            'B_step_logits': branch_B['step_logits'],
            'A_step_logits': branch_A['step_logits'],
            'trailer_logits': (None if trailer_out is None else trailer_out['logits'].detach()),
        })
    return result


@torch.no_grad()
def _prefill_then_decode_dual(self,
                              new_input_ids: torch.Tensor,
                              new_pixel_values: Optional[torch.Tensor],
                              new_image_grid_thw: Optional[torch.Tensor],
                              *,
                              branch_source_id: int,
                              branch_separator_ids,
                              turn_trailer_ids,
                              model_for_merge=None,
                              forced_B_tokens=None,
                              forced_A_tokens=None,
                              new_image_embeds: Optional[torch.Tensor] = None,
                              capture_branch_logits: bool = False) -> dict:
    """Dual-branch variant of MaskedStreamingRunner.prefill_then_decode.

    ``new_input_ids`` must end immediately before the terminal non-thinking
    prefix token P. This method forwards P_B through LoRA_B and P_A through
    LoRA_A from the same pre-P state, then appends the physical source-fork
    layout to the main cache/state.
    """
    if not hasattr(self, 'mrope_pos'):
        if self.state.total_len:
            raise RuntimeError('cannot reconstruct multimodal MRoPE from an existing dual '
                               'history; reset the runner before first dual decode')
        self.mrope_pos = 0
        self.logical_pos = 0

    begin = _runner_begin_dual(
        self,
        new_input_ids,
        new_pixel_values,
        new_image_grid_thw,
        new_image_embeds=new_image_embeds,
        capture_branch_logits=capture_branch_logits,
    )
    branch_start_mrope = int(begin['branch_start_mrope'])

    decode_mode = os.environ.get('MSS_DUAL_DECODE_MODE', 'batched').strip().lower()
    if decode_mode == 'batched':
        branch_B, branch_A = _runner_decode_dual_batched(
            self,
            branch_source_id=branch_source_id,
            branch_start_mrope=branch_start_mrope,
            separator_ids=branch_separator_ids,
            forced_B_tokens=forced_B_tokens,
            forced_A_tokens=forced_A_tokens,
            capture_logits=capture_branch_logits,
        )
    elif decode_mode == 'serial':
        branch_B = _runner_decode_one_dual_branch(
            self,
            branch_source_id=branch_source_id,
            route='B',
            include_b_separator=True,
            branch_start_mrope=branch_start_mrope,
            separator_ids=branch_separator_ids,
            forced_tokens=forced_B_tokens,
            capture_logits=capture_branch_logits,
        )
        branch_A = _runner_decode_one_dual_branch(
            self,
            branch_source_id=branch_source_id,
            route='A',
            include_b_separator=False,
            branch_start_mrope=branch_start_mrope,
            separator_ids=branch_separator_ids,
            forced_tokens=forced_A_tokens,
            capture_logits=capture_branch_logits,
        )
    else:
        raise ValueError('MSS_DUAL_DECODE_MODE must be batched|serial, got '
                         f'{decode_mode!r}')

    return _runner_finish_dual(
        self,
        branch_B=branch_B,
        branch_A=branch_A,
        begin=begin,
        turn_trailer_ids=turn_trailer_ids,
        model_for_merge=model_for_merge,
        capture_branch_logits=capture_branch_logits,
        decode_mode=decode_mode,
    )


def install_dual_methods_on_masked_runner():
    if getattr(MaskedStreamingRunner, '_dual_methods_installed', False):
        return
    MaskedStreamingRunner.prefill_then_decode_dual = _prefill_then_decode_dual
    MaskedStreamingRunner._dual_methods_installed = True


def patch_pil_deepcopy_passthrough():
    """Make ``PIL.Image.Image.__deepcopy__`` return self instead of cloning.

    Why this exists
    ---------------
    Swift's ``StdTemplateInputs.from_dict`` (template_inputs.py:58) does
    ``inputs = deepcopy(inputs)`` for every encode call. Since ``inputs`` is
    a dict containing the ``images`` list of PIL Images, every encode
    produces a FRESH PIL clone per image — original ``id()`` is lost,
    and our per-PIL fetch_image cache (which stores the cached fetch_image
    result as an attribute on the PIL itself) never sees a hit because
    the cloned PIL has no cached attribute.

    Cost of the deepcopy itself: ~1-5 ms per PIL of typical frame size
    (memcpy of pixel data + Image object construction). For a streaming
    query with N images this is O(N) per encode → O(N²) total.

    Fix: implement ``__deepcopy__`` to return ``self``. Safe in our
    pipeline because:
    * Frames are appended to ``ProactiveStreamRunner.all_images`` and
      never mutated.
    * Swift's downstream code reads PIL pixel data but does not modify it
      (rescale_image checks size and returns same PIL if under max_pixels;
      otherwise it creates a fresh resized PIL — orthogonal to our
      deepcopy short-circuit).

    Side benefit: also speeds up the bulk of swift's _preprocess_inputs
    that pays the deepcopy memcpy cost per encode.

    Idempotent.
    """
    from PIL import Image
    if getattr(Image.Image, '_passthrough_deepcopy_installed', False):
        return

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    Image.Image.__deepcopy__ = __deepcopy__
    Image.Image._passthrough_deepcopy_installed = True


def patch_swift_save_pil_image_passthrough():
    """Monkey-patch ``Template._save_pil_image`` to return the PIL unchanged.

    Why this exists
    ---------------
    Swift's default ``Template._save_pil_image`` (base.py:698) serializes
    every PIL into a SHA256-named PNG on tmp disk so that downstream
    consumers see a path string (picklable across DataLoader workers in
    distributed training). It's invoked in ``_preprocess_inputs``
    (base.py:305-308) for every image at every encode when
    ``load_images_origin=False`` (which Qwen3_5Template inherits because
    ``load_images = False`` on the class).

    Cost per call: ``image.tobytes()`` (memcpy ~700KB for a 640×480 RGB
    PIL) + ``hashlib.sha256(...)`` + optional ``image.save(path)``. For a
    streaming query of N accumulated frames this is O(N) per encode →
    O(N²) total per query. ~10-50ms/image/encode.

    Side effect: by converting PIL → path, it BREAKS the per-PIL cache
    attribute path that ``install_fetch_image_cache`` relies on.
    Downstream ``fetch_image`` then receives a fresh str each encode and
    has no stable identity to cache on.

    Fix: replace with a passthrough that returns the PIL as-is. Downstream
    ``fetch_image`` accepts both PIL and str (see qwen_vl_utils.fetch_image
    signature), so PIL flows through unchanged. PIL identity is preserved
    → our cache attr persists → fetch_image cache hits as intended.

    Safe because: (1) we run a single Python process (no DataLoader
    pickling), (2) inference mode (not training: no caching to disk
    needed across epochs).

    Idempotent.
    """
    from swift.template import base as _tpl_base
    if getattr(_tpl_base.Template._save_pil_image, '_is_passthrough', False):
        return _tpl_base.Template._save_pil_image
    orig = _tpl_base.Template._save_pil_image

    @staticmethod
    def passthrough(image):
        return image

    passthrough._is_passthrough = True
    passthrough._orig = orig
    _tpl_base.Template._save_pil_image = passthrough
    return passthrough


def install_fetch_image_cache():
    """Monkey-patch ``qwen_vl_utils.fetch_image`` with a per-PIL cache.

    Why this exists
    ---------------
    Swift's ``Qwen3VLTemplate._add_default_tags`` (templates/qwen.py:315)
    calls ``fetch_image({'image': inputs.images[index]}, image_patch_size=N)``
    for EVERY image at EVERY ``template.encode`` invocation. ``fetch_image``
    does PIL ``smart_resize`` to clamp into ``[min_pixels, max_pixels]`` and
    returns a NEW PIL each time. For a streaming query with N accumulated
    frames this is O(N) per encode → O(N²) total over the query. CachedImage-
    Processor (further downstream) already amortises the actual image-to-
    tensor processing, but fetch_image's PIL resize stays uncached and
    dominates per-frame CPU time in PVQA's 360-frame queries.

    Cache strategy
    --------------
    Attach the cached result as an attribute on the input PIL itself
    (``img._cached_fetch_image_result``). PIL.Image instances allow
    arbitrary attribute assignment (no ``__slots__``), so this gives:
      * O(1) lookup via ``hasattr``
      * automatic cleanup — when the PIL is GC'd (e.g., the runner's
        ``all_images`` list goes out of scope at query end), the cached
        attribute goes with it. No separate dict to manage.
      * works even though PIL.Image is unhashable (no WeakKeyDictionary
        / dict-key issues).

    Determinism
    -----------
    ``fetch_image`` is per-PIL deterministic (smart_resize uses only that
    PIL's size + the constant ``image_patch_size`` kwarg), so cached return
    is byte-equivalent to recomputing. We ignore kwargs because in our
    pipeline ``image_patch_size`` is fixed at process startup.

    Idempotent. Returns the installed wrapper (with ``._stats`` for
    inspection — note: no central ``._cache`` since storage is per-PIL).
    """
    import qwen_vl_utils
    if getattr(qwen_vl_utils.fetch_image, '_is_cached_wrapper', False):
        return qwen_vl_utils.fetch_image  # already patched
    orig = qwen_vl_utils.fetch_image
    stats = {'hits': 0, 'misses': 0}
    CACHE_ATTR = '_cached_fetch_image_result'

    def cached_fetch_image(arg, **kwargs):
        # Swift calls as fetch_image({'image': PIL}, image_patch_size=N).
        img = None
        if isinstance(arg, dict):
            img = arg.get('image')
        elif isinstance(arg, Image.Image):
            img = arg
        if isinstance(img, Image.Image):
            cached_result = getattr(img, CACHE_ATTR, None)
            if cached_result is not None:
                stats['hits'] += 1
                return cached_result
            result = orig(arg, **kwargs)
            try:
                setattr(img, CACHE_ATTR, result)
            except (AttributeError, TypeError):
                pass  # PIL subclass with __slots__; uncached miss
            stats['misses'] += 1
            return result
        return orig(arg, **kwargs)

    cached_fetch_image._is_cached_wrapper = True
    cached_fetch_image._stats = stats
    cached_fetch_image._orig = orig
    qwen_vl_utils.fetch_image = cached_fetch_image
    return cached_fetch_image


class IncrementalEncoder:
    """Bypass swift Template's O(N) per-frame chat re-tokenization by
    constructing the per-tick delta tokens directly.

    Background: ``runner.step()`` originally calls
    ``self.sm.template.encode({'messages': all, 'images': all})`` every
    frame. swift's ``_swift_encode`` walks the entire message history each
    call (base.py:1169), so encode_ms grows linearly with ``tick`` -> total
    encode cost is O(N²) per query. Measured: ~14 ms at tick 0, ~500 ms at
    tick 350 (1.4 s/frame just for encoding by frame 400).

    The fix relies on a structural observation (verified token-by-token in
    ``dump_encode_tokens.py``):

      For every tick >= 1, the encoded full_ids gain a FIXED suffix
      relative to the previous tick's state -- i.e. close prev assistant
      turn + open new user turn with the new image + assistant prefix +
      non_think_prefix. The only per-frame variable is the count of
      <|image_pad|> tokens, which equals
      ``image_grid_thw[0] * image_grid_thw[1] * image_grid_thw[2]
      // spatial_merge_size**2``.

    This encoder precomputes the constant chunks at construction and
    splices them with the new image's pad block at each tick. byte-exact
    vs the original path verified by ``verify_encode_increment.py``.
    """

    # Token ids hardcoded from Qwen3.5 (dump_encode_tokens.py confirmed).
    IM_START = 248045
    IM_END = 248046
    NEWLINE = 198
    VISION_START = 248053
    VISION_END = 248054
    IMAGE_PAD = 248056

    def __init__(self, tokenizer, spatial_merge_size: int = 2):
        nl = self.NEWLINE
        # <|im_end|>\n -- closes any (user/assistant) turn.
        self.chat_sep = np.array([self.IM_END, nl], dtype=np.int64)
        # <|im_start|>user\n
        user_tok = tokenizer.encode('user', add_special_tokens=False)
        asst_tok = tokenizer.encode('assistant', add_special_tokens=False)
        assert len(user_tok) == 1 and len(asst_tok) == 1, (
            f'expected single-token roles, got user={user_tok}, asst={asst_tok}')
        self.user_open = np.array([self.IM_START, user_tok[0], nl], dtype=np.int64)
        # <|im_start|>assistant\n
        self.asst_open = np.array([self.IM_START, asst_tok[0], nl], dtype=np.int64)
        # <|vision_start|> ... <|vision_end|>
        self.vision_start_arr = np.array([self.VISION_START], dtype=np.int64)
        self.vision_end_arr = np.array([self.VISION_END], dtype=np.int64)
        # <think>\n\n</think>\n\n (suppresses model thinking on this turn)
        self.non_think_prefix = np.asarray(
            tokenizer.encode('<think>\n\n</think>\n\n', add_special_tokens=False),
            dtype=np.int64,
        )
        self.spatial_merge_size = int(spatial_merge_size)

    def n_image_tokens(self, image_grid_thw) -> int:
        """Compute the number of <|image_pad|> tokens for ONE image given its
        grid_thw triple (T, H, W)."""
        if hasattr(image_grid_thw, 'view'):
            thw = image_grid_thw.view(-1).tolist()
        else:
            thw = list(np.asarray(image_grid_thw).flatten())
        prod = int(thw[0]) * int(thw[1]) * int(thw[2])
        return prod // (self.spatial_merge_size**2)

    def build_delta(self, image_grid_thw_for_new_image) -> np.ndarray:
        """Build the token delta to extend ``state.input_ids_np`` for a
        subsequent-frame (tick >= 1) prefill.

        Layout (matches ``dump_encode_tokens.py`` ground truth):
          chat_sep        [<|im_end|>, \\n]               -- close prev assistant
          user_open       [<|im_start|>, 'user', \\n]      -- open new user turn
          vision_start    [<|vision_start|>]
          image_pads      [<|image_pad|>] * n_image_tokens
          vision_end      [<|vision_end|>]
          chat_sep        [<|im_end|>, \\n]               -- close user turn
          asst_open       [<|im_start|>, 'assistant', \\n] -- open new assistant
          non_think_pref  [<think>\\n\\n</think>\\n\\n]    -- runner-appended
        """
        n = self.n_image_tokens(image_grid_thw_for_new_image)
        image_pad_block = np.full((n, ), self.IMAGE_PAD, dtype=np.int64)
        return np.concatenate([
            self.chat_sep,
            self.user_open,
            self.vision_start_arr,
            image_pad_block,
            self.vision_end_arr,
            self.chat_sep,
            self.asst_open,
            self.non_think_prefix,
        ])

    def build_dual_delta(self, image_grid_thw_for_new_image) -> np.ndarray:
        """Build a post-turn delta for source-fork dual inference.

        Unlike the legacy decoder, dual inference has already forwarded both
        branch ``<|im_end|>`` tokens and the shared trailer newline before the
        next frame begins.  Re-emitting ``chat_sep`` at the front would insert
        a duplicate close token and diverge from the training source-fork
        layout.  The current assistant prefix still ends in the terminal P;
        ``ProactiveStreamRunner.step`` removes it from shared prefill and
        forwards P_B/P_A independently after the cache fork.
        """
        n = self.n_image_tokens(image_grid_thw_for_new_image)
        image_pad_block = np.full((n, ), self.IMAGE_PAD, dtype=np.int64)
        return np.concatenate([
            self.user_open,
            self.vision_start_arr,
            image_pad_block,
            self.vision_end_arr,
            self.chat_sep,
            self.asst_open,
            self.non_think_prefix,
        ])


def prepare_new_image_features(processor, frame, fetch_image_kwargs=None) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """Single-image (fetch_image → image_processor) flow.

    Mirrors swift Template._swift_prepare_inputs's per-image processing:
    fetch_image runs smart_resize to a patch-aligned shape, then
    image_processor(do_resize=False) extracts patch tensors.

    Returns (pixel_values, image_grid_thw) for ONE image. byte-exact
    equivalent to ``encoded['pixel_values']`` sliced at this image's
    index from a full ``template.encode`` call.

    Both the runner fast path (``ProactiveStreamRunner.step`` when
    ``STEP_FAST_ENCODE=1``) and the verification script
    (``verify_encode_increment.py``) MUST call this same function so that
    the byte-exact verification actually covers the production code path.
    """
    import qwen_vl_utils as _qvl
    kwargs = fetch_image_kwargs or {}
    resized_pil = _qvl.fetch_image({'image': frame}, **kwargs)
    out = processor.image_processor(
        images=[resized_pil],
        return_tensors='pt',
        do_resize=False,
    )
    return out['pixel_values'], out['image_grid_thw']


class StreamModel:
    """Module-scope holder for model + template + warmup state.

    Build once per process; share across queries to amortise the multi-second
    flex_attention compile.
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda:0',
        window: Optional[int] = None,
        system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
        no_system: bool = False,
        warmup: bool = True,
        cache_image_processor: bool = True,
        cache_fetch_image: bool = True,
    ):
        os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '128')
        from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

        # A single long-lived eval process sees several legitimate FlexAttention
        # specialisations: first/later multimodal prefills, branch batches of
        # 2*N rows, and shrinking active batches.  PyTorch's default limit of
        # eight is low enough that a two-row dual-branch reference followed by
        # query batching falls back to eager FlexAttention, materialising the full score
        # matrix.  Keep this inference-process-only and configurable; training
        # never constructs StreamModel.
        flex_recompile_limit = int(os.environ.get('MSS_FLEX_RECOMPILE_LIMIT', '64'))
        if flex_recompile_limit < 8:
            raise ValueError('MSS_FLEX_RECOMPILE_LIMIT must be >= 8')
        torch._dynamo.config.recompile_limit = max(  # noqa: SLF001
            int(torch._dynamo.config.recompile_limit),
            flex_recompile_limit,
        )
        torch._dynamo.config.cache_size_limit = max(  # noqa: SLF001
            int(torch._dynamo.config.cache_size_limit),
            flex_recompile_limit,
        )
        self.dual_branch = checkpoint_has_dual_weights(model_path) or dual_lora_enabled()
        self.dual_runtime_spec = {}
        if self.dual_branch:
            self.dual_runtime_spec = configure_dual_runtime(model_path)
            contract_window = int(self.dual_runtime_spec['step_mask_window'])
            if window is None:
                window = contract_window
            elif int(window) != contract_window:
                raise RuntimeError(f'inference window={window} conflicts with checkpoint '
                                   f'step_mask_window={contract_window}')
            os.environ.setdefault('MSS_MERGE_MODE', 'additive')
            os.environ.setdefault('MSS_POSITION_ID_OVERLAP', '1')
            if system_prompt_path == DEFAULT_SYSTEM_PROMPT_PATH and Path(DUAL_SYSTEM_PROMPT_PATH).exists():
                system_prompt_path = DUAL_SYSTEM_PROMPT_PATH
            if not no_system:
                assert_dual_system_prompt(model_path, system_prompt_path, self.dual_runtime_spec)
        elif window is None:
            # Preserve the historical single-branch inference default.
            window = 30
        window = int(window)
        if window <= 0:
            raise ValueError('window must be a positive integer')

        enable_flex_attn_support()
        config = AutoConfig.from_pretrained(model_path)
        text_cfg = getattr(config, 'text_config', config)
        text_cfg._attn_implementation = 'flex_attention'
        config._attn_implementation = 'flex_attention'

        print(f'[StreamModel] loading {model_path} on {device} '
              f'(window={window}, dual={self.dual_branch})')

        ModelCls = Qwen3_5ForConditionalGeneration
        if self.dual_branch:
            asst_probe = get_processor(model_path).tokenizer.encode('assistant', add_special_tokens=False)
            if len(asst_probe) != 1:
                raise ValueError(f"'assistant' encoded to: {asst_probe}")
            asst_probe_id = int(asst_probe[0])

            class DualQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):

                def __init__(self, config, *args, **kwargs):
                    super().__init__(config, *args, **kwargs)
                    enable_dual_lora(self, assistant_role_id=asst_probe_id)

            ModelCls = DualQwen3_5ForConditionalGeneration

        loaded = ModelCls.from_pretrained(
            model_path,
            config=config,
            dtype=torch.bfloat16,
            device_map=device,
            output_loading_info=True,
        )
        if isinstance(loaded, tuple):
            self.model, loading_info = loaded
        else:
            self.model, loading_info = loaded, {}
        assert_dual_weights_loaded(loading_info, expected=self.dual_branch)
        if self.dual_branch:
            assert_dual_merge_policy(self.model, self.dual_runtime_spec)
            n_grouped_rmsnorm = patch_qwen3_5_query_group_rmsnorm(self.model)
            if n_grouped_rmsnorm == 0:
                raise RuntimeError('dual model has no Qwen3.5 RMSNorm modules to patch')
            n_grouped_projections = (patch_qwen3_5_query_group_shape_sensitive_projections(self.model))
            expected_projection_count = 2 * int(text_cfg.num_hidden_layers)
            if n_grouped_projections != expected_projection_count:
                raise RuntimeError('query-group projection patch count mismatch: '
                                   f'{n_grouped_projections} != {expected_projection_count}')

            n_cached_chunk = patch_qwen3_5_cached_chunk_linear_attention(self.model)
            if n_cached_chunk == 0:
                raise RuntimeError('dual model has no Qwen3.5 linear-attention layers to '
                                   'patch for cached multi-token prefill')
            print('[dual] runtime contract: ' + json.dumps(self.dual_runtime_spec, sort_keys=True), flush=True)
            print(
                f'[dual] cached Q>1 linear-attention patch: '
                f'{n_cached_chunk} layers',
                flush=True,
            )
            print(
                '[dual] query-group shape-stable RMSNorm patch: '
                f'{n_grouped_rmsnorm} modules',
                flush=True,
            )
            print(
                '[dual] query-group shape-stable BF16 projections: '
                f'{n_grouped_projections} modules',
                flush=True,
            )
        self.model.eval()

        # Install fetch_image cache (upstream of image_processor) BEFORE
        # any template.encode call so swift's per-image fetch_image is
        # cached too. Also disable swift's _save_pil_image PIL→disk
        # serialisation; otherwise the cache key (PIL identity) is
        # destroyed and per-encode sha256(image.tobytes()) becomes the
        # dominant CPU cost. See respective docstrings for rationale.
        if cache_fetch_image:
            # 3 layers of patches, all needed for the fetch_image cache to
            # actually hit:
            #  1. PIL.__deepcopy__ → self (swift from_dict deepcopies inputs
            #     including PILs every encode → otherwise identity is lost).
            #  2. Template._save_pil_image → passthrough (swift converts PIL
            #     to disk-path string for distributed dataloader, breaking
            #     PIL-identity AND adding per-encode sha256 cost).
            #  3. install fetch_image cache itself, keyed on PIL attr.
            patch_pil_deepcopy_passthrough()
            patch_swift_save_pil_image_passthrough()
            self._fetch_image_cache = install_fetch_image_cache()
        else:
            self._fetch_image_cache = None

        processor = get_processor(model_path)
        # Wrap image_processor with per-image cache. Content-hash key (PIL
        # rebuilt by fetch_image breaks id()-based caching). With fetch_image
        # cache also enabled, image_processor sees the SAME PIL ref every
        # encode → hashing is still safe but mostly redundant (kept for
        # defense-in-depth when fetch_image cache is off).
        if cache_image_processor:
            processor.image_processor = CachedImageProcessor(processor.image_processor)
        self.processor = processor
        self.tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

        asst_ids = self.tokenizer.encode('assistant', add_special_tokens=False)
        assert len(asst_ids) == 1, f"'assistant' encoded to: {asst_ids}"
        self.assistant_role_id = int(asst_ids[0])
        self.dual_source_prefix_ids = np.asarray(
            self.tokenizer.encode(NON_THINKING_PREFIX, add_special_tokens=False),
            dtype=np.int64,
        )
        if len(self.dual_source_prefix_ids) < 2:
            raise ValueError('non-thinking prefix tokenization is unexpectedly short')
        self.dual_branch_source_id = int(self.dual_source_prefix_ids[-1])
        self.dual_branch_separator_ids = np.asarray(
            self.tokenizer.encode(DUAL_BRANCH_SEPARATOR, add_special_tokens=False),
            dtype=np.int64,
        )
        self.dual_turn_trailer_ids = np.asarray(
            self.tokenizer.encode(DUAL_TURN_TRAILER, add_special_tokens=False),
            dtype=np.int64,
        )
        self.window = window
        self.device = device

        patch_qwen3_5_for_step_mask(
            self.model,
            window_frames=window,
            assistant_role_id=self.assistant_role_id,
        )
        query_group_flex_installed = (patch_qwen3_5_query_group_flex_attention())
        if self.dual_branch:
            print(
                '[dual] query-group compact full-attention patch: ' +
                ('installed' if query_group_flex_installed else 'active'),
                flush=True,
            )

        # Make MaskedStreamingRunner._forward_with_mask pre-build BlockMask
        # once per forward and reuse across all 8 full-attn layers (saves
        # ~400ms/frame on long-context steps; see docstring for details).
        patch_runner_prebuild_blockmask()
        install_dual_methods_on_masked_runner()

        self.no_system = no_system
        self.system_prompt = (
            None if no_system
            # Training passes the prompt file verbatim, including its final
            # newline.  Preserve that byte-for-byte here: stripping it changes
            # the token immediately before <|im_end|> and creates a one-token
            # train/inference mismatch.
            else Path(system_prompt_path).read_text(encoding='utf-8'))

        # Inference does not consume labels, but the slow encode path runs the
        # template in train mode so source-fork can reconstruct historical
        # physical turns.  Contract/oracle code also reads those labels.
        # Recover the checkpoint's real SFT settings instead of silently using
        # Template defaults (notably ``default`` would supervise the fixed
        # empty-think prefix that production masked with
        # ``ignore_empty_think``).
        self.template_loss_scale = str(self.dual_runtime_spec.get('loss_scale', 'default'))
        self.template_add_non_thinking_prefix = bool(self.dual_runtime_spec.get('add_non_thinking_prefix', True))
        if self.dual_branch and not self.template_add_non_thinking_prefix:
            raise RuntimeError('dual inference requires the checkpoint training contract '
                               'add_non_thinking_prefix=true')
        self.template = get_template(
            processor,
            default_system=self.system_prompt,
            template_type='qwen3_5',
            use_chat_template=True,
            enable_thinking=False,
            response_prefix='',
            add_non_thinking_prefix=self.template_add_non_thinking_prefix,
            loss_scale=self.template_loss_scale,
        )
        self.template.set_mode('train')

        # Manual non-thinking prefix (training-time only, see infer_stream_masked).
        self.non_think_prefix_ids = np.asarray(
            self.tokenizer.encode('<think>\n\n</think>\n\n', add_special_tokens=False),
            dtype=np.int64,
        )

        # Incremental encoder: bypass swift Template's O(N) full re-tokenization
        # for tick >= 1 by splicing precomputed constant token chunks with the
        # new image's image_pad block. See IncrementalEncoder docstring; toggle
        # via env var STEP_FAST_ENCODE=0 (default 1 = on).
        spatial_merge_size = getattr(
            self.processor.image_processor,
            'merge_size',
            getattr(self.processor.image_processor, 'spatial_merge_size', 2),
        )
        self.inc_encoder = IncrementalEncoder(
            self.tokenizer,
            spatial_merge_size=spatial_merge_size,
        )

        # Pre-cache the kwargs for the per-image processor call so the fast
        # path doesn't recompute them every frame.
        _ip_patch_size = getattr(self.processor.image_processor, 'patch_size', None)
        self._fetch_image_kwargs = ({'image_patch_size': _ip_patch_size} if _ip_patch_size is not None else {})

        # Pre-trigger flex_attention compilation at the Q_LEN buckets used by
        # decode, later-frame prefill, and first-frame prefill.  This block was
        # previously below an unconditional return in
        # ``prepare_new_image_features`` and therefore never ran.
        if warmup:
            warmup_flex_attention(
                self.model,
                window_frames=window,
                assistant_role_id=self.assistant_role_id,
                device=device,
            )

    @torch.no_grad()
    def prepare_new_image_features(self, frame) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Thin method-style wrapper over the module-level helper, threading
        the cached fetch_image kwargs. See the module-level
        ``prepare_new_image_features`` docstring for the full contract."""
        return prepare_new_image_features(
            self.processor,
            frame,
            fetch_image_kwargs=self._fetch_image_kwargs,
        )

    @torch.no_grad()
    def prepare_image_features_for_reuse(self, frame: Image.Image) -> PreparedImageFeatures:
        """Prepare one frame once for repeated, independent RL trajectories.

        The visual tower is frozen during dual GRPO.  Its eval-mode output is
        therefore a deterministic function of the frame and may be shared;
        language embeddings, caches, masks, MRoPE, and A/B states remain
        trajectory-local.
        """
        pixel_values, image_grid_thw = self.prepare_new_image_features(frame)
        pixel_values_device = pixel_values.to(device=self.device, dtype=torch.bfloat16)
        image_grid_thw_device = image_grid_thw.to(device=self.device, dtype=torch.long)
        image_outputs = self.model.get_image_features(
            pixel_values=pixel_values_device,
            image_grid_thw=image_grid_thw_device,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).detach()
        return PreparedImageFeatures(
            source_frame=frame,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_embeds=image_embeds,
        )


class FramePrefetcher:
    """Background thread that reads + pre-warms next frame's CPU caches
    while the main thread runs the previous frame's GPU forward.

    What it overlaps
    ----------------
    Per-frame CPU work (idle while GPU is busy):
      * VideoFrameReader.read_frame  (~30-80 ms via decord)
      * fetch_image  (PIL smart_resize, ~30-100 ms first time)
      * image_processor  (PIL → tensor patches, ~50-200 ms first time)

    All three populate caches keyed on the PIL identity / content. When
    the main thread's ``runner.step(frame)`` then runs ``template.encode``
    over the full message history, these caches hit immediately for the
    new frame instead of paying the cold-miss cost in-line with GPU work.

    Concurrency notes
    -----------------
    * Worker is a single threading.Thread (Python). PIL / numpy / decord
      release the GIL inside their C extensions, so the prefetch can
      genuinely overlap with PyTorch's GPU kernels (which also release
      the GIL during cudaLaunch).
    * Queue size 1 means we keep AT MOST one frame ahead. Bigger means
      more memory per query (PIL ~700 KB each) without much extra
      overlap.
    * Caches written by worker, read by main thread. No same-PIL race:
      worker only writes cache for frame N+1; main thread reads cache
      entries for frames 0..N inside its current encode call (frame N+1
      is appended to ``all_images`` only AFTER step(frame) starts on the
      main thread, so worker's write completes before main thread's
      read).
    """

    SENTINEL = (None, None)

    def __init__(self, video_path: str, sm: 'StreamModel', start_t: int, end_t: int):
        import queue
        import threading

        from examples.infer.proactive_eval.video_reader import VideoFrameReader
        self.vr = VideoFrameReader(video_path)
        self.sm = sm
        self.start_t = int(start_t)
        self.end_t = int(end_t)
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._exc: Optional[BaseException] = None

        def worker():
            try:
                import qwen_vl_utils
                patch_size = getattr(sm.processor.image_processor, 'patch_size', None)
                ip_kwargs = ({'image_patch_size': patch_size} if patch_size is not None else {})
                for t in range(self.start_t, self.end_t + 1):
                    if self._stop.is_set():
                        break
                    frame = self.vr.read_frame(float(t))
                    # Pre-warm fetch_image cache (populates PIL attr).
                    # Tolerate if cache wrapper isn't installed.
                    try:
                        qwen_vl_utils.fetch_image({'image': frame}, **ip_kwargs)
                    except Exception:
                        pass
                    # Pre-warm image_processor cache (populates content-hash dict).
                    try:
                        sm.processor.image_processor(images=[frame], return_tensors='pt', do_resize=False)
                    except Exception:
                        pass
                    self.queue.put((t, frame))
            except BaseException as e:
                self._exc = e
            finally:
                self.queue.put(self.SENTINEL)

        self._thread = threading.Thread(target=worker, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        # Bounded wait: a single frame should take at most a few seconds
        # even on heavily contended GPU + cold caches. 60s is generous;
        # exceeding it means the worker is hung (decord deadlock, GIL
        # starvation, etc.) and we should fail fast rather than block the
        # main eval forever.
        try:
            t, frame = self.queue.get(timeout=60.0)
        except Exception as e:
            # queue.Empty after timeout
            raise RuntimeError(f'FramePrefetcher worker did not produce next frame within '
                               f'60s (worker_alive={self._thread.is_alive()}, '
                               f'worker_exc={self._exc!r})') from e
        if t is None:
            # Either sentinel (end-of-range) or error in worker.
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return t, frame

    def close(self):
        self._stop.set()
        # Drain queue to unblock worker if it's mid-put.
        try:
            while True:
                self.queue.get_nowait()
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.vr.close()
        except Exception:
            pass


class ProactiveStreamRunner:
    """One-query streaming wrapper.

    Build with a query + start time; call ``step(frame_image)`` per second of
    video to advance one tick. Returns ``{mode, response, speak, elapsed}``.

    Re-uses a shared :class:`MaskedStreamingRunner` instance held on the
    :class:`StreamModel` across queries with the same config. Tearing down the
    runner between queries triggers ``CUDA error: illegal memory access`` in
    the streaming_cache + CUDA-graph path: the captured graph binds to
    specific device addresses inside ``StreamingCache``; once that cache is
    GC'd the mempool reuses those addresses, and the next replay reads
    garbage. Reusing the cache + graph keeps every captured pointer valid
    forever; we just reset bookkeeping (state, kept_positions, cur_len) per
    query, and zero conv/recurrent_states in place so the new query doesn't
    inherit the previous query's recurrent context.
    """

    def __init__(
        self,
        sm: StreamModel,
        query: str,
        *,
        evict: bool = True,
        max_tokens: int = 512,
        # SC (StreamingCache) is on by default: verified byte-exact vs LEG
        # under the full optimization stack (verify_sc.py: ALL 60 frames
        # identical). Adds ~1.1x speedup over LEG due to in-place K/V write
        # (avoids torch.cat O(kv) memcpy per layer per forward).
        use_streaming_cache: bool = True,
        # 16384 slots covers ~80 SILENT-heavy frames or ~30 INNER-heavy
        # frames after window=30 eviction. Bump if StreamingCache overflow
        # is hit on a long INNER/SPEAK-heavy query.
        max_kv_len: int = 16384,
        # Graph paths stay off: GraphDecoder has known numerical drift
        # (see GD root-cause investigation; not fixed yet).
        use_graph_decode: bool = False,
        use_graph_evict: bool = False,
        share_runner: bool = True,
        sampler=None,
    ):
        self.sm = sm
        self.query = query
        if (sampler is None and sm.dual_branch
                and sm.dual_runtime_spec.get('requires_constrained_decode') is True):
            sampler = GreedyDualBranchSampler(
                sm.tokenizer,
                max_response_tokens=max_tokens,
            )
        if sm.dual_branch:
            # Dual decode uses a preallocated cache only for the shared main
            # trajectory.  Each A/B fork is converted to a compact independent
            # DynamicCache, then B/A KV tails are appended in physical order
            # and linear states are additively merged back into the main cache.
            # CUDA graphs remain disabled until that hybrid fork is capturable.
            use_graph_decode = False
            use_graph_evict = False

        # Config tuple identifies a compatible runner; cache + graph can be
        # shared across queries iff the config matches.
        config_key = (evict, max_tokens, use_streaming_cache, max_kv_len, use_graph_decode, use_graph_evict)
        existing = (getattr(sm, '_shared_runner', None) if share_runner else None)
        existing_key = (getattr(sm, '_shared_runner_config', None) if share_runner else None)

        if existing is None or existing_key != config_key:
            # First query OR config mismatch → build a fresh runner.
            if existing is not None and share_runner:
                self._release_runner(existing)
                sm._shared_runner = None
            runner = MaskedStreamingRunner(
                model=sm.model,
                tokenizer=sm.tokenizer,
                window_frames=sm.window,
                assistant_role_id=sm.assistant_role_id,
                device=sm.device,
                evict=evict,
                max_response_tokens=max_tokens,
                use_streaming_cache=use_streaming_cache,
                max_kv_len=max_kv_len,
                use_graph_decode=use_graph_decode,
                use_graph_evict=use_graph_evict,
                sampler=sampler,
            )
            if share_runner:
                sm._shared_runner = runner
                sm._shared_runner_config = config_key
            runner.mrope_pos = 0
            runner.logical_pos = 0
            rope_owner = getattr(sm.model, 'model', None)
            if rope_owner is not None and hasattr(rope_owner, 'rope_deltas'):
                rope_owner.rope_deltas = None
        else:
            # Same config → reset state in place, reuse cache + captured graph.
            self._reset_runner_state(existing)
            existing.sampler = sampler
            runner = existing
        self.runner = runner
        if not hasattr(self.runner, 'mrope_pos'):
            self.runner.mrope_pos = 0
        if not hasattr(self.runner, 'logical_pos'):
            self.runner.logical_pos = 0

        self.messages: list = []
        if sm.system_prompt is not None:
            self.messages.append({'role': 'system', 'content': sm.system_prompt})
        self.all_images: list = []
        self.tick_count = 0

        # Clear the per-image cache that the CachedImageProcessor wrapper
        # holds. Scope is intentionally per-query: within a single query the
        # cache turns O(N²) image_processor calls into O(N) (since swift's
        # _encode re-passes the full accumulated images list each tick); but
        # across queries the images differ, so retaining the cache leaks
        # memory. A 360-frame query stores ~360 × ~400KB = ~140MB of tensor
        # patches; after ~15-20 queries on a single shard that's enough to
        # OOM-kill the worker (observed in pvqa_ego cluster runs).
        ip = getattr(sm.processor, 'image_processor', None)
        if ip is not None and hasattr(ip, 'clear'):
            ip.clear()

    @staticmethod
    def _reset_runner_state(runner) -> None:
        """Reset a shared MaskedStreamingRunner to begin a new query.

        Keeps the captured CUDA graph + ``StreamingCache`` buffers alive
        (their device pointers stay valid). Only resets the per-query
        bookkeeping that drives mask generation + sampling:

        * ``state.input_ids_np`` → empty (cumulative token sequence)
        * ``state.n_evicted`` → 0 (physical eviction count)
        * ``state._meta_cache_len`` → 0 (invalidates derived metadata cache)
        * ``kept_positions`` → [] (original positions occupying cache slots)
        * StreamingCache: ``cur_len`` → 0, ``cur_len_tensor`` → 0; linear-
          attention conv/recurrent states are ``zero_()``-d in place so the
          new query doesn't inherit recurrent context. K/V buffers are left
          dirty; ``mask_mod`` rejects ``kv_idx >= cur_len`` so they're
          invisible.
        * Legacy DynamicCache: rebuilt (no graph deps; cheap).
        """
        runner.state.input_ids_np = np.zeros(0, dtype=np.int64)
        runner.state.n_evicted = 0
        runner.state._meta_cache_len = 0
        runner.kept_positions = []
        runner._frame_idx = 0
        runner.mrope_pos = 0
        runner.logical_pos = 0
        rope_owner = getattr(runner.model, 'model', None)
        if rope_owner is not None and hasattr(rope_owner, 'rope_deltas'):
            rope_owner.rope_deltas = None
        if runner.use_streaming_cache:
            runner.past_kv.cur_len = 0
            if runner.past_kv.cur_len_tensor is not None:
                runner.past_kv.cur_len_tensor.fill_(0)
            for li in range(len(runner.past_kv.conv_states)):
                t = runner.past_kv.conv_states[li]
                if t is not None:
                    t.zero_()
            for li in range(len(runner.past_kv.recurrent_states)):
                t = runner.past_kv.recurrent_states[li]
                if t is not None:
                    t.zero_()
        else:
            text_cfg = getattr(runner.model.config, 'text_config', runner.model.config)
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
            runner.past_kv = Qwen3_5DynamicCache(config=text_cfg)

    @staticmethod
    def _release_runner(runner) -> None:
        """Best-effort release before swapping in a runner with different config."""
        if getattr(runner, 'graph_decoder', None) is not None:
            runner.graph_decoder.invalidate()
            runner.graph_decoder = None
        runner.eviction_graph = None
        runner.past_kv = None

    def _prepare_dual_step(
        self,
        frame: Image.Image,
        *,
        prepared_image_features: Optional[PreparedImageFeatures] = None,
    ) -> dict:
        """Encode one frame up to (but excluding) the fork source token."""
        if not self.sm.dual_branch:
            raise RuntimeError('_prepare_dual_step requires a dual checkpoint')
        if (prepared_image_features is not None
                and prepared_image_features.source_frame is not frame):
            raise ValueError('prepared image features do not belong to this frame object')
        t_start = time.time()
        self.all_images.append(frame)
        content = (f'Query: {self.query}<image>' if self.tick_count == 0 else '<image>')
        self.messages.append({'role': 'user', 'content': content})

        probe_on = os.environ.get('STEP_ENCODE_PROBE', '0') == '1'
        probe_every = int(os.environ.get('STEP_ENCODE_PROBE_EVERY', '20'))
        fast_encode = (os.environ.get('STEP_FAST_ENCODE', '1') != '0' and self.tick_count >= 1)
        encode_start = time.perf_counter() if probe_on else None
        if fast_encode:
            if prepared_image_features is None:
                new_pv_cpu, new_gt_cpu = self.sm.prepare_new_image_features(frame)
            else:
                new_pv_cpu = prepared_image_features.pixel_values
                new_gt_cpu = prepared_image_features.image_grid_thw
            delta_ids = self.sm.inc_encoder.build_dual_delta(new_gt_cpu[0])
            full_token_count = -1
        else:
            encoded = self.sm.template.encode({'messages': self.messages, 'images': self.all_images})
            full_ids = np.asarray(encoded['input_ids'], dtype=np.int64)
            full_ids = np.concatenate([full_ids, self.sm.non_think_prefix_ids])
            full_token_count = len(full_ids)
            state_ids = self.runner.state.input_ids_np
            common = common_prefix_len(state_ids, full_ids)
            if common != int(state_ids.shape[0]):
                raise RuntimeError('full template history diverges from the physical '
                                   f'inference history at position {common} '
                                   f'(state_len={state_ids.shape[0]}, '
                                   f'encoded_len={len(full_ids)}); refusing to append to '
                                   'a cache whose prefix no longer matches')
            delta_ids = full_ids[common:]
            pv = encoded.get('pixel_values')
            gt = encoded.get('image_grid_thw')
            if pv is None or gt is None:
                new_pv_cpu = new_gt_cpu = None
            else:
                new_pv_cpu, new_gt_cpu = slice_pixel_values_for_image(pv, gt, image_index=self.tick_count)
            if prepared_image_features is not None:
                if new_pv_cpu is None or new_gt_cpu is None:
                    raise RuntimeError('template omitted image tensors for a prepared frame')
                if (not torch.equal(new_pv_cpu, prepared_image_features.pixel_values)
                        or not torch.equal(new_gt_cpu, prepared_image_features.image_grid_thw)):
                    raise RuntimeError('prepared image features disagree with template encoding')
                new_pv_cpu = prepared_image_features.pixel_values
                new_gt_cpu = prepared_image_features.image_grid_thw

        if probe_on:
            encode_ms = (time.perf_counter() - encode_start) * 1000
            if self.tick_count % probe_every == 0 or self.tick_count < 3:
                print(
                    f'  [encode-probe] tick={self.tick_count:4d}  '
                    f'fast={fast_encode}  delta_tokens={len(delta_ids):4d}  '
                    f'full_tokens={full_token_count:6d}  '
                    f'encode_ms={encode_ms:7.1f}',
                    flush=True,
                )

        if delta_ids.size == 0:
            self.tick_count += 1
            return {
                'skip_result': {
                    'mode': 'SKIP',
                    'response': '',
                    'speak': None,
                    'elapsed': time.time() - t_start,
                },
            }
        if int(delta_ids[-1]) != self.sm.dual_branch_source_id:
            raise RuntimeError('dual inference expected delta to end at terminal '
                               'non-thinking prefix token '
                               f'P={self.sm.dual_branch_source_id}; '
                               f'got tail={delta_ids[-8:].tolist()}')

        if new_pv_cpu is None or new_gt_cpu is None:
            new_pv = new_gt = new_image_embeds = None
        else:
            new_gt = new_gt_cpu.to(device=self.sm.device, dtype=torch.long)
            if prepared_image_features is None:
                new_pv = new_pv_cpu.to(device=self.sm.device, dtype=torch.bfloat16)
                new_image_embeds = None
            else:
                new_pv = None
                new_image_embeds = prepared_image_features.image_embeds
        return {
            't_start': t_start,
            'pre_branch_input_ids': torch.tensor(delta_ids[:-1], dtype=torch.long).unsqueeze(0),
            'new_pv': new_pv,
            'new_gt': new_gt,
            'new_image_embeds': new_image_embeds,
        }

    def _finalize_dual_step(self, dual: dict, *, t_start: float, include_branch_logits: bool = False) -> dict:
        b_text = dual['B_text']
        a_text = dual['A_text']
        if a_text.endswith('<|im_end|>'):
            a_content = a_text[:-len('<|im_end|>')]
        else:
            a_content = a_text
        clean = b_text + ''.join(
            self.sm.tokenizer.decode([int(token)], skip_special_tokens=False) for token in dual['B_extra']) + a_content
        self.messages.append({'role': 'assistant', 'content': clean})
        self.tick_count += 1
        result = {
            'mode': classify_response(a_content.strip()),
            'response': clean,
            'speak': extract_speak(a_content),
            'elapsed': time.time() - t_start,
            'decode_mode': dual['decode_mode'],
            'B_token_count': len(dual['B_tokens']),
            'A_token_count': len(dual['A_tokens']),
            'B_tokens': list(dual['B_tokens']),
            'A_tokens': list(dual['A_tokens']),
            'internal_mode': classify_response(b_text.strip()),
            'internal_response': b_text,
            'external_response': a_content,
            'B_forced_stop': bool(dual['B_forced_stop']),
            'A_forced_stop': bool(dual['A_forced_stop']),
        }
        for route in ('B', 'A'):
            for suffix in ('token_logps', 'token_positions', 'sampled_tokens', 'sampled_positions', 'sampled_logps'):
                key = f'{route}_{suffix}'
                if key in dual:
                    result[key] = list(dual[key])
        if include_branch_logits:
            # Diagnostic-only: step_logits[i] is the distribution used to
            # select generated token i.  On an exact BF16 tie, CUDA argmax and
            # a later CPU argmax of the copy may name different maxima.  Extra
            # terminal entries may exist for already-selected im_end/B-
            # separator forwards and have no matching generated token.  Copy
            # to CPU so the diagnostic cannot retain full-vocabulary CUDA
            # tensors.
            result['branch_step_logits'] = {
                route: [
                    tensor.detach().to(device='cpu', dtype=torch.float32).clone()
                    for tensor in dual[f'{route}_step_logits']
                ]
                for route in ('B', 'A')
            }
        return result

    def _step_dual(self, frame: Image.Image, *, capture_branch_logits: bool = False) -> Dict:
        prepared = self._prepare_dual_step(frame)
        if 'skip_result' in prepared:
            return prepared['skip_result']
        dual = self.runner.prefill_then_decode_dual(
            prepared['pre_branch_input_ids'],
            prepared['new_pv'],
            prepared['new_gt'],
            branch_source_id=self.sm.dual_branch_source_id,
            branch_separator_ids=self.sm.dual_branch_separator_ids,
            turn_trailer_ids=self.sm.dual_turn_trailer_ids,
            model_for_merge=self.sm.model,
            capture_branch_logits=capture_branch_logits,
            new_image_embeds=prepared.get('new_image_embeds'),
        )
        return self._finalize_dual_step(
            dual,
            t_start=prepared['t_start'],
            include_branch_logits=capture_branch_logits,
        )

    def step(self, frame: Image.Image, *, capture_branch_logits: bool = False) -> Dict:
        if self.sm.dual_branch:
            return self._step_dual(frame, capture_branch_logits=capture_branch_logits)
        if capture_branch_logits:
            raise ValueError('capture_branch_logits is only supported by dual inference')
        t_start = time.time()
        self.all_images.append(frame)
        content = (f'Query: {self.query}<image>' if self.tick_count == 0 else '<image>')
        self.messages.append({'role': 'user', 'content': content})

        # Per-frame timing probe + fast-path toggle.
        #   STEP_ENCODE_PROBE=1   -> log encode_ms per N frames (off by default)
        #   STEP_FAST_ENCODE=0    -> force original full-encode path (default on)
        # The fast path bypasses swift Template's O(N) per-frame re-tokenize
        # (which is O(N²) per query, dominating CPU at high tick counts) by
        # splicing precomputed constant token chunks with the new image's
        # image_pad block. byte-exact vs full encode verified by
        # verify_encode_increment.py. Always uses full encode at tick 0 since
        # that path also produces the system prompt + Query: prefix.
        import os as _os
        _probe_on = _os.environ.get('STEP_ENCODE_PROBE', '0') == '1'
        _probe_every = int(_os.environ.get('STEP_ENCODE_PROBE_EVERY', '20'))
        _fast_encode = (_os.environ.get('STEP_FAST_ENCODE', '1') != '0' and self.tick_count >= 1)
        _enc_t0 = time.perf_counter() if _probe_on else None

        if _fast_encode:
            # ---- Fast path: route through StreamModel.prepare_new_image_features
            # (the SAME helper verify_encode_increment.py uses, so the
            # byte-exact verification actually covers this code path), then
            # splice the precomputed delta from IncrementalEncoder.
            new_pv_cpu, new_gt_cpu = self.sm.prepare_new_image_features(frame)
            if self.sm.dual_branch:
                delta_ids = self.sm.inc_encoder.build_dual_delta(new_gt_cpu[0])
            else:
                delta_ids = self.sm.inc_encoder.build_delta(new_gt_cpu[0])
            _n_tok_full = -1  # unknown without full encode
        else:
            # ---- Slow path: full template.encode + common_prefix_len. ----
            encoded = self.sm.template.encode({'messages': self.messages, 'images': self.all_images})
            full_ids = np.asarray(encoded['input_ids'], dtype=np.int64)
            full_ids = np.concatenate([full_ids, self.sm.non_think_prefix_ids])
            _n_tok_full = len(full_ids)
            state_ids = self.runner.state.input_ids_np
            common = common_prefix_len(state_ids, full_ids)
            if common != int(state_ids.shape[0]):
                raise RuntimeError('full template history diverges from the physical '
                                   f'inference history at position {common} '
                                   f'(state_len={state_ids.shape[0]}, '
                                   f'encoded_len={len(full_ids)}); refusing to append to '
                                   'a cache whose prefix no longer matches')
            delta_ids = full_ids[common:]
            pv = encoded.get('pixel_values')
            gt = encoded.get('image_grid_thw')
            if pv is None or gt is None:
                new_pv_cpu = new_gt_cpu = None
            else:
                new_pv_cpu, new_gt_cpu = slice_pixel_values_for_image(
                    pv,
                    gt,
                    image_index=self.tick_count,
                )

        if _probe_on:
            _enc_ms = (time.perf_counter() - _enc_t0) * 1000
            if self.tick_count % _probe_every == 0 or self.tick_count < 3:
                print(
                    f'  [encode-probe] tick={self.tick_count:4d}  '
                    f'fast={_fast_encode}  '
                    f'delta_tokens={len(delta_ids):4d}  '
                    f'full_tokens={_n_tok_full:6d}  '
                    f'encode_ms={_enc_ms:7.1f}',
                    flush=True)

        if delta_ids.size == 0:
            self.tick_count += 1
            return {
                'mode': 'SKIP',
                'response': '',
                'speak': None,
                'elapsed': time.time() - t_start,
            }

        delta_input_ids = torch.tensor(delta_ids, dtype=torch.long).unsqueeze(0)
        if new_pv_cpu is None or new_gt_cpu is None:
            new_pv = new_gt = None
        else:
            new_pv = new_pv_cpu.to(device=self.sm.device, dtype=torch.bfloat16)
            new_gt = new_gt_cpu.to(device=self.sm.device, dtype=torch.long)

        if self.sm.dual_branch:
            if delta_ids.size == 0 or int(delta_ids[-1]) != self.sm.dual_branch_source_id:
                raise RuntimeError('dual inference expected delta to end at terminal non-thinking '
                                   f'prefix token P={self.sm.dual_branch_source_id}; '
                                   f'got tail={delta_ids[-8:].tolist()}')
            pre_branch_delta = delta_ids[:-1]
            pre_branch_input_ids = torch.tensor(pre_branch_delta, dtype=torch.long).unsqueeze(0)
            dual = self.runner.prefill_then_decode_dual(
                pre_branch_input_ids,
                new_pv,
                new_gt,
                branch_source_id=self.sm.dual_branch_source_id,
                branch_separator_ids=self.sm.dual_branch_separator_ids,
                turn_trailer_ids=self.sm.dual_turn_trailer_ids,
                model_for_merge=self.sm.model,
            )
            b_text = dual['B_text']
            a_text = dual['A_text']
            if a_text.endswith('<|im_end|>'):
                a_content = a_text[:-len('<|im_end|>')]
            else:
                a_content = a_text
            clean = b_text + ''.join(
                self.sm.tokenizer.decode([int(x)], skip_special_tokens=False) for x in dual['B_extra']) + a_content
            response_for_log = clean
            mode = classify_response(a_content.strip())
            speak = extract_speak(a_content)
        else:
            raw = self.runner.prefill_then_decode(delta_input_ids, new_pv, new_gt)
            clean = raw.replace('<|im_end|>', '').strip()
            response_for_log = clean
            mode = classify_response(clean)
            speak = extract_speak(clean)
        self.messages.append({'role': 'assistant', 'content': clean})
        self.tick_count += 1

        result = {
            'mode': mode,
            'response': response_for_log,
            'speak': speak,
            'elapsed': time.time() - t_start,
        }
        if self.sm.dual_branch:
            result.update({
                'decode_mode': dual['decode_mode'],
                'B_token_count': len(dual['B_tokens']),
                'A_token_count': len(dual['A_tokens']),
                'internal_mode': classify_response(b_text.strip()),
                'internal_response': b_text,
                'external_response': a_content,
                'B_forced_stop': bool(dual['B_forced_stop']),
                'A_forced_stop': bool(dual['A_forced_stop']),
            })
        return result


class ProactiveBatchRunner:
    """Run different queries while preserving one state/cache per query.

    ``query_batch_mode='throughput'`` (the default) fuses the expensive
    autoregressive phase across queries. Shared multimodal prefills remain
    per-query; each query keeps independent state/cache/MRoPE, and arbitrary
    padded K/V holes are excluded by its exact per-row mask. RMSNorm and the
    eight full-attention layers retain canonical per-query reduction shapes,
    so fused decode is token/cache exact with independent query-bs=1 decode.
    When only one query remains active it switches directly to the canonical
    dual-branch path instead of paying ragged packing overhead.

    ``query_batch_mode='strict'`` is the diagnostic fallback: it schedules
    queries together but executes the canonical single-query path one query
    at a time. A/B remain synchronous in both modes; they are dual branches,
    not query batch entries.

    Pass ``None`` for a frame to leave that query inactive on a tick; this
    supports different video lengths without resetting or contaminating any
    other row.
    """

    def __init__(self,
                 sm: StreamModel,
                 queries,
                 *,
                 evict=True,
                 max_tokens=512,
                 use_streaming_cache=True,
                 max_kv_len=16384,
                 query_batch_mode='throughput',
                 samplers=None):
        queries = list(queries)
        if not sm.dual_branch:
            raise ValueError('different-query batching requires dual model')
        if len(queries) < 1:
            raise ValueError('queries must not be empty')
        query_batch_mode = str(query_batch_mode).strip().lower()
        if query_batch_mode not in ('strict', 'throughput'):
            raise ValueError('query_batch_mode must be strict|throughput, got '
                             f'{query_batch_mode!r}')
        self.sm = sm
        self.query_batch_mode = query_batch_mode
        if samplers is None:
            samplers = [None] * len(queries)
        else:
            samplers = list(samplers)
            if len(samplers) != len(queries):
                raise ValueError(f'samplers must contain {len(queries)} entries')
        self.streams = [
            ProactiveStreamRunner(
                sm,
                str(query),
                evict=evict,
                max_tokens=max_tokens,
                use_streaming_cache=use_streaming_cache,
                max_kv_len=max_kv_len,
                use_graph_decode=False,
                use_graph_evict=False,
                share_runner=False,
                sampler=sampler,
            ) for query, sampler in zip(queries, samplers)
        ]

    @torch.no_grad()
    def step(self,
             frames,
             *,
             forced_B_tokens=None,
             forced_A_tokens=None,
             prepared_image_features=None,
             capture_branch_logits: bool = False) -> list[Optional[dict]]:
        frames = list(frames)
        if len(frames) != len(self.streams):
            raise ValueError(f'expected {len(self.streams)} frames, got {len(frames)}')
        results: list[Optional[dict]] = [None] * len(self.streams)
        active_indices = [index for index, frame in enumerate(frames) if frame is not None]
        if not active_indices:
            return results

        def normalize_forced(value, name):
            if value is None:
                return [None] * len(self.streams)
            value = list(value)
            if len(value) != len(self.streams):
                raise ValueError(f'{name} must have {len(self.streams)} entries')
            return value

        forced_B_tokens = normalize_forced(forced_B_tokens, 'forced_B_tokens')
        forced_A_tokens = normalize_forced(forced_A_tokens, 'forced_A_tokens')
        prepared_image_features = normalize_forced(
            prepared_image_features,
            'prepared_image_features',
        )
        if any(frames[index] is None and prepared_image_features[index] is not None
               for index in range(len(frames))):
            raise ValueError('inactive batch rows must not carry prepared image features')

        def prepare_row(index):
            kwargs = {}
            if prepared_image_features[index] is not None:
                kwargs['prepared_image_features'] = prepared_image_features[index]
            return self.streams[index]._prepare_dual_step(frames[index], **kwargs)

        if self.query_batch_mode == 'strict':
            for index in active_indices:
                stream = self.streams[index]
                prepared = prepare_row(index)
                if 'skip_result' in prepared:
                    result = prepared['skip_result']
                else:
                    dual = stream.runner.prefill_then_decode_dual(
                        prepared['pre_branch_input_ids'],
                        prepared['new_pv'],
                        prepared['new_gt'],
                        branch_source_id=self.sm.dual_branch_source_id,
                        branch_separator_ids=(self.sm.dual_branch_separator_ids),
                        turn_trailer_ids=self.sm.dual_turn_trailer_ids,
                        model_for_merge=self.sm.model,
                        forced_B_tokens=forced_B_tokens[index],
                        forced_A_tokens=forced_A_tokens[index],
                        capture_branch_logits=capture_branch_logits,
                        new_image_embeds=prepared.get('new_image_embeds'),
                    )
                    result = stream._finalize_dual_step(
                        dual,
                        t_start=prepared['t_start'],
                        include_branch_logits=capture_branch_logits,
                    )
                result['query_batch_mode'] = 'strict'
                results[index] = result
            return results

        prepared_rows = []
        decode_indices = []
        for index in active_indices:
            prepared = prepare_row(index)
            if 'skip_result' in prepared:
                results[index] = prepared['skip_result']
            else:
                prepared_rows.append(prepared)
                decode_indices.append(index)
        if not decode_indices:
            return results

        # Once a ragged batch has shrunk to one active query there is nothing
        # left to fuse.  Reuse the canonical dual-branch path directly: it
        # avoids packing/compacting a one-query cache on every generated token
        # and is exactly the query-bs=1 reference implementation.
        if len(decode_indices) == 1:
            index = decode_indices[0]
            stream = self.streams[index]
            prepared = prepared_rows[0]
            dual = stream.runner.prefill_then_decode_dual(
                prepared['pre_branch_input_ids'],
                prepared['new_pv'],
                prepared['new_gt'],
                branch_source_id=self.sm.dual_branch_source_id,
                branch_separator_ids=self.sm.dual_branch_separator_ids,
                turn_trailer_ids=self.sm.dual_turn_trailer_ids,
                model_for_merge=self.sm.model,
                forced_B_tokens=forced_B_tokens[index],
                forced_A_tokens=forced_A_tokens[index],
                capture_branch_logits=capture_branch_logits,
                new_image_embeds=prepared.get('new_image_embeds'),
            )
            results[index] = stream._finalize_dual_step(
                dual,
                t_start=prepared['t_start'],
                include_branch_logits=capture_branch_logits,
            )
            results[index]['query_batch_mode'] = 'throughput'
            return results

        runners = [self.streams[index].runner for index in decode_indices]
        begins = [
            _runner_begin_dual(
                runner,
                prepared['pre_branch_input_ids'],
                prepared['new_pv'],
                prepared['new_gt'],
                capture_branch_logits=capture_branch_logits,
                new_image_embeds=prepared.get('new_image_embeds'),
            ) for runner, prepared in zip(runners, prepared_rows)
        ]
        pairs = _runner_decode_ragged_queries(
            runners,
            begins,
            branch_source_ids=[self.sm.dual_branch_source_id for _ in runners],
            separator_ids=[self.sm.dual_branch_separator_ids for _ in runners],
            forced_B_tokens=[forced_B_tokens[index] for index in decode_indices],
            forced_A_tokens=[forced_A_tokens[index] for index in decode_indices],
            capture_logits=capture_branch_logits,
        )
        for index, stream, runner, prepared, begin, pair in zip(
                decode_indices,
            (self.streams[index] for index in decode_indices),
                runners,
                prepared_rows,
                begins,
                pairs,
        ):
            branch_b, branch_a = pair
            dual = _runner_finish_dual(
                runner,
                branch_B=branch_b,
                branch_A=branch_a,
                begin=begin,
                turn_trailer_ids=self.sm.dual_turn_trailer_ids,
                model_for_merge=self.sm.model,
                capture_branch_logits=capture_branch_logits,
                decode_mode='query_batched',
            )
            results[index] = stream._finalize_dual_step(
                dual,
                t_start=prepared['t_start'],
                include_branch_logits=capture_branch_logits,
            )
            results[index]['query_batch_mode'] = 'throughput'
        return results
