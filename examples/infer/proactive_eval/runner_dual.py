"""Dual-branch inference runner (v1: correctness-focused).

Implements the 6-step per-frame flow discussed in design:
    Step 1  prefill user/frame tokens (base FFN)           -> main cache
    Step 2  prefill shared assistant prefix without P       -> state_before_P
    Step 3  fork main cache into independent A/B caches
    Step 4  decode P_B/B... with LoRA_B and P_A/A... with LoRA_A
    Step 5  merge B trail + A trail, then append shared newline
    Step 6  next frame

Scope of this v1:
    - The text path is the current correctness target. The image helper is
      kept for experiments but is not yet the final streaming-eval path.
    - Additive merge uses the loaded per-layer ``dual_merge`` gates: either
      shared ``gamma_logit`` or independent A/B ``alpha_logit`` and
      ``beta_logit`` tensors. Sequential merge is not
      implemented in this reference.
    - No streaming cache / graph decoder. Uses plain HF DynamicCache. This
      is intentionally not speed-optimised; treat as a reference.

Usage:
    MSS_DUAL_BRANCH=1 MSS_MERGE_MODE=additive \\
    CUDA_VISIBLE_DEVICES=4 \\
    python examples/infer/proactive_eval/runner_dual.py \\
        --model /path/to/base_or_ckpt \\
        --query "What color is the mug?" \\
        --n_frames 3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from examples.infer.proactive_eval.checkpoint_utils import \
    checkpoint_has_dual_weights as _checkpoint_has_dual_weights  # noqa: E402
from examples.infer.proactive_eval.checkpoint_utils import is_dual_weight_key as _is_dual_weight_key
from swift.template.step_mask import dual_lora_enabled, enable_dual_lora  # noqa: E402
from swift.template.step_mask.dual_lora import _dual_lora_ctx, clear_ctx  # noqa: E402

NON_THINKING_PREFIX = '<think>\n\n</think>\n\n'
B_BRANCH_SEPARATOR = '\n'
A_TURN_TRAILER = '\n'

# ---------------------------------------------------------------------------
# Small helpers: KV cache manipulation
# ---------------------------------------------------------------------------

_STATE_LIST_NAMES = ('key_cache', 'value_cache', 'conv_states', 'recurrent_states')


def _is_streaming_cache(cache) -> bool:
    """Return whether ``cache`` owns a fixed-capacity KV buffer."""
    return (hasattr(cache, 'cur_len') and hasattr(cache, 'max_kv_len') and callable(getattr(cache, 'advance', None)))


def _iter_state_lists(cache):
    """Yield (name, list) for every per-layer state list on the cache.
    Handles Qwen3_5DynamicCache which stores four parallel lists:
        key_cache, value_cache, conv_states, recurrent_states
    """
    for name in _STATE_LIST_NAMES:
        if hasattr(cache, name):
            lst = getattr(cache, name)
            if isinstance(lst, list):
                yield name, lst


def _empty_like_cache(cache):
    """Instantiate an empty cache of the same class as ``cache`` without
    copying any tensors. Preserves attributes like ``layer_types`` and
    ``transformer_layers`` that Qwen3_5DynamicCache relies on for dispatch,
    but leaves K/V/conv/recur lists filled with ``None`` for the caller to
    populate.

    This is ~1000x faster than :func:`copy.deepcopy` for a 32-layer cache
    with a few thousand tokens' worth of K/V, because deepcopy walks every
    tensor's storage.
    """
    cls = type(cache)
    new = cls.__new__(cls)
    # Copy over lightweight scalar attributes (layer_types, etc.) that the
    # cache methods need for dispatch. Skip the four state lists.
    for k, v in vars(cache).items():
        if k in _STATE_LIST_NAMES:
            continue
        # Deep-copy is safe for these because they're small (bools, ints, lists of ints).
        setattr(new, k, copy.deepcopy(v))
    # Initialize the state lists to same length, all None. Caller fills.
    n_layers = len(cache.key_cache) if hasattr(cache, 'key_cache') else 0
    for name in _STATE_LIST_NAMES:
        if hasattr(cache, name):
            setattr(new, name, [None] * n_layers)
    return new


def _empty_dynamic_like_cache(cache):
    """Create an empty Qwen dynamic cache from either cache representation.

    Branch decode must grow only the valid prefix, not a StreamingCache's
    entire fixed-capacity tail.  Qwen's cache constructor only needs four
    topology attributes, so copying those directly avoids retaining the
    heavyweight model config on every cache object.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

    new = Qwen3_5DynamicCache.__new__(Qwen3_5DynamicCache)
    new.layer_types = copy.deepcopy(cache.layer_types)
    new.transformer_layers = copy.deepcopy(cache.transformer_layers)
    new.last_linear_layer = int(cache.last_linear_layer)
    n_layers = len(cache.key_cache)
    for name in _STATE_LIST_NAMES:
        setattr(new, name, [None] * n_layers)
    return new


def _cache_duplicate_batch(cache, factor: int = 2):
    """Return a new cache with batch dim expanded by ``factor`` for every
    per-layer state (K, V, conv_state, recurrent_state).

    For the production ``Qwen3_5DynamicCache`` and ``factor=1``, full-attention
    K/V prefixes are immutable during a branch: ``update`` rebinds each layer
    to a new ``torch.cat`` result. A/B can therefore share those prefix
    storages copy-on-append. Conv/recurrent tensors are never shared because
    causal-conv update mutates its state in place. This avoids two full K/V
    prefix copies per frame while preserving branch isolation.

    Other cache types and real batch expansion retain the conservative
    ``repeat_interleave`` path.
    """
    source_is_streaming = _is_streaming_cache(cache)
    # A fixed-capacity main cache is deliberately converted to a DynamicCache
    # at the fork.  This keeps each short A/B trail compact while the shared
    # main history continues to benefit from in-place prefill/appends.
    new = (_empty_dynamic_like_cache(cache) if source_is_streaming else _empty_like_cache(cache))
    share_qwen_prefix = (factor == 1 and (source_is_streaming or type(cache).__name__ == 'Qwen3_5DynamicCache'))
    for name, src_lst in _iter_state_lists(cache):
        dst_lst = getattr(new, name)
        for i, t in enumerate(src_lst):
            if t is not None:
                if source_is_streaming and name in ('key_cache', 'value_cache'):
                    # Never expose the invalid preallocated tail to a branch.
                    t = t[:, :, :int(cache.cur_len), :]
                if share_qwen_prefix and name in ('key_cache', 'value_cache'):
                    dst_lst[i] = t
                elif (source_is_streaming and name in ('key_cache', 'value_cache') and int(t.shape[0]) == 1):
                    # Read-only stride-0 prefix. Qwen's first branch update
                    # concatenates into fresh storage, so rows stay isolated
                    # without copying the long shared prefix at fork time.
                    dst_lst[i] = t.expand(factor, *t.shape[1:])
                else:
                    dst_lst[i] = t.repeat_interleave(factor, dim=0).contiguous()
    return new


def _cache_select_batch(cache, index: int):
    """Return a one-row cache view without copying the sequence prefix.

    The view is used when one row of a batched A/B decode finishes before the
    other. Cache updates either rebind K/V/recurrent tensors or mutate only
    the selected leading-dimension slice, so disjoint row views remain
    isolated while avoiding a transient full-cache copy during batch shrink.
    """
    new = _empty_like_cache(cache)
    batch_size = None
    for name, src_lst in _iter_state_lists(cache):
        dst_lst = getattr(new, name)
        for layer_idx, tensor in enumerate(src_lst):
            if tensor is None:
                continue
            current_batch = int(tensor.shape[0])
            if batch_size is None:
                batch_size = current_batch
            elif current_batch != batch_size:
                raise ValueError('cache state batch dimensions disagree: '
                                 f'expected {batch_size}, got {current_batch} at '
                                 f'{name}[{layer_idx}]')
            if index < 0 or index >= current_batch:
                raise IndexError(f'cache batch index {index} outside [0, {current_batch})')
            dst_lst[layer_idx] = tensor[index:index + 1]
    if batch_size is None:
        raise ValueError('cannot select a row from an empty cache')
    return new


def _cache_select_batch_rows(cache, indices):
    """Select arbitrary cache batch rows in order."""
    rows = torch.as_tensor(indices, dtype=torch.long)
    if rows.ndim != 1 or rows.numel() == 0:
        raise ValueError('indices must be a non-empty 1-D row list')
    min_row = int(rows.min())
    max_row = int(rows.max())
    new = _empty_like_cache(cache)
    batch_size = None
    device_rows = {}
    for name, src_lst in _iter_state_lists(cache):
        dst_lst = getattr(new, name)
        for layer_idx, tensor in enumerate(src_lst):
            if tensor is None:
                continue
            current_batch = int(tensor.shape[0])
            if batch_size is None:
                batch_size = current_batch
            elif current_batch != batch_size:
                raise ValueError('cache state batch dimensions disagree: '
                                 f'expected {batch_size}, got {current_batch}')
            if min_row < 0 or max_row >= current_batch:
                raise IndexError(f'cache rows outside [0, {current_batch}): '
                                 f'{rows.tolist()}')
            row_index = device_rows.get(tensor.device)
            if row_index is None:
                row_index = rows.to(device=tensor.device)
                device_rows[tensor.device] = row_index
            dst_lst[layer_idx] = tensor.index_select(0, row_index).contiguous()
    if batch_size is None:
        raise ValueError('cannot select rows from an empty cache')
    return new


def _pack_dynamic_caches(caches):
    """Pad independent compact bs=1 caches into one tensor batch.

    K/V prefixes are left-aligned and zero-padded to the maximum length.
    The returned bool mask identifies real slots; later decode tokens append
    after the common padded width and therefore create arbitrary per-row holes.
    """
    caches = list(caches)
    if not caches:
        raise ValueError('caches must not be empty')
    if any(_is_streaming_cache(cache) for cache in caches):
        raise TypeError('pack compact DynamicCaches, not StreamingCaches')
    packed = _empty_dynamic_like_cache(caches[0])
    batch_size = len(caches)
    lengths = []
    for cache in caches:
        length = None
        for tensor in cache.key_cache:
            if tensor is not None:
                length = int(tensor.shape[2])
                break
        lengths.append(0 if length is None else length)
    max_len = max(lengths)

    for name in _STATE_LIST_NAMES:
        destination = getattr(packed, name)
        for layer_idx in range(len(destination)):
            tensors = [getattr(cache, name)[layer_idx] for cache in caches]
            present = [tensor is not None for tensor in tensors]
            if not any(present):
                continue
            if not all(present):
                raise ValueError(f'{name}[{layer_idx}] is missing from only some caches')
            if name in ('key_cache', 'value_cache'):
                exemplar = tensors[0]
                rows = torch.zeros(
                    batch_size,
                    *exemplar.shape[1:2],
                    max_len,
                    *exemplar.shape[3:],
                    device=exemplar.device,
                    dtype=exemplar.dtype,
                )
                for row, (tensor, length) in enumerate(zip(tensors, lengths)):
                    if int(tensor.shape[0]) != 1:
                        raise ValueError('each cache passed to pack must have batch size 1')
                    if int(tensor.shape[2]) != length:
                        raise ValueError(f'cache layer length {tensor.shape[2]} != {length}')
                    rows[row:row + 1, :, :length, :].copy_(tensor)
                destination[layer_idx] = rows
            else:
                if any(int(tensor.shape[0]) != 1 for tensor in tensors):
                    raise ValueError('each cache passed to pack must have batch size 1')
                destination[layer_idx] = torch.cat(tensors, dim=0).contiguous()

    mask_device = next(tensor.device for tensor in packed.key_cache if tensor is not None)
    valid_mask = torch.arange(
        max_len, device=mask_device).unsqueeze(0) < torch.tensor(
            lengths, device=mask_device).unsqueeze(1)
    return packed, valid_mask


def _pack_query_prefix_caches(caches, *, repeats: int = 2):
    """Pack independent bs=1 main caches and repeat each query's prefix.

    Query-batched dual decode needs rows ``[q0_B, q0_A, q1_B, q1_A, ...]``.
    Building two temporary DynamicCaches per query first duplicates every
    conv/recurrent state and adds substantial Python work.  Pack directly
    from each main cache instead; StreamingCache contributes only ``cur_len``
    valid K/V slots, never its fixed-capacity tail.
    """
    caches = list(caches)
    if not caches:
        raise ValueError('caches must not be empty')
    if repeats < 1:
        raise ValueError('repeats must be positive')
    packed = _empty_dynamic_like_cache(caches[0])
    lengths = []
    for cache in caches:
        if _is_streaming_cache(cache):
            length = int(cache.cur_len)
        else:
            length = None
            for tensor in cache.key_cache:
                if tensor is not None:
                    length = int(tensor.shape[2])
                    break
            length = 0 if length is None else length
        lengths.append(length)
    repeated_lengths = [length for length in lengths for _ in range(repeats)]
    max_len = max(lengths)
    batch_size = len(caches) * repeats

    for name in _STATE_LIST_NAMES:
        destination = getattr(packed, name)
        for layer_idx in range(len(destination)):
            tensors = [getattr(cache, name)[layer_idx] for cache in caches]
            present = [tensor is not None for tensor in tensors]
            if not any(present):
                continue
            if not all(present):
                raise ValueError(f'{name}[{layer_idx}] is missing from only some caches')
            if any(int(tensor.shape[0]) != 1 for tensor in tensors):
                raise ValueError('each query cache must have batch size 1')
            exemplar = tensors[0]
            if name in ('key_cache', 'value_cache'):
                rows = torch.zeros(
                    batch_size,
                    *exemplar.shape[1:2],
                    max_len,
                    *exemplar.shape[3:],
                    device=exemplar.device,
                    dtype=exemplar.dtype,
                )
                for query_index, (tensor, length) in enumerate(zip(tensors, lengths)):
                    valid = tensor[:, :, :length, :]
                    row_start = query_index * repeats
                    rows[row_start:row_start + repeats, :, :length, :].copy_(valid.expand(repeats, *valid.shape[1:]))
                destination[layer_idx] = rows
            else:
                destination[layer_idx] = torch.cat([tensor.expand(repeats, *tensor.shape[1:]) for tensor in tensors],
                                                   dim=0).contiguous()

    mask_device = next(tensor.device for tensor in packed.key_cache if tensor is not None)
    valid_mask = torch.arange(
        max_len, device=mask_device).unsqueeze(0) < torch.tensor(
            repeated_lengths, device=mask_device).unsqueeze(1)
    return packed, valid_mask


def _extract_packed_cache_row(cache, row: int, valid_slots: torch.Tensor):
    """Compact one padded cache row back to physical KV order."""
    valid_slots = valid_slots.to(dtype=torch.bool)
    if valid_slots.ndim != 1:
        raise ValueError('valid_slots must be 1-D')
    new = _empty_dynamic_like_cache(cache)
    slot_indices = torch.nonzero(valid_slots, as_tuple=False).flatten()
    for name, src_lst in _iter_state_lists(cache):
        dst_lst = getattr(new, name)
        for layer_idx, tensor in enumerate(src_lst):
            if tensor is None:
                continue
            if not 0 <= row < int(tensor.shape[0]):
                raise IndexError(row)
            selected = tensor[row:row + 1]
            if name in ('key_cache', 'value_cache'):
                selected = selected.index_select(2, slot_indices.to(device=tensor.device))
            dst_lst[layer_idx] = selected.contiguous()
    return new


def _cache_append_trail(main_cache, trail_cache, trail_start: int):
    """Copy ``trail_cache``'s new K/V entries (positions >= trail_start) into
    ``main_cache`` along the sequence dim.

    Only touches the full-attention layers (key_cache / value_cache); the
    linear-attn state is handled separately by :func:`_merge_linear_state`.
    """
    if not hasattr(main_cache, 'key_cache'):
        return
    if _is_streaming_cache(main_cache):
        write_start = int(main_cache.cur_len)
        if write_start != int(trail_start):
            # The second append in a dual turn starts from the same branch
            # fork point but follows B physically in the main cache.
            if write_start < int(trail_start):
                raise RuntimeError('StreamingCache append begins before branch fork: '
                                   f'cur_len={write_start} trail_start={trail_start}')
        n_new = None
        for i, (k_trail, v_trail) in enumerate(zip(trail_cache.key_cache, trail_cache.value_cache)):
            if k_trail is None:
                continue
            k_new = k_trail[:, :, trail_start:, :]
            v_new = v_trail[:, :, trail_start:, :]
            layer_n = int(k_new.shape[2])
            if n_new is None:
                n_new = layer_n
            elif layer_n != n_new:
                raise RuntimeError('branch KV trail lengths disagree across layers: '
                                   f'{layer_n} != {n_new}')
            write_end = write_start + layer_n
            if write_end > int(main_cache.max_kv_len):
                raise RuntimeError('StreamingCache overflow while appending dual branch: '
                                   f'{write_end} > {main_cache.max_kv_len}')
            if main_cache.key_cache[i] is None:
                main_cache._lazy_allocate_layer(i, k_new, v_new)
            main_cache.key_cache[i][:, :, write_start:write_end, :].copy_(k_new)
            main_cache.value_cache[i][:, :, write_start:write_end, :].copy_(v_new)
        if n_new is None:
            raise RuntimeError('cannot append an empty dual branch KV trail')
        main_cache.advance(n_new)
        return
    for i, (k_trail, v_trail) in enumerate(zip(trail_cache.key_cache, trail_cache.value_cache)):
        if k_trail is None:
            continue
        k_new = k_trail[:, :, trail_start:, :]
        v_new = v_trail[:, :, trail_start:, :]
        if main_cache.key_cache[i] is None:
            main_cache.key_cache[i] = k_new.contiguous()
            main_cache.value_cache[i] = v_new.contiguous()
        else:
            main_cache.key_cache[i] = torch.cat([main_cache.key_cache[i], k_new], dim=2).contiguous()
            main_cache.value_cache[i] = torch.cat([main_cache.value_cache[i], v_new], dim=2).contiguous()


def _merge_linear_state_from_branches(main_cache, cache_A, cache_B, *, model=None, mode: str = 'additive'):
    """Merge linear-attention states from independent A/B branch caches.

    ``main_cache`` is still at the shared pre-branch state. ``cache_A`` and
    ``cache_B`` were decoded independently from copies of that same state.
    This mirrors training's additive fork merge:

        shared: state_pre + sigmoid(gamma) * (delta_A + delta_B)
        split:  state_pre + sigmoid(alpha) * delta_A + sigmoid(beta) * delta_B

    The checkpoint ``dual_merge`` gate modules live on the model's
    Qwen3_5GatedDeltaNet layers. If a layer has no recurrent state, it is
    skipped. Convolution states follow the physical post-merge sequence and
    therefore use A's tail, matching the earlier reference runner.
    """
    if mode != 'additive':
        raise NotImplementedError(f'independent-branch inference merge_mode={mode!r} is not '
                                  'implemented; use MSS_MERGE_MODE=additive for train/infer parity.')

    recur = getattr(main_cache, 'recurrent_states', None)
    if isinstance(recur, list) \
            and isinstance(getattr(cache_A, 'recurrent_states', None), list) \
            and isinstance(getattr(cache_B, 'recurrent_states', None), list):
        for i, state_pre in enumerate(main_cache.recurrent_states):
            state_A = cache_A.recurrent_states[i]
            state_B = cache_B.recurrent_states[i]
            if state_pre is None or state_A is None or state_B is None:
                continue
            merge_mod = _get_dual_merge_module(model, i)
            if model is not None and merge_mod is None:
                raise RuntimeError(f'linear-attention layer {i} has recurrent state but no '
                                   'loaded dual_merge module; refusing the silent fixed-0.5 '
                                   'fallback')
            if merge_mod is not None and type(merge_mod).__name__ == 'GatedAdditiveMerge':
                main_cache.recurrent_states[i] = merge_mod(state_pre, state_B, state_A).contiguous()
            elif merge_mod is not None:
                raise RuntimeError(f'linear-attention layer {i} has unsupported merge '
                                   f'module {type(merge_mod).__name__}')
            else:
                # Retain the model-less fallback for tiny cache unit tests.
                main_cache.recurrent_states[i] = (state_pre + 0.5 * ((state_A - state_pre) +
                                                                     (state_B - state_pre))).contiguous()

    if isinstance(getattr(main_cache, 'conv_states', None), list) \
            and isinstance(getattr(cache_A, 'conv_states', None), list):
        for i, cs in enumerate(cache_A.conv_states):
            if cs is not None:
                main_cache.conv_states[i] = cs.contiguous()


def _get_dual_merge_module(model, layer_idx: int):
    if model is None:
        return None
    # Qwen3_5ForConditionalGeneration owns the text stack at
    # ``model.model.language_model``; tiny/reference wrappers may expose
    # ``language_model`` directly.  Missing this nesting silently selected
    # the fixed-0.5 fallback even for learned-gamma checkpoints.
    for candidate in (model, getattr(model, 'model', None)):
        lm = getattr(candidate, 'language_model', None)
        layers = getattr(lm, 'layers', None)
        if layers is None or layer_idx >= len(layers):
            continue
        linear_attn = getattr(layers[layer_idx], 'linear_attn', None)
        return getattr(linear_attn, 'dual_merge', None)
    return None


# ---------------------------------------------------------------------------
# Route mask helpers
# ---------------------------------------------------------------------------


def _set_route_none(seq_len: int, device):
    """Ctx = no LoRA (base only): used for prefill of image + prefix."""
    clear_ctx()


def _set_route_A(seq_len: int, device):
    clear_ctx()
    _dual_lora_ctx['route_A'] = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    _dual_lora_ctx['route_B'] = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    _dual_lora_ctx['has_A'] = True
    _dual_lora_ctx['has_B'] = False


def _set_route_B(seq_len: int, device):
    clear_ctx()
    _dual_lora_ctx['route_A'] = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    _dual_lora_ctx['route_B'] = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    _dual_lora_ctx['has_A'] = False
    _dual_lora_ctx['has_B'] = True


# ---------------------------------------------------------------------------
# Dual-branch decoder
# ---------------------------------------------------------------------------


class DualBranchRunner:
    """Owns a model + tokenizer + growing chat history and drives per-frame
    dual-branch generation."""

    def __init__(self,
                 model,
                 tokenizer,
                 *,
                 processor=None,
                 device='cuda:0',
                 max_new_per_branch=64,
                 system_prompt: Optional[str] = None,
                 position_overlap: bool = False):
        self.model = model
        self.tok = tokenizer
        self.processor = processor  # None => text-only; required for images
        self.device = device
        self.max_new_per_branch = max_new_per_branch
        self.im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        self.assistant_role_id = tokenizer.encode('assistant', add_special_tokens=False)[0]
        self.user_role_id = tokenizer.encode('user', add_special_tokens=False)[0]
        self.nl_id = tokenizer.encode('\n', add_special_tokens=False)[0]
        self.source_prefix_ids = tokenizer.encode(
            f'<|im_start|>assistant\n{NON_THINKING_PREFIX}',
            add_special_tokens=False,
        )
        if len(self.source_prefix_ids) < 2:
            raise ValueError('assistant non-thinking prefix is unexpectedly short')
        self.branch_source_id = int(self.source_prefix_ids[-1])
        self.shared_prefix_without_source = self.source_prefix_ids[:-1]
        self.branch_separator_ids = tokenizer.encode(B_BRANCH_SEPARATOR, add_special_tokens=False)
        self.turn_trailer_ids = tokenizer.encode(A_TURN_TRAILER, add_special_tokens=False)
        if self.branch_separator_ids != [self.nl_id] or self.turn_trailer_ids != [self.nl_id]:
            raise ValueError(f'expected branch/trailer separator to tokenize as newline id '
                             f'{self.nl_id}; got branch={self.branch_separator_ids} '
                             f'trailer={self.turn_trailer_ids}')
        # spatial_merge_size: image_grid_thw product / merge_size^2 = # image_pad tokens.
        # For Qwen3.5 this is 2 (patch 16 with 2x2 merge). Fall back to 2.
        self.spatial_merge_size = int(getattr(getattr(model.config, 'vision_config', None), 'spatial_merge_size', 2))

        # Growing token history + KV cache
        self.history: List[int] = []
        self.past_key_values = None
        self.turn = 0

        self.position_overlap = position_overlap
        # ``logical_pos`` tracks the RoPE position id of the *next* text token.
        # For image-containing forwards we defer to HF's MRoPE by NOT passing
        # position_ids explicitly, then sync logical_pos to the resulting
        # cache length. During pure-text prefill and parallel decode we pass
        # position_ids ourselves; during decode-merge we advance by
        # ``max(lb, la)`` when overlap is on (see step_frame docs).
        self.logical_pos = 0

        if system_prompt is not None:
            self._prefill_system(system_prompt)

    # ---------- primitives ----------
    def _prefill(self, token_ids: List[int], *, route: str = 'none'):
        """Forward the given tokens through the model, extending KV cache.

        ``route`` selects which LoRA route applies to every token in this
        prefill chunk. The shared prefix and post-merge newline use
        ``route='none'``; branch-source/action chunks use ``A`` or ``B``.
        When ``position_overlap=True`` we pass explicit position_ids anchored
        to ``self.logical_pos``; otherwise HF's default sequential-from-cache
        behavior is used.
        """
        input_ids = torch.tensor([token_ids], device=self.device)
        N = input_ids.size(1)
        if route == 'none':
            _set_route_none(N, self.device)
        elif route == 'A':
            _set_route_A(N, self.device)
        elif route == 'B':
            _set_route_B(N, self.device)
        else:
            raise ValueError(f'unknown route={route!r}')
        kwargs = dict(
            input_ids=input_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        if self.position_overlap:
            pos = torch.arange(
                self.logical_pos,
                self.logical_pos + N,
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0)  # [1, N]
            kwargs['position_ids'] = pos
        with torch.no_grad():
            out = self.model(**kwargs)
        self.past_key_values = out.past_key_values
        self.history.extend(token_ids)
        self.logical_pos += N
        return out.logits[:, -1, :]

    def _prefill_system(self, system_prompt: str):
        text = f'<|im_start|>system\n{system_prompt}<|im_end|>\n'
        ids = self.tok.encode(text, add_special_tokens=False)
        self._prefill(ids)

    def add_user_turn(self, content: str):
        """Append a text-only user message and prefill it."""
        text = f'<|im_start|>user\n{content}<|im_end|>\n'
        ids = self.tok.encode(text, add_special_tokens=False)
        self._prefill(ids)

    def add_user_image_frame(self, image: Image.Image, *, query_text: Optional[str] = None):
        """Append a user turn containing exactly one image (and optional
        query text prefixing it). Runs through the processor to compute
        pixel_values + image_grid_thw, builds the token layout
            <|im_start|>user\\n[query_text]<|vision_start|><|image_pad|>*N<|vision_end|><|im_end|>\\n
        and prefills. During image prefill we let HF compute MRoPE
        position_ids automatically; ``logical_pos`` is re-synced to the
        resulting cache length afterwards.
        """
        assert self.processor is not None, ('processor required for image frames; pass processor= to DualBranchRunner')
        # Extract pixel_values + image_grid_thw for this single image.
        # Qwen3.5's Qwen2VLImageProcessorFast handles the smart-resize
        # internally (do_resize=True default); we pass the PIL image directly.
        vp = self.processor.image_processor(
            images=[image.convert('RGB')],
            return_tensors='pt',
        )
        pixel_values = vp['pixel_values'].to(self.device, dtype=self.model.dtype)
        image_grid_thw = vp['image_grid_thw'].to(self.device)

        # Number of <|image_pad|> tokens for this image
        thw = image_grid_thw.view(-1).tolist()
        n_pad = (int(thw[0]) * int(thw[1]) * int(thw[2])) // (self.spatial_merge_size**2)

        # Build the token id sequence for this user turn.
        pre_ids: List[int] = [self.im_start_id, self.user_role_id, self.nl_id]
        if query_text:
            pre_ids += self.tok.encode(query_text, add_special_tokens=False)
        pre_ids += [self.vision_start_id]
        pre_ids += [self.image_pad_id] * n_pad
        pre_ids += [self.vision_end_id, self.im_end_id, self.nl_id]

        input_ids = torch.tensor([pre_ids], device=self.device)
        N = input_ids.size(1)
        _set_route_none(N, self.device)
        kwargs = dict(
            input_ids=input_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        # NOTE: no explicit position_ids -- HF computes MRoPE 3D positions
        # from input_ids + image_grid_thw. We re-sync logical_pos below.
        with torch.no_grad():
            out = self.model(**kwargs)
        self.past_key_values = out.past_key_values
        self.history.extend(pre_ids)
        # Sync logical_pos to physical cache length after image prefill (HF
        # advanced MRoPE by N text-equivalent positions).
        self.logical_pos = self._main_seqlen()
        return out.logits[:, -1, :]

    def _prefill_shared_prefix(self):
        """Prefill assistant prefix up to, but not including, terminal P.

        Training rewrites the terminal prefix token into two branch source
        tokens:

            shared_prefix_without_terminal_P, P_B, B..., P_A, A...

        So inference must stop before P, fork, then feed P through the
        branch-specific LoRA route. Prefilling the full prefix as base would
        add an extra un-routed P and shift branch positions by one.
        """
        return self._prefill(self.shared_prefix_without_source, route='none')

    # ---------- branch decode ----------
    def _branch_forward_token(self, cache, token_id: int, *, route: str, position_id: int):
        input_ids = torch.tensor([[int(token_id)]], device=self.device, dtype=torch.long)
        if route == 'A':
            _set_route_A(1, self.device)
        elif route == 'B':
            _set_route_B(1, self.device)
        else:
            raise ValueError(f'branch route must be A/B, got {route!r}')

        kwargs = dict(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        if self.position_overlap:
            kwargs['position_ids'] = torch.tensor([[int(position_id)]], device=self.device, dtype=torch.long)
        with torch.no_grad():
            out = self.model(**kwargs)
        return out.past_key_values, out.logits[:, -1, :]

    def _decode_one_branch(self, *, route: str, include_b_separator: bool):
        """Decode one branch from a copy of the pre-P main cache.

        The duplicated terminal prefix token P is the first forwarded token
        and is routed through the branch LoRA. Its logits predict the first
        action token. Generated ``<|im_end|>`` is forwarded into the branch KV
        before stopping because training includes it inside the branch span.
        For B, the deterministic separator newline after the mid ``im_end`` is
        also forwarded through LoRA_B and included in the B span.
        """
        cache = _cache_duplicate_batch(self.past_key_values, factor=1)
        generated: List[int] = []
        extra: List[int] = []
        forced_stop = False

        pending = self.branch_source_id
        offset = 0
        while True:
            cache, logits = self._branch_forward_token(
                cache, pending, route=route, position_id=self.logical_pos + offset)
            offset += 1

            if pending == self.im_end_id:
                break
            if len(generated) >= self.max_new_per_branch:
                # Keep the physical chat/template layout valid for the next
                # turn. This is a safety fallback for under-trained/debug
                # checkpoints or very small max_new; normal trained models
                # should generate <|im_end|> themselves before this branch.
                generated.append(self.im_end_id)
                pending = self.im_end_id
                forced_stop = True
                continue

            next_token = int(logits.argmax(dim=-1).item())
            generated.append(next_token)
            pending = next_token

        if include_b_separator and generated and generated[-1] == self.im_end_id:
            sep_id = self.branch_separator_ids[0]
            cache, _ = self._branch_forward_token(cache, sep_id, route=route, position_id=self.logical_pos + offset)
            offset += 1
            extra.append(sep_id)

        # Span length includes P plus generated branch tokens and any fixed
        # branch-local extra token. It is the length appended from this branch
        # cache to the main full-attention KV.
        span_len = 1 + len(generated) + len(extra)
        return generated, extra, cache, span_len, forced_stop

    def _parallel_decode_reference(self):
        """Correctness-first branch decode.

        This runs A and B as two independent batch=1 forwards from identical
        pre-P cache snapshots. It is semantically the same as parallel
        decoding for attention and recurrent-state purposes: both branches
        start from the same state and never see each other's tokens.
        """
        b_tokens, b_extra, b_cache, b_span_len, b_forced = self._decode_one_branch(route='B', include_b_separator=True)
        a_tokens, a_extra, a_cache, a_span_len, a_forced = self._decode_one_branch(route='A', include_b_separator=False)
        assert not a_extra
        return {
            'B_tokens': b_tokens,
            'B_extra': b_extra,
            'B_cache': b_cache,
            'B_span_len': b_span_len,
            'B_forced_stop': b_forced,
            'A_tokens': a_tokens,
            'A_cache': a_cache,
            'A_span_len': a_span_len,
            'A_forced_stop': a_forced,
        }

    def _decode_merge_after_prefix(self) -> dict:
        """Fork from the current pre-P state, decode both branches, merge, and
        append the shared post-turn newline.

        Precondition: the new user/frame tokens and the assistant shared
        prefix without terminal P are already in ``self.past_key_values``.
        """
        prefix_end_len = self._main_seqlen()
        branch_start_pos = self.logical_pos

        branch = self._parallel_decode_reference()
        b_tokens = branch['B_tokens']
        a_tokens = branch['A_tokens']
        b_extra = branch['B_extra']

        _cache_append_trail(self.past_key_values, branch['B_cache'], prefix_end_len)
        _cache_append_trail(self.past_key_values, branch['A_cache'], prefix_end_len)
        _merge_linear_state_from_branches(
            self.past_key_values,
            cache_A=branch['A_cache'],
            cache_B=branch['B_cache'],
            model=self.model,
            mode='additive')

        # Update logical position counter:
        #   position_overlap=True:  advance by max(B_span, A_span), matching
        #                           position_overlap.py's post-A shift.
        #   position_overlap=False: advance by B_span + A_span.
        lb, la = branch['B_span_len'], branch['A_span_len']
        if self.position_overlap:
            self.logical_pos = branch_start_pos + max(lb, la)
        else:
            self.logical_pos = branch_start_pos + lb + la

        # Physical history matches training after source-fork rewrite:
        #   P_B, B..., B_sep, P_A, A...
        self.history.extend([self.branch_source_id])
        self.history.extend(b_tokens)
        self.history.extend(b_extra)
        self.history.extend([self.branch_source_id])
        self.history.extend(a_tokens)

        # The final newline after A's closing <|im_end|> is shared
        # post-merge context, not part of either branch route.
        if self.turn_trailer_ids:
            self._prefill(self.turn_trailer_ids, route='none')

        return {
            'turn': self.turn,
            'B_tokens': b_tokens,
            'A_tokens': a_tokens,
            'B_extra': b_extra,
            'B_forced_stop': branch['B_forced_stop'],
            'A_forced_stop': branch['A_forced_stop'],
            'B_text': self.tok.decode(b_tokens),
            'A_text': self.tok.decode(a_tokens),
        }

    # ---------- per-turn orchestrator ----------
    def step_frame(self, frame_content: Optional[str] = None) -> dict:
        """One full frame: (optional new user turn) → shared pre-P prefix →
        branch decode → merge → append.

        Returns dict with:
            - B_tokens: list of B branch's decoded token ids
            - A_tokens: list of A branch's decoded token ids
            - B_text: decoded text
            - A_text: decoded text
        """
        self.turn += 1
        if frame_content is not None:
            self.add_user_turn(frame_content)
        self._prefill_shared_prefix()
        return self._decode_merge_after_prefix()

    def _main_seqlen(self) -> int:
        if self.past_key_values is None:
            return 0
        if hasattr(self.past_key_values, 'key_cache') and self.past_key_values.key_cache:
            for k in self.past_key_values.key_cache:
                if k is not None:
                    return int(k.shape[2])
        return len(self.history)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_model(model_path: str, device: str = 'cuda:0'):
    from transformers import Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.add_special_tokens({'additional_special_tokens': ['<inner>', '</inner>', '<silent>', '<speak>', '</speak>']})
    asst_id = tok.encode('assistant', add_special_tokens=False)[0]

    # AutoProcessor for the vision path (image_processor + tokenizer bundled)
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f'[warn] AutoProcessor unavailable ({e!r}); text-only path only', flush=True)
        processor = None

    has_dual = _checkpoint_has_dual_weights(model_path)
    if has_dual:
        # These must be set before the model constructor runs: the subclass
        # below injects the dual modules in __init__, before HF loads weights.
        os.environ.setdefault('MSS_DUAL_BRANCH', '1')
        os.environ.setdefault('MSS_MERGE_MODE', 'additive')
        os.environ.setdefault('MSS_POSITION_ID_OVERLAP', '1')
    use_dual = has_dual or dual_lora_enabled()

    ModelCls = Qwen3_5ForConditionalGeneration
    if use_dual:

        class DualQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):

            def __init__(self, config, *args, **kwargs):
                super().__init__(config, *args, **kwargs)
                enable_dual_lora(self, assistant_role_id=asst_id)

        ModelCls = DualQwen3_5ForConditionalGeneration

    print(f'[loading] model={model_path} dual={use_dual}', flush=True)
    loaded = ModelCls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        device_map=device,
        output_loading_info=True,
    )
    if isinstance(loaded, tuple):
        model, loading_info = loaded
    else:
        model, loading_info = loaded, {}
    _assert_dual_weights_loaded(loading_info, expected=use_dual)

    if model.get_input_embeddings().num_embeddings != len(tok):
        model.resize_token_embeddings(len(tok))
    model.eval()
    print(f'[loaded ]  dtype={model.dtype}, device={next(model.parameters()).device}', flush=True)
    return model, tok, processor


def _assert_dual_weights_loaded(loading_info: Dict, *, expected: bool) -> None:
    if not expected or not loading_info:
        return
    unexpected = [k for k in loading_info.get('unexpected_keys', []) if _is_dual_weight_key(k)]
    missing = [k for k in loading_info.get('missing_keys', []) if _is_dual_weight_key(k)]
    if unexpected or missing:
        raise RuntimeError('dual checkpoint did not load cleanly: '
                           f'unexpected_dual_keys={unexpected[:8]} '
                           f'missing_dual_keys={missing[:8]}')


def _resolve_frames(spec: str) -> list:
    """Given ``spec`` = a directory of jpgs OR a comma-separated list of paths,
    return a sorted list of image file paths."""
    from pathlib import Path
    if ',' in spec:
        return [p.strip() for p in spec.split(',') if p.strip()]
    path = Path(spec)
    if path.is_dir():
        pats = ['*.jpg', '*.jpeg', '*.png']
        files = sorted(sum([list(path.glob(p)) for p in pats], []))
        return [str(f) for f in files]
    # Single file
    return [str(path)]


def _demo_run(args):
    model, tok, processor = load_model(args.model, device=args.device)
    system = None
    if args.system:
        with open(args.system) as f:
            system = f.read()
    pos_overlap = os.environ.get('MSS_POSITION_ID_OVERLAP', '1').strip() \
        not in ('', '0', 'false', 'False')
    runner = DualBranchRunner(
        model,
        tok,
        processor=processor,
        device=args.device,
        max_new_per_branch=args.max_new,
        system_prompt=system,
        position_overlap=pos_overlap)
    print(f'[runner ]  position_overlap={pos_overlap}  vision={processor is not None}', flush=True)

    if args.video:
        frame_paths = _resolve_frames(args.video)
        if args.n_frames > 0:
            frame_paths = frame_paths[:args.n_frames]
        print(f'[frames ]  {len(frame_paths)} images from {args.video!r}', flush=True)

        import time
        for i, fp in enumerate(frame_paths):
            img = Image.open(fp).convert('RGB')
            query = f'Query: {args.query}' if i == 0 else None
            print(f'\n=== turn {i+1}: {fp} ===', flush=True)
            runner.turn += 1
            # ---------- timing ----------
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            runner.add_user_image_frame(img, query_text=query)
            torch.cuda.synchronize()
            t_img = time.perf_counter() - t0
            t0 = time.perf_counter()
            runner._prefill_shared_prefix()
            torch.cuda.synchronize()
            t_prefix = time.perf_counter() - t0
            t0 = time.perf_counter()
            r = runner._decode_merge_after_prefix()
            torch.cuda.synchronize()
            t_decode = time.perf_counter() - t0
            b_tokens = r['B_tokens']
            a_tokens = r['A_tokens']
            lb, la = len(b_tokens), len(a_tokens)
            print(f'  B ({lb}t): {tok.decode(b_tokens)!r}', flush=True)
            print(f'  A ({la}t): {tok.decode(a_tokens)!r}', flush=True)
            print(
                f'  [time] img_prefill={t_img*1000:.0f}ms '
                f'prefix={t_prefix*1000:.0f}ms '
                f'decode_merge={t_decode*1000:.0f}ms '
                f'total={(t_img+t_prefix+t_decode)*1000:.0f}ms',
                flush=True)
    else:
        # Text-only fallback (v1 behavior)
        query = f'Query: {args.query} [Frame 0]'
        print(f'\n=== turn 1 (text): {query!r} ===', flush=True)
        r = runner.step_frame(frame_content=query)
        print(f'  B: {r["B_text"]!r}', flush=True)
        print(f'  A: {r["A_text"]!r}', flush=True)
        for i in range(1, args.n_frames):
            frame = f'[Frame {i}]'
            print(f'\n=== turn {i+1} (text): {frame!r} ===', flush=True)
            r = runner.step_frame(frame_content=frame)
            print(f'  B: {r["B_text"]!r}', flush=True)
            print(f'  A: {r["A_text"]!r}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Qwen3.5-4B base or LoRA-adapted checkpoint directory')
    p.add_argument('--query', default='Describe what is happening.', help='initial query text')
    p.add_argument(
        '--video',
        type=str,
        default=None,
        help='directory of frame jpgs, comma-separated jpg list, '
        'or single jpg path. If omitted, runs text-only mode.')
    p.add_argument('--n_frames', type=int, default=3, help='cap number of frames (0 = all in --video dir)')
    p.add_argument('--max_new', type=int, default=48, help='max new tokens per branch per frame')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--system', type=str, default=None, help='path to system prompt file (defaults to dual prompt)')
    args = p.parse_args()

    # Point at dual system prompt by default if none provided
    if args.system is None:
        default = REPO_ROOT / 'data' / 'system_prompt_dual.txt'
        if default.exists():
            args.system = str(default)

    _demo_run(args)


if __name__ == '__main__':
    main()
