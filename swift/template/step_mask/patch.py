"""Wire the step+window mask into Qwen3.5's transformers implementation.

Activation is gated by the env var ``STEP_MASK_WINDOW`` (positive int = window
in frames; absent / 0 = feature off, code path bit-for-bit identical to
upstream). When active:

* The outer ``Qwen3_5ForConditionalGeneration.forward`` stashes ``input_ids``,
  ``labels``, and the time-axis ``position_ids`` in a context variable.
* ``transformers.models.qwen3_5.modeling_qwen3_5.create_causal_mask`` is
  replaced by a wrapper that, when the context variable is set, returns a
  ``BlockMask`` built by :func:`build_step_window_block_mask` instead of the
  standard 4D attention mask. The 8 full-attention layers route this through
  ``flex_attention``; the 24 linear-attention layers are untouched (they
  receive the unrelated ``linear_attn_mask``).
* The model's ``text_config._attn_implementation`` is forced to
  ``"flex_attention"`` so the full-attention interface knows how to consume a
  ``BlockMask``.

Everything is reversible: original references are kept and re-applied if the
patch is asked to undo itself.
"""

from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .execution_replay import physical_logits_positions
from .flex import build_step_window_block_mask

log = logging.getLogger(__name__)

_ENV_VAR = 'STEP_MASK_WINDOW'
_DEBUG_ENV = 'STEP_MASK_DEBUG'
_ACTOR_FUSED_FROZEN_LINEAR_BACKWARD_ENV = (
    'MSS_DUAL_ACTOR_FUSED_FROZEN_LINEAR_BACKWARD'
)
_ACTOR_FUSED_TRAINABLE_LINEAR_INPUT_BACKWARD_ENV = (
    'MSS_DUAL_ACTOR_FUSED_TRAINABLE_LINEAR_INPUT_BACKWARD'
)
_ACTOR_SEGMENTED_RMSNORM_BACKWARD_ENV = (
    'MSS_DUAL_ACTOR_SEGMENTED_RMSNORM_BACKWARD'
)
_ACTOR_FUSED_GATED_RMSNORM_BACKWARD_ENV = (
    'MSS_DUAL_ACTOR_FUSED_GATED_RMSNORM_BACKWARD'
)
_ACTOR_FUSED_LM_HEAD_INPUT_BACKWARD_ENV = (
    'MSS_DUAL_ACTOR_FUSED_LM_HEAD_INPUT_BACKWARD'
)
# Bound each custom-autograd block so recomputed vocabulary logits stay
# memory-bounded. Its input derivative still follows the original physical
# one/two-row groups; combining those groups changes CUDA GEMM numerics.
_ACTOR_FUSED_LM_HEAD_MAX_ROWS = 8
# Print the first ``_DEBUG_MAX_CALLS`` forwards' mask diagnostics, then go
# silent. Set ``_DEBUG_MAX_CALLS=-1`` to keep logging forever.
_DEBUG_MAX_CALLS = 5
_debug_call_count = 0


def _debug_enabled() -> bool:
    return os.environ.get(_DEBUG_ENV, '').strip() not in ('', '0', 'false', 'False')


# Holds the per-forward inputs we need to recompute the BlockMask.
# None when feature is inactive or no labels were provided (e.g. inference).
_ctx: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar('step_mask_ctx', default=None)

# Set by ``patch_qwen3_5_for_step_mask`` so we can restore on demand.
_actor_attention_replay_ctx: dict = {}
_actor_precomputed_visual_ctx: contextvars.ContextVar[Optional[dict]] = (
    contextvars.ContextVar('actor_precomputed_visual_ctx', default=None))
_actor_deferred_lm_head_ctx: contextvars.ContextVar[Optional[dict]] = (
    contextvars.ContextVar('actor_deferred_lm_head_ctx', default=None))
_patched_state: dict = {}


@contextmanager
def use_actor_precomputed_visual_features(features):
    """Temporarily supply frozen per-image features to one actor forward."""
    normalized = tuple(features)
    if not normalized or any(not isinstance(feature, torch.Tensor) for feature in normalized):
        raise ValueError('precomputed actor visual features must be non-empty tensors')
    state = {'features': normalized, 'consumed': False}
    token = _actor_precomputed_visual_ctx.set(state)
    try:
        yield
    finally:
        _actor_precomputed_visual_ctx.reset(token)
    if not state['consumed']:
        raise RuntimeError('actor forward did not consume precomputed visual features')


@contextmanager
def defer_actor_lm_head_projection():
    """Return selected actor hidden states from the outer model forward.

    Actor replay only needs sampled-token log-probabilities.  Keeping the
    selected ``hidden_size`` rows lets the caller checkpoint the much wider
    LM-head/policy tail instead of retaining ``vocab_size`` logits until
    backward.  The patched LM head marks the scope as consumed so an absent
    or bypassed patch fails closed.
    """
    state = {'consumed': False}
    token = _actor_deferred_lm_head_ctx.set(state)
    try:
        yield
    finally:
        _actor_deferred_lm_head_ctx.reset(token)
    if not state['consumed']:
        raise RuntimeError('actor forward did not defer the LM-head projection')


def clear_step_mask_context() -> None:
    """Drop a completed training forward's mask context.

    Training installs a fresh context in ``pre_forward_hook``.  Explicitly
    clearing it after backward prevents a stale full-sequence context from
    becoming the reset target of the next streaming rollout.
    """
    _actor_attention_replay_ctx.clear()
    _ctx.set(None)


def env_window_frames() -> int:
    raw = os.environ.get(_ENV_VAR, '').strip()
    if not raw:
        return 0
    try:
        v = int(raw)
    except ValueError:
        log.warning(f'{_ENV_VAR}={raw!r} is not an integer; treating as disabled')
        return 0
    return max(v, 0)


def _maybe_print_mask_diagnostics(ctx, block_mask) -> None:
    """One-shot per-forward summary: only fires when STEP_MASK_DEBUG is set and
    while the global call counter is below the cap. Prints from rank 0 only.
    """
    global _debug_call_count
    if not _debug_enabled():
        return
    if _DEBUG_MAX_CALLS >= 0 and _debug_call_count >= _DEBUG_MAX_CALLS:
        return
    rank = int(os.environ.get('RANK', '0') or 0)
    if rank != 0:
        return

    from .meta import build_step_mask_meta

    # Streaming-mode ctx has no input_ids tensor; pull from state.input_ids_np.
    if ctx.get('streaming'):
        ids_np = ctx['state'].input_ids_np
        L = int(ids_np.shape[0])
    else:
        input_ids = ctx['input_ids']
        L = int(input_ids.shape[-1])
        ids_np = input_ids[0].detach().cpu().numpy()
    meta = build_step_mask_meta(ids_np, assistant_role_id=ctx['assistant_role_id'])
    n_visual = int(meta.is_visual.sum())
    n_frames = int(meta.frame_id.max()) + 1 if (meta.frame_id >= 0).any() else 0
    n_close = int(meta.step_close.sum())
    cu_seqlens = ctx.get('cu_seqlens')
    n_samples = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else 1

    # Block-level density estimate of the BlockMask.
    try:
        dense = block_mask.to_dense().bool().squeeze().float().mean().item()
    except Exception:
        dense = float('nan')

    print(
        f'[step_mask][call#{_debug_call_count}] W={ctx["window_frames"]} L={L} '
        f'n_samples={n_samples} n_frames={n_frames} n_visual_tokens={n_visual} '
        f'n_step_closes={n_close} block_density~{dense:.3f}',
        flush=True)
    _debug_call_count += 1


def enable_flex_attn_support() -> None:
    """Mark Qwen3_5 as flex_attention-capable.

    transformers gates the dispatcher on ``_supports_flex_attn`` being True on
    the model class. Upstream sets only flash + sdpa for Qwen3.5; we know flex
    works because the only thing it does differently is consume a BlockMask
    instead of a tensor mask. Must be called *before* ``from_pretrained``.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5
    modeling_qwen3_5.Qwen3_5PreTrainedModel._supports_flex_attn = True


_dynamic_flex_patched: dict = {'installed': False}
_kernel_opts_patched: dict = {'installed': False}


def _env_flex_kernel_options() -> Optional[dict]:
    """Optional per-forward kernel_options for flex_attention, driven by env.

    ``MSS_FLEX_KERNEL_OPTS`` is a comma-separated ``k=v`` list:
        MSS_FLEX_KERNEL_OPTS='BLOCK_M=64,BLOCK_N=64,num_stages=1,num_warps=4'

    Purpose: on GPUs with tight SRAM (e.g. L20Z: 228 KB per SM), the default
    Triton kernel config for flex_attention requires more shared memory than
    available (~393 KB). Autotune then fails with
    ``No valid triton configs. OutOfMemoryError: out of resource: ...``.
    Passing smaller BLOCK sizes / fewer pipeline stages shrinks the kernel's
    SRAM footprint enough to fit. The trade-off is slightly slower kernels;
    for training (fixed shapes) the slowdown is typically <15%.

    Also always includes ``FORCE_USE_FLEX_ATTENTION=True`` (needed for the
    dynamic=True compile to consistently pick the prefill kernel path).
    """
    raw = os.environ.get('MSS_FLEX_KERNEL_OPTS', '').strip()
    if not raw:
        return None
    opts: dict = {}
    for pair in raw.split(','):
        pair = pair.strip()
        if not pair or '=' not in pair:
            continue
        k, v = pair.split('=', 1)
        k, v = k.strip(), v.strip()
        # Values are either int (BLOCK_M / num_stages / num_warps) or bool
        if v.lower() in ('true', 'false'):
            opts[k] = v.lower() == 'true'
        else:
            try:
                opts[k] = int(v)
            except ValueError:
                opts[k] = v
    return opts or None


def _actor_attention_replay_enabled(module, query, key) -> bool:
    if os.environ.get('MSS_DUAL_ACTOR_LINEAR_MODE', '').strip().lower() != 'inference_aligned':
        return False
    # The same Transformers FlexAttention integration is also used by the
    # vision tower. Only the eight language full-attention layers consume
    # the physical dual-stream history reconstructed below.
    if type(module).__name__ != 'Qwen3_5Attention':
        return False
    input_ids = _actor_attention_replay_ctx.get('input_ids')
    if input_ids is None:
        return False
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise RuntimeError('inference-aligned actor attention requires physical [1, L] '
                           f'input_ids, got {tuple(input_ids.shape)}')
    seq_len = int(input_ids.shape[1])
    matches_physical_layout = (
        int(query.shape[0]) == 1 and int(key.shape[0]) == 1 and int(query.shape[2]) == seq_len
        and int(key.shape[2]) == seq_len)
    if not matches_physical_layout:
        raise RuntimeError('inference-aligned actor full attention does not match the '
                           'authoritative physical layout: '
                           f'query={tuple(query.shape)} key={tuple(key.shape)} L={seq_len}')
    return True


def _evict_actor_visual_positions(state, kept, *, processed_end: int):
    if processed_end <= 0:
        raise RuntimeError('actor attention eviction has no processed token')
    state._refresh_meta()
    last = processed_end - 1
    current_segment = int(state._seg_id[last])
    current_frame = int(state._cur_frame[last])
    window = int(state.window_frames)
    result = []
    for position in kept:
        is_visual = bool(state._is_visual[position])
        segment = int(state._seg_id[position])
        frame = int(state._frame_id[position])
        evict = is_visual and (segment < current_segment or
                               (segment == current_segment and frame <= current_frame - window))
        if not evict:
            result.append(position)
    return result


def _build_actor_attention_replay_schedule(device: torch.device):
    cached = _actor_attention_replay_ctx.get('schedule')
    if cached is not None:
        if cached['device'] != device:
            raise RuntimeError('actor attention schedule device changed during replay')
        return cached['records']

    from .inference import StreamingStepMaskState, build_batched_streaming_block_mask, build_streaming_block_mask
    from .linear_fork import get_linear_fork_ctx

    linear_ctx = get_linear_fork_ctx()
    if linear_ctx is None or not linear_ctx.get('b_spans'):
        raise RuntimeError('inference-aligned actor attention lacks dual fork metadata')
    input_ids = _actor_attention_replay_ctx['input_ids']
    total_length = int(input_ids.shape[1])
    sample_spans = list(linear_ctx['sample_spans'])
    b_spans = list(linear_ctx['b_spans'])
    a_spans = list(linear_ctx['a_spans'])
    frame_starts = list(linear_ctx['frame_starts'])
    records = []
    covered_queries = []
    branch_index = 0

    for sample_start, sample_end in sample_spans:
        sample_start, sample_end = int(sample_start), int(sample_end)
        sample_length = sample_end - sample_start
        local_ids = (input_ids[0, sample_start:sample_end].detach().cpu().numpy())
        state = StreamingStepMaskState(
            window_frames=int(_actor_attention_replay_ctx['window_frames']),
            assistant_role_id=int(_actor_attention_replay_ctx['assistant_role_id']),
        )
        state.extend(local_ids)
        local_frame_starts = [
            int(value) - sample_start for value in frame_starts if sample_start <= int(value) < sample_end
        ]

        def add_record(q_positions, kv_positions):
            q_positions = list(q_positions)
            kv_positions = list(kv_positions)
            if not q_positions:
                return
            if not kv_positions:
                raise RuntimeError('actor attention segment has an empty KV sequence')
            q_local = torch.tensor(q_positions, device=device, dtype=torch.long)
            kv_local = torch.tensor(kv_positions, device=device, dtype=torch.long)
            block_mask = build_streaming_block_mask(
                state,
                q_local,
                kv_local,
                num_heads=1,
            )
            q_global = q_local + sample_start
            kv_global = kv_local + sample_start
            records.append({
                'batched_rows': False,
                'q_indices': q_global,
                'kv_indices': kv_global,
                'attention_mask': block_mask,
            })
            covered_queries.extend(int(value) for value in q_global.detach().cpu().tolist())

        def add_synchronous_branches(prefix, branch_b, branch_a):
            max_branch_length = max(len(branch_b), len(branch_a))
            for offset in range(max_branch_length):
                q_rows = []
                kv_rows = []
                if offset < len(branch_b):
                    q_rows.append(branch_b[offset])
                    kv_rows.append(prefix + branch_b[:offset + 1])
                if offset < len(branch_a):
                    q_rows.append(branch_a[offset])
                    kv_rows.append(prefix + branch_a[:offset + 1])
                if not q_rows:
                    continue
                kv_lengths = {len(row) for row in kv_rows}
                if len(kv_lengths) != 1:
                    raise RuntimeError('synchronous actor branches have unequal KV lengths')
                q_local = torch.tensor(q_rows, device=device, dtype=torch.long).unsqueeze(1)
                kv_local = torch.tensor(kv_rows, device=device, dtype=torch.long)
                block_mask = build_batched_streaming_block_mask(
                    [state] * len(q_rows),
                    q_local,
                    kv_local,
                    num_heads=1,
                )
                q_global = q_local[:, 0] + sample_start
                kv_global = kv_local + sample_start
                records.append({
                    'batched_rows': True,
                    'q_indices': q_global,
                    'kv_indices': kv_global,
                    'attention_mask': block_mask,
                })
                covered_queries.extend(int(value) for value in q_global.detach().cpu().tolist())

        kept = []
        cursor = 0
        sample_branch_count = 0
        while (branch_index < len(b_spans) and int(b_spans[branch_index][0]) < sample_end):
            b_start, b_end = (int(value) - sample_start for value in b_spans[branch_index])
            a_start, a_end = (int(value) - sample_start for value in a_spans[branch_index])
            if not (0 <= b_start < b_end == a_start < a_end <= sample_length):
                raise RuntimeError('invalid actor attention branch pair: '
                                   f'B={(b_start, b_end)} A={(a_start, a_end)}')

            if sample_branch_count == 0:
                shared = list(range(cursor, b_start))
                add_record(shared, kept + shared)
                kept.extend(shared)
            else:
                boundaries = [value for value in local_frame_starts if cursor < value < b_start]
                if len(boundaries) != 1:
                    raise RuntimeError('actor attention expected one frame boundary in '
                                       f'[{cursor}, {b_start}), got {boundaries}')
                frame_start = boundaries[0]
                trailer = list(range(cursor, frame_start))
                add_record(trailer, kept + trailer)
                kept.extend(trailer)
                kept = _evict_actor_visual_positions(state, kept, processed_end=frame_start)
                prefill = list(range(frame_start, b_start))
                add_record(prefill, kept + prefill)
                kept.extend(prefill)

            prefix = list(kept)
            branch_b = list(range(b_start, b_end))
            branch_a = list(range(a_start, a_end))
            add_synchronous_branches(prefix, branch_b, branch_a)
            kept = prefix + branch_b + branch_a
            cursor = a_end
            branch_index += 1
            sample_branch_count += 1

        if sample_branch_count == 0:
            raise RuntimeError('inference-aligned actor sample has no dual turn')
        trailing = list(range(cursor, sample_length))
        add_record(trailing, kept + trailing)

    if branch_index != len(b_spans):
        raise RuntimeError('actor attention branch spans fall outside sample bounds')
    if sorted(covered_queries) != list(range(total_length)):
        raise RuntimeError('actor attention segments do not cover physical queries once: '
                           f'covered={len(covered_queries)} total={total_length}')
    _actor_attention_replay_ctx['schedule'] = {
        'device': device,
        'records': records,
    }
    return records


def _actor_attention_replay_forward(
    original_forward,
    module,
    query,
    key,
    value,
    *,
    scaling,
    softcap,
    head_mask,
    s_aux,
    kwargs,
):
    if not _actor_attention_replay_enabled(module, query, key):
        return None
    records = _build_actor_attention_replay_schedule(query.device)
    outputs = []
    lses = []
    output_positions = []
    saw_lse = None
    for record in records:
        q_indices = record['q_indices']

        def run_segment(
            query_base,
            key_base,
            value_base,
            _record=record,
        ):
            segment_q_indices = _record['q_indices']
            segment_kv_indices = _record['kv_indices']
            if _record['batched_rows']:
                query_rows = query_base[0].index_select(
                    1,
                    segment_q_indices,
                ).permute(1, 0, 2).unsqueeze(2)
                key_rows = torch.stack([key_base[0].index_select(1, row) for row in segment_kv_indices], dim=0)
                value_rows = torch.stack([value_base[0].index_select(1, row) for row in segment_kv_indices], dim=0)
                output, lse = original_forward(
                    module,
                    query_rows,
                    key_rows,
                    value_rows,
                    _record['attention_mask'],
                    scaling=scaling,
                    softcap=softcap,
                    head_mask=head_mask,
                    s_aux=s_aux,
                    **kwargs,
                )
                output = output.permute(1, 0, 2, 3)
                if lse is not None:
                    lse = lse.permute(2, 1, 0)
            else:
                output, lse = original_forward(
                    module,
                    query_base.index_select(2, segment_q_indices),
                    key_base.index_select(2, segment_kv_indices),
                    value_base.index_select(2, segment_kv_indices),
                    _record['attention_mask'],
                    scaling=scaling,
                    softcap=softcap,
                    head_mask=head_mask,
                    s_aux=s_aux,
                    **kwargs,
                )
            return output if lse is None else (output, lse)

        # Full-attention replay keeps its execution-aligned segment boundary
        # even when outer decoder checkpointing is selective. Retaining all
        # long-stream segment activations exceeds the production memory cap.
        should_checkpoint = (
            torch.is_grad_enabled()
            and (query.requires_grad or key.requires_grad or value.requires_grad)
        )
        result = (
            activation_checkpoint(
                run_segment,
                query,
                key,
                value,
                use_reentrant=False,
                preserve_rng_state=False,
            ) if should_checkpoint else run_segment(query, key, value))
        if isinstance(result, tuple):
            output, lse = result
        else:
            output, lse = result, None
        outputs.append(output)
        output_positions.append(q_indices)
        has_lse = lse is not None
        if saw_lse is None:
            saw_lse = has_lse
        elif saw_lse != has_lse:
            raise RuntimeError('actor attention segments returned inconsistent LSEs')
        if has_lse:
            lses.append(lse)
    if not outputs:
        raise RuntimeError('actor attention replay produced no segments')
    output = torch.cat(outputs, dim=1)
    positions = torch.cat(output_positions)
    order = torch.argsort(positions)
    output = output.index_select(1, order)
    if int(output.shape[1]) != int(query.shape[2]):
        raise RuntimeError('actor attention output length differs from input sequence')
    lse = (torch.cat(lses, dim=-1).index_select(-1, order) if saw_lse else None)
    return output, lse


def _build_actor_execution_schedule(device: torch.device):
    cached = _actor_attention_replay_ctx.get('execution_schedule')
    if cached is not None:
        if cached['device'] != device:
            raise RuntimeError('actor execution schedule device changed during replay')
        return cached

    from .linear_fork import get_linear_fork_ctx

    linear_ctx = get_linear_fork_ctx()
    if linear_ctx is None or not linear_ctx.get('b_spans'):
        raise RuntimeError('inference-aligned actor execution lacks dual fork metadata')
    input_ids = _actor_attention_replay_ctx.get('input_ids')
    if input_ids is None or input_ids.ndim != 2:
        raise RuntimeError('actor execution lacks physical input_ids')
    total_length = int(input_ids.shape[1])
    sample_spans = list(linear_ctx['sample_spans'])
    b_spans = list(linear_ctx['b_spans'])
    a_spans = list(linear_ctx['a_spans'])
    frame_starts = sorted(int(value) for value in linear_ctx['frame_starts'])
    records = []
    covered_queries = []
    branch_index = 0

    def add_sequence(start, end):
        if start >= end:
            return
        q_indices = torch.arange(start, end, device=device, dtype=torch.long)
        records.append({
            'batched_rows': False,
            'q_positions': tuple(range(start, end)),
            'q_indices': q_indices,
        })
        covered_queries.extend(range(start, end))

    def add_synchronous_branches(b_start, b_end, a_start, a_end):
        max_length = max(b_end - b_start, a_end - a_start)
        for offset in range(max_length):
            q_rows = []
            if b_start + offset < b_end:
                q_rows.append(b_start + offset)
            if a_start + offset < a_end:
                q_rows.append(a_start + offset)
            q_indices = torch.tensor(q_rows, device=device, dtype=torch.long)
            records.append({
                'batched_rows': True,
                'q_positions': tuple(q_rows),
                'q_indices': q_indices,
            })
            covered_queries.extend(q_rows)

    for sample_start, sample_end in sample_spans:
        sample_start, sample_end = int(sample_start), int(sample_end)
        cursor = sample_start
        sample_branch_count = 0
        while (branch_index < len(b_spans) and int(b_spans[branch_index][0]) < sample_end):
            b_start, b_end = map(int, b_spans[branch_index])
            a_start, a_end = map(int, a_spans[branch_index])
            if not (sample_start <= b_start < b_end == a_start < a_end <= sample_end):
                raise RuntimeError('invalid actor execution branch pair: '
                                   f'B={(b_start, b_end)} A={(a_start, a_end)} '
                                   f'sample={(sample_start, sample_end)}')
            if sample_branch_count == 0:
                add_sequence(cursor, b_start)
            else:
                boundaries = [value for value in frame_starts if cursor < value < b_start]
                if len(boundaries) != 1:
                    raise RuntimeError('actor execution expected one frame boundary in '
                                       f'[{cursor}, {b_start}), got {boundaries}')
                frame_start = boundaries[0]
                add_sequence(cursor, frame_start)
                add_sequence(frame_start, b_start)
            add_synchronous_branches(b_start, b_end, a_start, a_end)
            cursor = a_end
            branch_index += 1
            sample_branch_count += 1

        if sample_branch_count == 0:
            raise RuntimeError('inference-aligned actor sample has no dual turn')
        add_sequence(cursor, sample_end)

    if branch_index != len(b_spans):
        raise RuntimeError('actor execution branch spans fall outside sample bounds')
    if sorted(covered_queries) != list(range(total_length)):
        raise RuntimeError('actor execution segments do not cover physical queries once: '
                           f'covered={len(covered_queries)} total={total_length}')
    positions = torch.cat([record['q_indices'] for record in records])
    schedule = {
        'device': device,
        'records': records,
        'order': torch.argsort(positions),
    }
    _actor_attention_replay_ctx['execution_schedule'] = schedule
    return schedule


def _actor_unary_execution_replay(original_forward, x):
    if os.environ.get('MSS_DUAL_ACTOR_LINEAR_MODE', '').strip().lower() != 'inference_aligned':
        return None
    input_ids = _actor_attention_replay_ctx.get('input_ids')
    if (input_ids is None or x.ndim < 3 or int(x.shape[0]) != 1 or int(x.shape[1]) != int(input_ids.shape[1])):
        return None

    schedule = _build_actor_execution_schedule(x.device)
    outputs = []
    for record in schedule['records']:
        q_indices = record['q_indices']
        if record['batched_rows']:
            segment = x[0].index_select(0, q_indices).unsqueeze(1)
            output = original_forward(segment).transpose(0, 1)
        else:
            segment = x.index_select(1, q_indices)
            output = original_forward(segment)
        outputs.append(output)
    return torch.cat(outputs, dim=1).index_select(1, schedule['order'])


class _ActorLinearFusedBackward(torch.autograd.Function):
    """Keep production-shaped Linear forwards but fuse their derivative.

    Forward calls retain the exact streaming row groups. Since this path is
    restricted to frozen projections, backward needs only the row-independent
    input derivative and can evaluate it in one full-row matrix product.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, original_forward, schedule):
        ctx.save_for_backward(weight)
        outputs = []
        for record in schedule['records']:
            q_indices = record['q_indices']
            if record['batched_rows']:
                segment = x[0].index_select(0, q_indices).unsqueeze(1)
                output = original_forward(segment).transpose(0, 1)
            else:
                segment = x.index_select(1, q_indices)
                output = original_forward(segment)
            outputs.append(output)
        return torch.cat(outputs, dim=1).index_select(1, schedule['order'])

    @staticmethod
    def backward(ctx, grad_output):
        (weight, ) = ctx.saved_tensors
        grad_x = torch.nn.functional.linear(grad_output, weight.transpose(0, 1))
        return grad_x, None, None, None, None


class _ActorTrainableLinearFusedInputBackward(torch.autograd.Function):
    """Fuse Linear input derivatives while preserving weight-reduction order.

    The accepted use is a bias-free trainable LoRA projection. Its input
    derivative is row-independent and bitwise stable when evaluated over all
    physical rows. Weight derivatives remain shape-sensitive, so they are
    computed in the original execution groups and accumulated in the reverse
    autograd traversal order.
    """

    @staticmethod
    def forward(ctx, x, weight, original_forward, schedule):
        ctx.save_for_backward(x, weight)
        ctx.schedule = schedule
        outputs = []
        for record in schedule['records']:
            q_indices = record['q_indices']
            if record['batched_rows']:
                segment = x[0].index_select(0, q_indices).unsqueeze(1)
                output = original_forward(segment).transpose(0, 1)
            else:
                segment = x.index_select(1, q_indices)
                output = original_forward(segment)
            outputs.append(output)
        return torch.cat(outputs, dim=1).index_select(1, schedule['order'])

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = torch.nn.functional.linear(grad_output, weight.transpose(0, 1))
        grad_weight = None
        for record in reversed(ctx.schedule['records']):
            q_indices = record['q_indices']
            x_segment = x.index_select(1, q_indices)
            grad_segment = grad_output.index_select(1, q_indices)
            contribution = grad_segment.flatten(0, -2).transpose(0, 1).matmul(
                x_segment.flatten(0, -2))
            if grad_weight is None:
                grad_weight = contribution
            else:
                grad_weight.add_(contribution)
        if grad_weight is None:
            raise RuntimeError('actor Linear execution schedule is empty')
        return grad_x, grad_weight, None, None


def _actor_linear_execution_replay(module, original_forward, x):
    fuse_frozen = os.environ.get(_ACTOR_FUSED_FROZEN_LINEAR_BACKWARD_ENV) == '1'
    fuse_trainable_input = (
        os.environ.get(_ACTOR_FUSED_TRAINABLE_LINEAR_INPUT_BACKWARD_ENV) == '1'
    )
    is_trainable = (module.weight.requires_grad
                    or (module.bias is not None and module.bias.requires_grad))
    if ((is_trainable and not fuse_trainable_input)
            or (not is_trainable and not fuse_frozen)):
        return _actor_unary_execution_replay(original_forward, x)
    input_ids = _actor_attention_replay_ctx.get('input_ids')
    if (os.environ.get('MSS_DUAL_ACTOR_LINEAR_MODE', '').strip().lower() != 'inference_aligned'
            or input_ids is None or x.ndim < 3 or int(x.shape[0]) != 1
            or int(x.shape[1]) != int(input_ids.shape[1])):
        return None
    schedule = _build_actor_execution_schedule(x.device)
    if is_trainable:
        if module.bias is not None:
            return _actor_unary_execution_replay(original_forward, x)
        return _ActorTrainableLinearFusedInputBackward.apply(
            x,
            module.weight,
            original_forward,
            schedule,
        )
    return _ActorLinearFusedBackward.apply(
        x,
        module.weight,
        module.bias,
        original_forward,
        schedule,
    )


class _ActorSegmentedRMSNormBackward(torch.autograd.Function):
    """Replay RMSNorm forward and derivative in identical execution groups."""

    @staticmethod
    def forward(ctx, x, weight, eps, original_forward, schedule):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.schedule = schedule
        outputs = []
        for record in schedule['records']:
            q_indices = record['q_indices']
            if record['batched_rows']:
                segment = x[0].index_select(0, q_indices).unsqueeze(1)
                output = original_forward(segment).transpose(0, 1)
            else:
                segment = x.index_select(1, q_indices)
                output = original_forward(segment)
            outputs.append(output)
        return torch.cat(outputs, dim=1).index_select(1, schedule['order'])

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        width = x.shape[-1]
        scale = 1.0 + weight.float()
        outputs = []
        for record in ctx.schedule['records']:
            q_indices = record['q_indices']
            if record['batched_rows']:
                x_segment = x[0].index_select(0, q_indices).unsqueeze(1)
                grad_segment = grad_output[0].index_select(
                    0, q_indices).unsqueeze(1)
            else:
                x_segment = x.index_select(1, q_indices)
                grad_segment = grad_output.index_select(1, q_indices)
            x_float = x_segment.float()
            scaled_grad = grad_segment.float() * scale
            reciprocal_rms = torch.rsqrt(
                x_float.pow(2).mean(-1, keepdim=True) + ctx.eps)
            dot = (scaled_grad * x_float).sum(-1, keepdim=True)
            grad_x = (
                scaled_grad * reciprocal_rms
                - x_float * (dot * reciprocal_rms.pow(3) / width)
            ).type_as(x_segment)
            outputs.append(
                grad_x.transpose(0, 1) if record['batched_rows'] else grad_x)
        grad_x = torch.cat(outputs, dim=1).index_select(
            1, ctx.schedule['order'])
        return grad_x, None, None, None, None


def _actor_rmsnorm_execution_replay(module, original_forward, x):
    if (os.environ.get(_ACTOR_SEGMENTED_RMSNORM_BACKWARD_ENV) != '1'
            or module.weight.requires_grad):
        return _actor_unary_execution_replay(original_forward, x)
    input_ids = _actor_attention_replay_ctx.get('input_ids')
    if (os.environ.get('MSS_DUAL_ACTOR_LINEAR_MODE', '').strip().lower()
            != 'inference_aligned' or input_ids is None or x.ndim < 3
            or int(x.shape[0]) != 1
            or int(x.shape[1]) != int(input_ids.shape[1])):
        return None
    schedule = _build_actor_execution_schedule(x.device)
    return _ActorSegmentedRMSNormBackward.apply(
        x,
        module.weight,
        module.eps,
        original_forward,
        schedule,
    )


class _ActorFusedGatedRMSNormBackward(torch.autograd.Function):
    """Preserve grouped FLA forwards and fuse the frozen-norm derivative.

    FLA keeps an FP32 reciprocal RMS for backward. Recomputing that value in
    a differently sized full-row forward can change a few BF16 gradients, even
    when the rounded forward output is identical. This function therefore
    saves each production group's original reciprocal RMS, restores physical
    row order, and runs one fused backward with those exact statistics.
    """

    @staticmethod
    def forward(ctx, x, gate, weight, eps, activation, schedule):
        from fla.modules.fused_norm_gate import layer_norm_gated_fwd

        x = x.contiguous()
        gate = gate.contiguous()
        token_count = int(schedule['order'].numel())
        if token_count <= 0 or int(x.shape[0]) % token_count != 0:
            raise RuntimeError('invalid actor gated RMSNorm physical layout')
        value_heads = int(x.shape[0]) // token_count
        head_dim = int(x.shape[1])
        x_by_token = x.reshape(token_count, value_heads, head_dim)
        gate_by_token = gate.reshape(token_count, value_heads, head_dim)
        outputs = []
        reciprocal_rms = []
        for record in schedule['records']:
            q_indices = record['q_indices']
            rows = int(q_indices.numel())
            x_segment = x_by_token.index_select(0, q_indices).reshape(-1, head_dim)
            gate_segment = gate_by_token.index_select(0, q_indices).reshape(-1, head_dim)
            output, _, rstd, _ = layer_norm_gated_fwd(
                x=x_segment,
                g=gate_segment,
                weight=weight,
                bias=None,
                activation=activation,
                eps=eps,
                is_rms_norm=True,
            )
            outputs.append(output.reshape(rows, value_heads, head_dim))
            reciprocal_rms.append(rstd.reshape(rows, value_heads))
        order = schedule['order']
        output = torch.cat(outputs, dim=0).index_select(0, order).reshape_as(x)
        rstd = torch.cat(reciprocal_rms, dim=0).index_select(0, order).reshape(-1)
        ctx.save_for_backward(x, gate, weight, rstd)
        ctx.activation = activation
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from fla.modules.fused_norm_gate import layer_norm_gated_bwd

        x, gate, weight, rstd = ctx.saved_tensors
        grad_x, grad_gate, _, _, _ = layer_norm_gated_bwd(
            dy=grad_output.contiguous(),
            x=x,
            g=gate,
            weight=weight,
            bias=None,
            activation=ctx.activation,
            eps=ctx.eps,
            rstd=rstd,
            is_rms_norm=True,
            x_dtype=x.dtype,
        )
        return grad_x, grad_gate, None, None, None, None


def _actor_gated_norm_execution_replay(module, original_forward, x, gate):
    """Replay the linear-attention gated norm in production token groups.

    Qwen3.5 flattens [batch, sequence, value_heads, head_dim] to
    [batch * sequence * value_heads, head_dim] before calling FLA's fused
    RMSNorm+gate. The fused Triton kernel is numerically shape-dependent, so
    a one-shot actor sequence does not reproduce the rollout's prefill and
    synchronous dual-branch decode shapes even though every row is
    mathematically independent. Regroup the flattened rows by the same token
    execution schedule used by production inference.
    """
    if os.environ.get('MSS_DUAL_ACTOR_LINEAR_MODE', '').strip().lower() != 'inference_aligned':
        return None
    input_ids = _actor_attention_replay_ctx.get('input_ids')
    if (input_ids is None or x.ndim != 2 or gate.ndim != 2 or x.shape != gate.shape):
        return None
    token_count = int(input_ids.shape[1])
    row_count = int(x.shape[0])
    if token_count <= 0 or row_count % token_count != 0:
        return None
    value_heads = row_count // token_count
    if value_heads <= 0:
        return None

    schedule = _build_actor_execution_schedule(x.device)
    fuse_backward = (
        os.environ.get(_ACTOR_FUSED_GATED_RMSNORM_BACKWARD_ENV) == '1'
        and module.weight is not None
        and not module.weight.requires_grad
        and module.bias is None
    )
    if fuse_backward:
        return _ActorFusedGatedRMSNormBackward.apply(
            x,
            gate,
            module.weight,
            module.eps,
            module.activation,
            schedule,
        )
    head_dim = int(x.shape[1])
    x_by_token = x.reshape(token_count, value_heads, head_dim)
    gate_by_token = gate.reshape(token_count, value_heads, head_dim)
    outputs = []
    for record in schedule['records']:
        q_indices = record['q_indices']
        x_segment = x_by_token.index_select(0, q_indices).reshape(-1, head_dim)
        gate_segment = gate_by_token.index_select(0, q_indices).reshape(-1, head_dim)
        output = original_forward(x_segment, gate_segment)
        outputs.append(output.reshape(int(q_indices.numel()), value_heads, head_dim))
    return torch.cat(outputs, dim=0).index_select(0, schedule['order']).reshape_as(x)


def _actor_lm_head_execution_replay(original_forward, hidden_states):
    """Run the selective actor LM head in production decode row groups."""
    if os.environ.get('MSS_DUAL_ACTOR_LINEAR_MODE', '').strip().lower() != 'inference_aligned':
        return None
    input_ids = _actor_attention_replay_ctx.get('input_ids')
    if input_ids is None:
        return None
    row_groups = _actor_lm_head_execution_row_groups(hidden_states)
    outputs = []
    emitted_rows = []
    for rows in row_groups:
        row_indices = torch.tensor(rows, device=hidden_states.device, dtype=torch.long)
        segment = hidden_states[0].index_select(0, row_indices).unsqueeze(1)
        segment_output = original_forward(segment)
        if (segment_output.ndim != 3 or tuple(segment_output.shape[:2]) != (len(rows), 1)):
            raise RuntimeError('actor LM head returned an unexpected grouped shape: '
                               f'{tuple(segment_output.shape)}')
        outputs.append(segment_output[:, 0, :])
        emitted_rows.extend(rows)

    output = torch.cat(outputs, dim=0)
    restore_order = torch.argsort(torch.tensor(emitted_rows, device=hidden_states.device, dtype=torch.long))
    return output.index_select(0, restore_order).unsqueeze(0)


def _actor_lm_head_execution_row_groups(hidden_states):
    """Map selected LM-head rows to the exact production call groups."""
    positions = _actor_attention_replay_ctx.get('lm_head_positions')
    if positions is None:
        raise RuntimeError('inference-aligned actor LM head lacks physical positions')
    if (hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1
            or int(hidden_states.shape[1]) != int(positions.numel())):
        raise RuntimeError('inference-aligned actor LM head shape mismatch: '
                           f'hidden={tuple(hidden_states.shape)} '
                           f'positions={tuple(positions.shape)}')

    position_rows = {int(position): row for row, position in enumerate(positions.detach().cpu().tolist())}
    schedule = _build_actor_execution_schedule(hidden_states.device)
    row_groups = []
    emitted_rows = []
    for record in schedule['records']:
        q_positions = record['q_positions']
        rows = [position_rows.get(position) for position in q_positions]
        present = [row is not None for row in rows]
        if not any(present):
            continue
        if not all(present):
            raise RuntimeError('actor LM-head positions omit a production companion row: '
                               f'group={q_positions} present={present}')
        row_groups.append(tuple(int(row) for row in rows))
        emitted_rows.extend(int(row) for row in rows)

    expected_rows = list(range(int(positions.numel())))
    if sorted(emitted_rows) != expected_rows:
        raise RuntimeError('actor LM-head execution groups do not cover selected rows once: '
                           f'emitted={len(emitted_rows)} expected={len(expected_rows)}')
    return tuple(row_groups)


def actor_lm_head_execution_plan(module, hidden_states):
    """Return the exact grouped projection used by actor LM-head replay.

    This is consumed by the memory-bounded RL policy tail.  It intentionally
    fails closed unless the module was patched and the current actor forward
    installed authoritative physical row metadata.
    """
    original_forward = getattr(module, '_mss_actor_execution_original_forward', None)
    if original_forward is None:
        raise RuntimeError('actor LM head is not execution-aligned')
    if os.environ.get('MSS_DUAL_ACTOR_LINEAR_MODE', '').strip().lower() != 'inference_aligned':
        raise RuntimeError('actor LM-head execution plan requires inference_aligned mode')
    return original_forward, _actor_lm_head_execution_row_groups(hidden_states)


class _ActorFrozenLmHeadGroupedInputBackward(torch.autograd.Function):
    """Keep grouped LM-head forwards and an exact, memory-light derivative."""

    @staticmethod
    def forward(ctx, hidden_states, weight, projection_forward, group_sizes):
        if hidden_states.ndim != 2 or weight.ndim != 2:
            raise ValueError('fused actor LM head expects [rows, hidden] and '
                             '[vocab, hidden] tensors')
        group_sizes = tuple(int(size) for size in group_sizes)
        if (not group_sizes or any(size < 1 for size in group_sizes)
                or sum(group_sizes) != int(hidden_states.shape[0])):
            raise ValueError('fused actor LM-head groups do not cover hidden rows')
        if sum(group_sizes) > _ACTOR_FUSED_LM_HEAD_MAX_ROWS:
            raise ValueError('fused actor LM-head block exceeds the exact '
                             f'{_ACTOR_FUSED_LM_HEAD_MAX_ROWS}-row limit')
        if int(weight.shape[1]) != int(hidden_states.shape[1]):
            raise ValueError('fused actor LM-head hidden width mismatch')

        ctx.save_for_backward(weight)
        ctx.group_sizes = group_sizes
        outputs = []
        cursor = 0
        for group_size in group_sizes:
            segment = hidden_states[cursor:cursor + group_size].unsqueeze(1)
            output = projection_forward(segment)
            if (output.ndim != 3
                    or tuple(output.shape[:2]) != (group_size, 1)):
                raise RuntimeError('actor LM head returned an unexpected grouped '
                                   f'shape: {tuple(output.shape)}')
            outputs.append(output[:, 0, :])
            cursor += group_size
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        (weight, ) = ctx.saved_tensors
        # A single GEMM over all rows is mathematically equivalent, but CUDA
        # selects a different reduction kernel for the larger M dimension.
        # Production BF16 gradients then differ by a few low bits. Preserve
        # the exact physical execution shapes instead of accepting a tolerance.
        grad_parts = []
        cursor = 0
        for group_size in ctx.group_sizes:
            grad_parts.append(torch.nn.functional.linear(
                grad_output[cursor:cursor + group_size],
                weight.transpose(0, 1),
            ))
            cursor += group_size
        return torch.cat(grad_parts, dim=0), None, None, None


def actor_fused_lm_head_input_backward_enabled(module) -> bool:
    """Validate and report whether the exact frozen-LM-head path is active."""
    enabled = (
        os.environ.get(_ACTOR_FUSED_LM_HEAD_INPUT_BACKWARD_ENV) == '1'
    )
    if not enabled:
        return False
    if not isinstance(module, torch.nn.Linear):
        raise RuntimeError('fused actor LM-head input backward requires '
                           'torch.nn.Linear')
    if module.weight.requires_grad:
        raise RuntimeError('fused actor LM-head input backward requires a '
                           'frozen weight')
    if module.bias is not None and module.bias.requires_grad:
        raise RuntimeError('fused actor LM-head input backward requires a '
                           'frozen bias')
    return True


def actor_fused_lm_head_projection(
    hidden_states,
    module,
    projection_forward,
    group_sizes,
):
    """Project at most eight rows with exact grouped forward/backward.

    The public name is retained for checkpoint/CLI compatibility. Physical
    groups are deliberately not fused into a larger GEMM.
    """
    if not actor_fused_lm_head_input_backward_enabled(module):
        raise RuntimeError('fused actor LM-head projection is not enabled')
    return _ActorFrozenLmHeadGroupedInputBackward.apply(
        hidden_states,
        module.weight,
        projection_forward,
        tuple(group_sizes),
    )


def _patch_actor_lm_head_execution(model) -> int:
    module = model.get_output_embeddings()
    if module is None:
        raise RuntimeError('Qwen actor model has no output embedding')
    if getattr(module, '_mss_actor_execution_patched', False):
        return 0
    original_forward = module.forward

    def replay_forward(hidden_states):
        deferred = _actor_deferred_lm_head_ctx.get()
        if deferred is not None:
            if hidden_states.ndim != 3:
                raise RuntimeError('deferred actor LM head expects [batch, rows, hidden], '
                                   f'got {tuple(hidden_states.shape)}')
            deferred['consumed'] = True
            return hidden_states
        replayed = _actor_lm_head_execution_replay(original_forward, hidden_states)
        return (original_forward(hidden_states) if replayed is None else replayed)

    module.forward = replay_forward
    module._mss_actor_execution_patched = True
    module._mss_actor_execution_original_forward = original_forward
    return 1


def _patch_actor_rmsnorm_execution(model) -> int:
    patched = 0
    for module in model.modules():
        if type(module).__name__ != 'Qwen3_5RMSNorm':
            continue
        if getattr(module, '_mss_actor_execution_patched', False):
            continue
        original_forward = module.forward

        def replay_forward(x, _module=module, _original=original_forward):
            replayed = _actor_rmsnorm_execution_replay(
                _module, _original, x)
            return _original(x) if replayed is None else replayed

        module.forward = replay_forward
        module._mss_actor_execution_patched = True
        module._mss_actor_execution_original_forward = original_forward
        patched += 1
    return patched


def _actor_visual_per_image_replay(
    original_forward,
    hidden_states,
    grid_thw,
    kwargs,
):
    """Run actor vision exactly as streaming rollout: one image per call."""
    if os.environ.get('MSS_DUAL_ACTOR_LINEAR_MODE', '').strip().lower() != 'inference_aligned':
        return None
    if _actor_attention_replay_ctx.get('input_ids') is None:
        return None
    if grid_thw is None or grid_thw.ndim != 2:
        return None
    image_count = int(grid_thw.shape[0])
    if kwargs.get('output_hidden_states') or kwargs.get('output_attentions'):
        raise RuntimeError('per-image actor vision replay does not expose intermediate '
                           'vision outputs')

    precomputed_state = _actor_precomputed_visual_ctx.get()
    if precomputed_state is not None:
        if precomputed_state['consumed']:
            raise RuntimeError('actor forward consumed precomputed visual features more than once')
        precomputed = precomputed_state['features']
        if len(precomputed) != image_count:
            raise RuntimeError('precomputed actor image count does not match grids: '
                               f'{len(precomputed)} != {image_count}')
        reference = precomputed[0]
        if reference.ndim != 2:
            raise RuntimeError('precomputed actor image features must be rank two')
        for index, feature in enumerate(precomputed):
            if (feature.ndim != 2 or feature.shape[1:] != reference.shape[1:]
                    or feature.device != hidden_states.device
                    or feature.dtype != hidden_states.dtype):
                raise RuntimeError('invalid precomputed actor image feature at '
                                   f'{index}: shape={tuple(feature.shape)} '
                                   f'dtype={feature.dtype} device={feature.device}')
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        precomputed_state['consumed'] = True
        return BaseModelOutputWithPooling(
            last_hidden_state=None,
            pooler_output=torch.cat(precomputed, dim=0),
        )

    if image_count <= 1:
        return None

    patch_counts = [int(value) for value in grid_thw.prod(dim=-1).detach().cpu().tolist()]
    if sum(patch_counts) != int(hidden_states.shape[0]):
        raise RuntimeError('actor vision patch rows do not match image grids: '
                           f'rows={hidden_states.shape[0]} patches={patch_counts}')

    outputs = []
    start = 0
    for image_index, patch_count in enumerate(patch_counts):
        end = start + patch_count
        outputs.append(original_forward(
            hidden_states[start:end],
            grid_thw[image_index:image_index + 1],
            **kwargs,
        ))
        start = end

    if any(not hasattr(output, 'last_hidden_state') or not hasattr(output, 'pooler_output') for output in outputs):
        raise RuntimeError('actor vision replay requires return_dict model outputs')
    if any(
            getattr(output, 'hidden_states', None) is not None or getattr(output, 'attentions', None) is not None
            for output in outputs):
        raise RuntimeError('actor vision replay received unexpected intermediate outputs')
    output_type = type(outputs[0])
    if any(type(output) is not output_type for output in outputs):
        raise RuntimeError('actor vision replay output types changed')
    return output_type(
        last_hidden_state=torch.cat([output.last_hidden_state for output in outputs], dim=0),
        pooler_output=torch.cat([output.pooler_output for output in outputs], dim=0),
    )


def _patch_actor_visual_per_image_execution(model) -> int:
    patched = 0
    for module in model.modules():
        if type(module).__name__ != 'Qwen3_5VisionModel':
            continue
        if getattr(module, '_mss_actor_per_image_patched', False):
            continue
        original_forward = module.forward

        def replay_forward(hidden_states, grid_thw, _original=original_forward, **kwargs):
            replayed = _actor_visual_per_image_replay(_original, hidden_states, grid_thw, kwargs)
            return (_original(hidden_states, grid_thw, **kwargs) if replayed is None else replayed)

        module.forward = replay_forward
        module._mss_actor_per_image_patched = True
        module._mss_actor_per_image_original_forward = original_forward
        patched += 1
    return patched


def _patch_actor_shape_sensitive_projections(model) -> int:
    """Replay only language GEMMs proven to depend on execution row count.

    L20Z BF16 checks isolate the unstable reductions to the 9216 -> 2560 MLP
    base down projections and the 4096 -> 2560 token-mixer output
    projections. A real rollout/actor layer trace additionally isolates the
    9216 -> rank A/B LoRA inputs of the MLP down projection. All other
    language projections keep their fused path.
    """
    patched = 0
    for module in model.modules():
        module_type = type(module).__name__
        if module_type == 'Qwen3_5MLP':
            projection = module.down_proj
            base = getattr(projection, 'base', projection)
            projections = [base]
            if type(projection).__name__ == 'DualLoraLinear':
                projections.extend([
                    projection.lora_A_down,
                    projection.lora_B_down,
                ])
        elif module_type in ('Qwen3_5Attention', 'Qwen3_5GatedDeltaNet'):
            base = module.o_proj if module_type == 'Qwen3_5Attention' \
                else module.out_proj
            projections = [base]
        else:
            continue
        for current in projections:
            if not isinstance(current, torch.nn.Linear):
                raise RuntimeError('shape-sensitive Qwen3.5 projection is not Linear')
            if getattr(current, '_mss_actor_projection_execution_patched', False):
                continue
            original_forward = current.forward

            def replay_forward(x, _module=current, _original=original_forward):
                replayed = _actor_linear_execution_replay(_module, _original, x)
                return _original(x) if replayed is None else replayed

            current.forward = replay_forward
            current._mss_actor_projection_execution_patched = True
            current._mss_actor_projection_execution_original_forward = (original_forward)
            patched += 1
    return patched


def _patch_actor_gated_norm_execution(model) -> int:
    patched = 0
    for module in model.modules():
        if type(module).__name__ != 'FusedRMSNormGated':
            continue
        if getattr(module, '_mss_actor_execution_patched', False):
            continue
        original_forward = module.forward

        def replay_forward(
            x,
            gate,
            residual=None,
            prenorm=False,
            residual_in_fp32=False,
            _module=module,
            _original=original_forward,
        ):
            replayed = None
            if residual is None and not prenorm and not residual_in_fp32:
                replayed = _actor_gated_norm_execution_replay(
                    _module, _original, x, gate)
            if replayed is not None:
                return replayed
            return _original(
                x,
                gate,
                residual=residual,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
            )

        module.forward = replay_forward
        module._mss_actor_execution_patched = True
        module._mss_actor_execution_original_forward = original_forward
        patched += 1
    return patched


def patch_flex_kernel_options() -> None:
    """Monkey-patch ``transformers.integrations.flex_attention.flex_attention_forward``
    so every call injects the env-driven ``kernel_options`` dict (adding to /
    overriding whatever kwargs the caller passed).

    Idempotent.
    """
    if _kernel_opts_patched['installed']:
        return
    from transformers.integrations import flex_attention as _tflex
    original_forward = _tflex.flex_attention_forward

    def wrapped_flex_forward(module,
                             query,
                             key,
                             value,
                             attention_mask,
                             scaling=None,
                             softcap=None,
                             head_mask=None,
                             s_aux=None,
                             **kwargs):
        override = _env_flex_kernel_options()
        if override is not None:
            existing = kwargs.get('kernel_options') or {}
            merged = {**existing, **override}
            # Ensure the FORCE flag so dynamic=True compile picks prefill kernel
            merged.setdefault('FORCE_USE_FLEX_ATTENTION', True)
            kwargs['kernel_options'] = merged
        # Actor full attention must execute with the same physical prefill,
        # synchronous B/A decode, merge, trailer, and eviction boundaries as
        # production rollout. This is part of inference_aligned semantics,
        # not an optional switch: one-shot BF16 reductions change policy logp.
        replayed = _actor_attention_replay_forward(
            original_forward,
            module,
            query,
            key,
            value,
            scaling=scaling,
            softcap=softcap,
            head_mask=head_mask,
            s_aux=s_aux,
            kwargs=kwargs,
        )
        return replayed if replayed is not None else original_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            softcap=softcap,
            head_mask=head_mask,
            s_aux=s_aux,
            **kwargs)

    _tflex.flex_attention_forward = wrapped_flex_forward
    # Also patch the registered "flex_attention" implementation in
    # ALL_ATTENTION_FUNCTIONS so callers dispatched via that registry pick
    # up our wrapper.
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if 'flex_attention' in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS['flex_attention'] = wrapped_flex_forward
    except Exception:  # noqa: BLE001
        pass
    _kernel_opts_patched.update(installed=True, original_forward=original_forward)
    log.info('[step_mask] patched flex_attention_forward to inject '
             'env-driven kernel_options (MSS_FLEX_KERNEL_OPTS)')


def force_dynamic_flex_compile() -> None:
    """Patch transformers' ``WrappedFlexAttention.__init__`` so the singleton
    compiles ``flex_attention`` with ``dynamic=True``.

    Why this is necessary
    ---------------------
    transformers/integrations/flex_attention.py:46-84 defines a
    ``WrappedFlexAttention`` singleton that compiles ``flex_attention`` once.
    On PyTorch >= 2.7 (we run 2.10) it hits the unconditional fallback at line
    79:

        self._compiled_flex_attention = torch.compile(flex_attention)

    which uses ``dynamic=None`` (PyTorch's default = effectively static for
    flex_attention). Every new ``(Q_LEN, KV_LEN)`` BlockMask shape triggers
    a fresh Inductor compile (5-60 seconds each). Catastrophic for streaming
    inference where shapes change every forward.

    Empirical evidence (``tools/probe_flex_dynamic.py`` on L20Z, PyTorch 2.10):
      dynamic=False, Q=1 KV={256, 512, 1024, 1025}: 855-1113 ms each (recompile)
      dynamic=True , Q=1 KV={256, 512, 1024, 1025}: 1218 ms first, then 0.3 ms each

    Fix: rewrite ``__init__`` to use ``dynamic=True`` regardless of torch
    version. Training shapes are fixed so the difference is in the noise
    (~10% kernel speed at most, since dynamic shape compilation produces
    slightly less optimized kernels). Inference shapes vary so the
    difference is decisive.

    Idempotent: safe to call multiple times.
    """
    if _dynamic_flex_patched['installed']:
        return
    from transformers.integrations import flex_attention as _tflex
    original_init = _tflex.WrappedFlexAttention.__init__
    flex_attention = _tflex.flex_attention

    # Mirror the disable decorator on the original so monkey-patched __init__
    # does not get traced by dynamo on the first call.
    #
    # Why dynamic=True is safe ONLY WITH ``FORCE_USE_FLEX_ATTENTION=True``:
    #
    # flex_attention has TWO internal kernels:
    #   * decode kernel  (Q_LEN == 1, sometimes Q_LEN < 128) -- flex_decoding.py
    #   * prefill kernel (everything else)                   -- flex_attention.py
    #
    # ``dynamic=True`` alone: a single compile may bind to one kernel path. If
    # the first traced shape has Q_LEN=1 it lowers to the decode kernel, and
    # Q_LEN>=8 (Qwen3.5 GQA=4) hits AssertionError at flex_decoding.py:286
    # because ``seq_len_q * gqa_shared_heads <= BLOCK_M`` no longer holds.
    #
    # We side-step this by setting ``FORCE_USE_FLEX_ATTENTION=True`` in the
    # per-call kernel_options at *every* attention call (see
    # inference.py and infer_stream_masked.py). With that flag, every call --
    # including Q_LEN=1 -- routes through the prefill kernel, eliminating the
    # mixing problem.
    #
    # ``dynamic=True`` then gives us its real value: a single compile handles
    # any (Q_LEN, KV_LEN) combination. Streaming decode increments KV_LEN by 1
    # every token; without dynamic=True dynamo recompiles per KV_LEN, hits the
    # recompile_limit (default 8) after a few decode steps, and falls back to
    # eager flex_attention -- which materialises the full Q*K^T matrix and is
    # 5-10x slower.
    #
    # Evidence: probe_flex_dynamic_v2.py shows dynamic=True correctly reuses
    # compile across varying KV_LEN at fixed Q_LEN.
    @torch.compiler.disable(recursive=False)
    def _patched_init(self, training):
        if not self._is_flex_compiled or training != self.training:
            self.training = training
            self._compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
            self._is_flex_compiled = True

    _tflex.WrappedFlexAttention.__init__ = _patched_init
    # Reset the singleton state. If the singleton was already created (e.g.,
    # earlier in this process by a training forward), its class-level
    # ``_is_flex_compiled`` is True and our patched ``__init__`` would skip
    # re-compilation. Clearing the flag forces the next ``__init__`` call to
    # build a fresh dynamic-mode compile.
    _tflex.WrappedFlexAttention._is_flex_compiled = False
    _tflex.WrappedFlexAttention._compiled_flex_attention = None
    _dynamic_flex_patched.update(installed=True, original_init=original_init)
    log.info('[step_mask] patched WrappedFlexAttention.__init__ to use '
             'torch.compile(flex_attention, dynamic=True) -- avoids per-shape '
             'recompile during streaming inference')


def patch_qwen3_5_for_step_mask(model, window_frames: int, *, assistant_role_id: int) -> None:
    """Install monkey-patches on ``model``, the qwen3_5 module, and swift's
    Template.pre_forward_hook so that input_ids reaches our BlockMask builder.

    Idempotent: re-calling with the same model is a no-op.

    Parameters
    ----------
    model              : the base ``Qwen3_5ForConditionalGeneration`` instance.
    window_frames      : W; <= 0 disables and this function is a no-op.
    assistant_role_id  : single-token id of the literal text ``"assistant"``;
                         used to filter out the system prompt's literal
                         ``<silent>`` / ``</inner>`` / ``</speak>`` occurrences
                         from the metadata extractor.
    """
    if window_frames <= 0:
        return
    if _patched_state.get('installed'):
        return

    # Force ``torch.compile(flex_attention, dynamic=True)`` so the singleton
    # compile is shared across all (Q_LEN, KV_LEN) shapes. Critical for
    # streaming inference; benign for training. See docstring for evidence.
    force_dynamic_flex_compile()

    # If MSS_FLEX_KERNEL_OPTS is set, inject those kernel_options at every
    # flex_attention_forward call. Needed on GPUs with tight SRAM (e.g. L20Z
    # 228 KB per SM) where the default kernel config exceeds SRAM.
    patch_flex_kernel_options()

    from transformers.models.qwen3_5 import modeling_qwen3_5
    actor_rmsnorm_count = 0
    actor_gated_norm_count = 0
    actor_lm_head_count = 0
    actor_projection_count = 0
    actor_visual_count = 0
    if os.environ.get('MSS_DUAL_ACTOR_LINEAR_MODE', '').strip().lower() == 'inference_aligned':
        actor_rmsnorm_count = _patch_actor_rmsnorm_execution(model)
        actor_gated_norm_count = _patch_actor_gated_norm_execution(model)
        actor_lm_head_count = _patch_actor_lm_head_execution(model)
        actor_projection_count = (_patch_actor_shape_sensitive_projections(model))
        actor_visual_count = _patch_actor_visual_per_image_execution(model)
        if actor_visual_count != 1:
            raise RuntimeError('actor per-image vision replay patch count mismatch: '
                               f'{actor_visual_count} != 1')

        expected_projection_count = 4 * int(getattr(model.config, 'text_config', model.config).num_hidden_layers)
        if actor_projection_count != expected_projection_count:
            raise RuntimeError('actor shape-sensitive projection patch count mismatch: '
                               f'{actor_projection_count} != {expected_projection_count}')
        log.info('[step_mask] actor execution-aligned RMSNorm replay: '
                 f'{actor_rmsnorm_count} modules')
        log.info('[step_mask] actor execution-aligned gated RMSNorm replay: '
                 f'{actor_gated_norm_count} modules')
        log.info('[step_mask] actor execution-aligned BF16 projections: '
                 f'{actor_projection_count} modules')
        log.info('[step_mask] actor per-image vision replay: '
                 f'{actor_visual_count} module')
        log.info('[step_mask] actor execution-aligned LM-head replay: '
                 f'{actor_lm_head_count} module')

    # Force flex_attention for the 8 full-attention layers. The 24 linear
    # layers do not consult ``_attn_implementation`` at all.
    text_config = getattr(model.config, 'text_config', model.config)
    original_attn_impl = text_config._attn_implementation
    text_config._attn_implementation = 'flex_attention'
    log.info(f'[step_mask] forcing text_config._attn_implementation: '
             f'{original_attn_impl!r} -> "flex_attention" (W={window_frames})')

    original_create_causal_mask = modeling_qwen3_5.create_causal_mask

    def patched_create_causal_mask(*, config, inputs_embeds, attention_mask, cache_position, past_key_values,
                                   position_ids):
        ctx = _ctx.get()
        if ctx is None or ctx.get('window_frames', 0) <= 0:
            return original_create_causal_mask(
                config=config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        if ctx.get('streaming'):
            # Graph-decode short-circuit: GraphDecoder has pre-built a static
            # BlockMask and updated its contents in-place outside the captured
            # graph. Return it directly -- skip all host-bound work
            # (numpy->torch transfers, create_block_mask Python loops) that
            # would otherwise be disallowed during stream capture.
            precomputed = ctx.get('precomputed_block_mask')
            if precomputed is not None:
                return precomputed
            # Inference mode: the cumulative state lives in ctx['state']; the
            # caller has already populated q_/kv_token_positions for this
            # forward (= the original-coordinate positions of queries and the
            # KV slots currently in cache). With StreamingCache the
            # kv_token_positions array is sized at max_kv_len and ctx carries
            # ``valid_kv_len`` to bound the populated prefix.
            from .inference import build_streaming_block_mask
            bm = build_streaming_block_mask(
                state=ctx['state'],
                q_token_positions=ctx['q_token_positions'],
                kv_token_positions=ctx['kv_token_positions'],
                num_heads=1,
                valid_kv_len=ctx.get('valid_kv_len'),
            )
            _maybe_print_mask_diagnostics(ctx, bm)
            return bm
        bm = build_step_window_block_mask(
            input_ids=ctx['input_ids'],
            position_ids=position_ids if position_ids is not None else ctx['position_ids'],
            window_frames=ctx['window_frames'],
            assistant_role_id=ctx['assistant_role_id'],
            num_heads=1,
            cu_seqlens=ctx.get('cu_seqlens'),
        )
        _maybe_print_mask_diagnostics(ctx, bm)
        return bm

    modeling_qwen3_5.create_causal_mask = patched_create_causal_mask

    # Wrap swift's Template.pre_forward_hook so we observe input_ids before it
    # is popped from kwargs (after _post_encode converts to inputs_embeds).
    # This is the last place input_ids exists as a tensor in MLLM training.
    # ``labels`` is not available here either -- swift's seq2seq_trainer pops
    # them in compute_loss when ``loss_scale`` is non-None -- which is why
    # build_step_mask_meta now accepts ``assistant_role_id=`` and discovers
    # assistant turns from input_ids alone.
    from swift.template import base as _tpl_base
    original_pre_forward_hook = _tpl_base.Template.pre_forward_hook

    _probed = {'done': False}

    def wrapped_pre_forward_hook(self, model, args, kwargs):
        input_ids = kwargs.get('input_ids')
        time_pos = None
        if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
            p = kwargs['position_ids']
            time_pos = p[0] if (p.dim() == 3 and p.shape[0] == 4) else p
        if input_ids is not None:
            _ctx.set({
                'window_frames': window_frames,
                'input_ids': input_ids,
                'position_ids': time_pos,
                'cu_seqlens': kwargs.get('cu_seq_lens_q'),
                'assistant_role_id': assistant_role_id,
            })
            actor_cu_seqlens = kwargs.get('cu_seq_lens_q')
            if os.environ.get('MSS_DUAL_ACTOR_LINEAR_MODE', '').strip().lower() == 'inference_aligned':
                # Rollout also passes input_ids through this hook.  Actor
                # execution replay is valid only for the authoritative
                # padding-free physical layout emitted by the train collator.
                # Clear first so no prior actor context can leak into rollout.
                _actor_attention_replay_ctx.clear()
                if actor_cu_seqlens is not None:
                    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
                        raise RuntimeError('actor replay requires padding-free [1, L] '
                                           f'input_ids, got {tuple(input_ids.shape)}')
                    lm_head_positions = physical_logits_positions(kwargs.get('logits_to_keep'), input_ids)
                    _actor_attention_replay_ctx.update({
                        'input_ids': input_ids.detach(),
                        'window_frames': window_frames,
                        'assistant_role_id': assistant_role_id,
                        'schedule': None,
                        'execution_schedule': None,
                        'lm_head_positions': lm_head_positions.detach(),
                    })
        if _debug_enabled() and not _probed['done']:
            _probed['done'] = True
            rank = int(os.environ.get('RANK', '0') or 0)
            if rank == 0:

                def _desc(x):
                    if x is None:
                        return 'None'
                    if hasattr(x, 'shape'):
                        return f'{type(x).__name__}{tuple(x.shape)}'
                    return type(x).__name__

                print(
                    f'[step_mask][hook_probe] input_ids={_desc(kwargs.get("input_ids"))} '
                    f'labels={_desc(kwargs.get("labels"))} '
                    f'inputs_embeds={_desc(kwargs.get("inputs_embeds"))} '
                    f'position_ids={_desc(kwargs.get("position_ids"))} '
                    f'kwargs_keys={sorted(kwargs.keys())}',
                    flush=True)
        # Hand off to swift's original hook (it pops input_ids and runs
        # _post_encode to produce inputs_embeds).
        return original_pre_forward_hook(self, model, args, kwargs)

    _tpl_base.Template.pre_forward_hook = wrapped_pre_forward_hook

    _patched_state.update(
        installed=True,
        actor_rmsnorm_count=actor_rmsnorm_count,
        actor_gated_norm_count=actor_gated_norm_count,
        actor_projection_count=actor_projection_count,
        actor_visual_count=actor_visual_count,
        actor_lm_head_count=actor_lm_head_count,
        original_attn_impl=original_attn_impl,
        original_create_causal_mask=original_create_causal_mask,
        original_pre_forward_hook=original_pre_forward_hook,
        text_config=text_config,
    )
    log.info('[step_mask] patched modeling_qwen3_5.create_causal_mask and '
             'swift.template.base.Template.pre_forward_hook')
    if _debug_enabled():
        print(
            f'[step_mask] DEBUG ON. Will print mask diagnostics for the first '
            f'{_DEBUG_MAX_CALLS} forwards (rank 0 only).',
            flush=True)
