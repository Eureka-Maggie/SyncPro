"""Linear-attention state fork + merge for dual-branch training/inference.

Design (see design discussion for full context):

Per frame, the linear-attn scan splits into three sub-segments:

    [shared prefix] -> [B branch] -> [A branch] -> merge -> [shared continuation]
                                fork /       fork \\

* The **shared prefix** (image tokens + assistant role marker + non-thinking
  ``<think>\\n\\n</think>\\n\\n`` prefix) evolves the main recurrent state
  normally.
* At the position where the per-branch source token starts, we **fork**:
  both B and A sub-segments start from the same ``state_pre`` snapshot.
  In dual-source training those source tokens are duplicated copies of the
  shared prefix's last token, so the first ``<inner>/<silent>`` and
  ``<speak>/<silent>`` action tokens are predicted from branch-specific LoRA
  routes.
* B sub-segment scans from state_pre, produces ``state_B_end``.
* A sub-segment scans from state_pre, produces ``state_A_end``. **A never
  sees B during its scan** -- this is what keeps parallel inference decode
  train/infer-consistent.
* At frame end, the two forked states are **merged** into ``state_next``:
    - ``sequential``: state_next = chunk_kernel(A_KV, initial_state=state_B_end).final_state
      i.e. scan A's K/V on top of state_B_end. Mathematically equivalent to
      ``state_pre -> scan(B) -> scan(A)``. Exact and correct.
    - ``additive`` with a shared gate:
      state_next = state_pre + sigmoid(gamma) * (dB + dA)
      where ``dB = state_B_end - state_pre``, ``dA = state_A_end - state_pre``.
      Per-head ``gamma`` initialized from ``MSS_MERGE_GAMMA`` (default 0.5,
      where the merge equals ``(state_B + state_A) / 2``). The gate can stay
      fixed or be trained to learn how much of each branch's delta to keep.
    - ``additive`` with split gates (``MSS_MERGE_GATE_MODE=split``):
      state_next = state_pre + sigmoid(alpha) * dA + sigmoid(beta) * dB.
      ``alpha`` and ``beta`` are independent per-head parameters initialized
      from ``MSS_MERGE_ALPHA`` / ``MSS_MERGE_BETA`` (both default to 0.5).
* Any shared tokens after A (e.g. trailing ``\\n``, next frame's image tokens)
  continue with ``state_next``.

Qwen3.5 always installs the packed-sample isolation wrapper. The B/A
fork path is active only when ``MSS_DUAL_BRANCH=1`` supplies branch spans.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
import torch
import torch.nn as nn
from typing import List, Literal, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

_ENV_MERGE_MODE = 'MSS_MERGE_MODE'  # 'sequential' (default) or 'additive'
_ENV_MERGE_GAMMA = 'MSS_MERGE_GAMMA'
_ENV_MERGE_GATE_MODE = 'MSS_MERGE_GATE_MODE'
_ENV_MERGE_ALPHA = 'MSS_MERGE_ALPHA'
_ENV_MERGE_BETA = 'MSS_MERGE_BETA'
_ENV_ACTOR_LINEAR_MODE = 'MSS_DUAL_ACTOR_LINEAR_MODE'


def get_actor_linear_mode() -> Literal['training_chunk', 'inference_aligned']:
    """Choose the numerical scan used by dual actor replay."""
    raw = os.environ.get(_ENV_ACTOR_LINEAR_MODE, 'training_chunk').strip().lower()
    if raw not in ('training_chunk', 'inference_aligned'):
        raise ValueError(f'{_ENV_ACTOR_LINEAR_MODE}={raw!r} invalid; expected '
                         'training_chunk|inference_aligned')
    return raw  # type: ignore[return-value]


# Module-level dict populated per forward by the pre_forward_hook wrapper,
# read by the linear-attn kernel wrapper. Same rationale as
# ``_dual_lora_ctx`` in dual_lora.py: contextvars do not survive
# ``torch.utils.checkpoint``'s recomputation boundary (recomputed forward
# runs in a fresh Context and sees the default value), so any dependency
# on ctx state creates a divergent code path and triggers CheckpointError.
# A plain module-level dict persists across all Python-level state changes
# within a single training step and is preserved into backward's recompute.
#
# Fields include global B/A spans plus padding-free sample boundaries:
# ``sample_spans`` / ``cu_seqlens`` drive recurrent-state reset, while
# ``seq_idx`` drives causal-conv reset. B/A lists are empty for ordinary
# non-dual padding-free batches.
# Empty dict when no boundary-aware forward is active.
_linear_fork_ctx: dict = {}


def get_merge_mode() -> Literal['sequential', 'additive']:
    raw = os.environ.get(_ENV_MERGE_MODE, 'sequential').strip().lower()
    if raw not in ('sequential', 'additive'):
        raise ValueError(f'{_ENV_MERGE_MODE}={raw!r} invalid; expected sequential|additive')
    return raw  # type: ignore[return-value]


def get_merge_gamma() -> float:
    """Return the additive-merge gate value (or learnable-gate initial value)."""
    return _get_merge_weight(_ENV_MERGE_GAMMA, 0.5)


def get_merge_gate_mode() -> Literal['shared', 'split']:
    """Return whether additive merge shares one gate or learns one per branch."""
    raw = os.environ.get(_ENV_MERGE_GATE_MODE, 'shared').strip().lower()
    if raw not in ('shared', 'split'):
        raise ValueError(f'{_ENV_MERGE_GATE_MODE}={raw!r} invalid; expected shared|split')
    return raw  # type: ignore[return-value]


def _get_merge_weight(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f'{env_name}={raw!r} invalid; expected a float in (0, 1)') from exc
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f'{env_name}={raw!r} invalid; expected a finite float in (0, 1)')
    return value


def get_merge_alpha() -> float:
    return _get_merge_weight(_ENV_MERGE_ALPHA, 0.5)


def get_merge_beta() -> float:
    return _get_merge_weight(_ENV_MERGE_BETA, 0.5)


class GatedAdditiveMerge(nn.Module):
    """Per-head shared or split learnable gates for additive merge.

    Forward:
        delta_A = state_A_end - state_pre
        delta_B = state_B_end - state_pre
        shared: state_pre + sigmoid(gamma_logit) * (delta_A + delta_B)
        split:  state_pre + sigmoid(alpha_logit) * delta_A
                          + sigmoid(beta_logit) * delta_B

    Shared mode preserves the historical ``gamma_logit`` checkpoint layout.
    Split mode uses independent ``alpha_logit`` (A branch) and ``beta_logit``
    (B branch) parameters, initialized to 0.5 by default. All logits stay in
    fp32 and are sigmoid-bounded to (0, 1).

    State shape: [B, num_v_heads, head_k_dim, head_v_dim]. Gates broadcast
    across the head-value / head-key dims.
    """

    def __init__(
        self,
        num_v_heads: int,
        gamma: Optional[float] = None,
        *,
        gate_mode: Optional[str] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ):
        super().__init__()
        self.num_v_heads = num_v_heads
        self.gate_mode = get_merge_gate_mode() if gate_mode is None else str(gate_mode)
        if self.gate_mode == 'shared':
            gamma = get_merge_gamma() if gamma is None else float(gamma)
            self.gamma_logit = nn.Parameter(self._initial_logits(num_v_heads, gamma, 'gamma'))
        elif self.gate_mode == 'split':
            alpha = get_merge_alpha() if alpha is None else float(alpha)
            beta = get_merge_beta() if beta is None else float(beta)
            self.alpha_logit = nn.Parameter(self._initial_logits(num_v_heads, alpha, 'alpha'))
            self.beta_logit = nn.Parameter(self._initial_logits(num_v_heads, beta, 'beta'))
        else:
            raise ValueError(f'gate_mode={self.gate_mode!r} invalid; expected shared|split')

    @staticmethod
    def _initial_logits(num_v_heads: int, value: float, name: str) -> torch.Tensor:
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f'{name} must be a finite float in (0, 1), got {value!r}')
        logit = math.log(value / (1.0 - value))
        return torch.full((num_v_heads, ), logit, dtype=torch.float32)

    def forward(self, state_pre: torch.Tensor, state_B_end: torch.Tensor, state_A_end: torch.Tensor) -> torch.Tensor:
        # state_*: [B, num_v_heads, d_k, d_v]
        delta_A = state_A_end - state_pre
        delta_B = state_B_end - state_pre
        if self.gate_mode == 'shared':
            gamma = torch.sigmoid(self.gamma_logit).to(state_pre.dtype).view(1, -1, 1, 1)
            return state_pre + gamma * (delta_A + delta_B)
        alpha = torch.sigmoid(self.alpha_logit).to(state_pre.dtype).view(1, -1, 1, 1)
        beta = torch.sigmoid(self.beta_logit).to(state_pre.dtype).view(1, -1, 1, 1)
        return state_pre + alpha * delta_A + beta * delta_B


class _ProductionRecurrentForward(torch.autograd.Function):
    """Use production recurrent numerics with a differentiable replay."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        forward_kernel,
        backward_kernel,
    ):
        ctx.set_materialize_grads(False)
        ctx.has_initial_state = initial_state is not None
        saved_initial = (q.new_empty(0) if initial_state is None else initial_state)
        ctx.save_for_backward(q, k, v, g, beta, saved_initial)
        ctx.backward_kernel = backward_kernel
        output, final_state = forward_kernel(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        if final_state is None:
            raise RuntimeError('production recurrent kernel returned no final state')
        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        q, k, v, g, beta, saved_initial = ctx.saved_tensors
        originals = [q, k, v, g, beta]
        if ctx.has_initial_state:
            originals.append(saved_initial)
        with torch.enable_grad():
            replay = [value.detach().requires_grad_(True) for value in originals]
            q_ref, k_ref, v_ref, g_ref, beta_ref = replay[:5]
            initial_ref = replay[5] if ctx.has_initial_state else None
            output_ref, final_ref = ctx.backward_kernel(
                q_ref,
                k_ref,
                v_ref,
                g=g_ref,
                beta=beta_ref,
                initial_state=initial_ref,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if grad_output is None:
                grad_output = torch.zeros_like(output_ref)
            if grad_final_state is None:
                grad_final_state = torch.zeros_like(final_ref)
            gradients = torch.autograd.grad(
                (output_ref, final_ref),
                replay,
                (grad_output, grad_final_state),
                allow_unused=False,
            )
        grad_initial = gradients[5] if ctx.has_initial_state else None
        return (*gradients[:5], grad_initial, None, None)


def _production_recurrent_with_torch_backward(
    q,
    k,
    v,
    *,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel,
    forward_kernel,
    backward_kernel,
):
    """Exact production forward; equivalent PyTorch backward replay.

    FLA's fused recurrent kernel is the production Q=1 decode path but has
    no backward. Its whole-segment forward is elementwise identical to
    repeated Q=1 calls. The Transformers PyTorch recurrence is used only
    to differentiate the same equations during this custom backward.
    """
    if not output_final_state:
        raise ValueError('inference-aligned recurrent replay requires final state')
    if not use_qk_l2norm_in_kernel:
        raise ValueError('inference-aligned recurrent replay requires Q/K L2 norm')
    return _ProductionRecurrentForward.apply(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        forward_kernel,
        backward_kernel,
    )


def _segment_kernel_call(chunk_kernel, Q, K, V, g, beta, seg_start, seg_end, initial_state):
    """Slice Q/K/V/g/beta along the sequence axis and call the chunk kernel.

    Returns (per_token_output, final_state).

    Q, K, V shape: [B, L, num_v_heads, head_dim]
    g, beta shape: [B, L, num_v_heads]
    """
    q_seg = Q[:, seg_start:seg_end]
    k_seg = K[:, seg_start:seg_end]
    v_seg = V[:, seg_start:seg_end]
    g_seg = g[:, seg_start:seg_end]
    b_seg = beta[:, seg_start:seg_end]

    if q_seg.shape[1] == 0:
        # Empty segment: no state update, no output. Return initial state
        # and an empty output tensor.
        empty_out = q_seg.new_empty((q_seg.shape[0], 0, q_seg.shape[2], V.shape[-1]))
        return empty_out, initial_state

    out, final_state = chunk_kernel(
        q_seg,
        k_seg,
        v_seg,
        g=g_seg,
        beta=b_seg,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    return out, final_state


def _sample_spans_from_cu_seqlens(
    cu_seqlens: Optional[Sequence[int] | torch.Tensor],
    seq_len: int,
) -> List[Tuple[int, int]]:
    """Convert packed-sequence boundaries into validated Python spans."""
    if cu_seqlens is None:
        return [(0, seq_len)]
    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.dim() != 1:
            raise ValueError(f'cu_seqlens must be 1D, got {tuple(cu_seqlens.shape)}')
        boundaries = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    else:
        boundaries = [int(x) for x in cu_seqlens]
    if len(boundaries) < 2:
        raise ValueError(f'cu_seqlens must contain at least [0, L], got {boundaries}')
    if boundaries[0] != 0 or boundaries[-1] != seq_len:
        raise ValueError(f'cu_seqlens must start at 0 and end at seq_len={seq_len}, got {boundaries}')
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(f'cu_seqlens must be strictly increasing, got {boundaries}')
    return list(zip(boundaries[:-1], boundaries[1:]))


def _seq_idx_from_sample_spans(
    sample_spans: Sequence[Tuple[int, int]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build causal-conv sequence IDs for a padding-free packed row."""
    if len(sample_spans) <= 1:
        return None
    chunks = [
        torch.full((end - start, ), idx, dtype=torch.int32, device=device)
        for idx, (start, end) in enumerate(sample_spans)
    ]
    return torch.cat(chunks, dim=0).unsqueeze(0)


def _segmented_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    sample_spans: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Autograd-safe fallback that resets depthwise conv at every sample."""
    if activation not in (None, 'silu', 'swish'):
        raise NotImplementedError(f'unsupported causal-conv activation: {activation}')
    if x.shape[0] != 1 and len(sample_spans) > 1:
        raise ValueError(f'packed segmented causal conv expects batch size 1, got {x.shape[0]}')
    dtype_in = x.dtype
    dim, width = weight.shape
    outputs = []
    for start, end in sample_spans:
        x_seg = x[:, :, start:end].to(weight.dtype)
        out = torch.nn.functional.conv1d(x_seg, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
        out = out[:, :, :end - start]
        if activation in ('silu', 'swish'):
            out = torch.nn.functional.silu(out)
        outputs.append(out.to(dtype=dtype_in))
    return torch.cat(outputs, dim=-1)


def _correct_branch_causal_conv(
    output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    b_spans: Sequence[Tuple[int, int]],
    a_spans: Sequence[Tuple[int, int]],
    sample_spans: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Replace A's contaminated causal-conv windows with prefix-forked ones.

    The regular convolution is correct everywhere except the first
    ``kernel_width - 1`` positions on A's path: in physical storage those
    windows include B, while logically A starts from the state immediately
    before B. Recomputing only those short windows keeps the fast convolution
    for the full sequence and removes both forward and backward B-to-A flow.
    """
    if not b_spans or weight.shape[1] <= 1:
        return output
    if len(b_spans) != len(a_spans):
        raise ValueError('B/A span count mismatch during causal-conv correction')
    if x.shape[0] != 1:
        raise ValueError(f'dual branch causal conv expects batch size 1, got {x.shape[0]}')
    if activation not in (None, 'silu', 'swish'):
        raise NotImplementedError(f'unsupported causal-conv activation: {activation}')

    width = weight.shape[1]
    target_positions = []
    window_positions = []
    window_valid = []
    sample_idx = 0

    for branch_idx, ((b_start, b_end), (a_start, a_end)) in enumerate(zip(b_spans, a_spans)):
        if b_end != a_start:
            raise ValueError(f'non-contiguous branch pair #{branch_idx}: B={(b_start, b_end)} '
                             f'A={(a_start, a_end)}')
        while sample_idx < len(sample_spans) and b_start >= sample_spans[sample_idx][1]:
            sample_idx += 1
        if sample_idx == len(sample_spans):
            raise ValueError(f'branch pair #{branch_idx} lies outside sample spans')
        sample_start, sample_end = sample_spans[sample_idx]
        if b_start < sample_start or a_end > sample_end:
            raise ValueError(f'branch pair #{branch_idx} crosses a sample boundary')

        next_branch_start = (b_spans[branch_idx + 1][0] if branch_idx + 1 < len(b_spans) else sample_end)
        correction_end = min(a_start + width - 1, sample_end, next_branch_start)
        for target in range(a_start, correction_end):
            relative_target = target - a_start
            indices = []
            valid = []
            for weight_idx in range(width):
                relative_source = relative_target - (width - 1 - weight_idx)
                source = (a_start + relative_source if relative_source >= 0 else b_start + relative_source)
                is_valid = source >= sample_start
                indices.append(source if is_valid else sample_start)
                valid.append(is_valid)
            target_positions.append(target)
            window_positions.append(indices)
            window_valid.append(valid)

    if not target_positions:
        return output

    target_idx = torch.tensor(target_positions, dtype=torch.long, device=x.device)
    window_idx = torch.tensor(window_positions, dtype=torch.long, device=x.device)
    valid_mask = torch.tensor(window_valid, dtype=torch.bool, device=x.device)
    batch_size, channels, _ = x.shape
    windows = x.to(weight.dtype).index_select(2, window_idx.flatten())
    windows = windows.view(batch_size, channels, len(target_positions), width)
    windows = windows * valid_mask.view(1, 1, len(target_positions), width)
    corrected = (windows * weight.view(1, channels, 1, width)).sum(dim=-1)
    if bias is not None:
        corrected = corrected + bias.view(1, channels, 1)
    if activation in ('silu', 'swish'):
        corrected = torch.nn.functional.silu(corrected)
    return output.index_copy(2, target_idx, corrected.to(dtype=output.dtype))


class _CausalConvBackwardLayout(torch.autograd.Function):
    """Keep production causal-conv backward on its supported memory layout."""

    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None
        # A sliced/reordered actor output can arrive with stride(1) == 1 but
        # stride(0)/stride(2) not divisible by 8. causal_conv1d's Python
        # wrapper then mistakes it for a valid channel-last tensor and the
        # CUDA kernel rejects it. Materialize the canonical [B, L, C]
        # storage and expose it again as logical [B, C, L].  Do not rely on
        # transpose(...).contiguous().transpose(...): for L == 1 PyTorch is
        # free to collapse the singleton stride back to 1, which still fails
        # causal_conv1d's channel-last CUDA backward check.  Explicit padded
        # strides preserve the required 8-element alignment for every L.
        batch, channels, length = grad_output.shape
        channel_stride = ((int(channels) + 7) // 8) * 8
        aligned = torch.empty_strided(
            (batch, channels, length),
            (max(int(length), 1) * channel_stride, 1, channel_stride),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        aligned.copy_(grad_output)
        return aligned


def _causal_conv_backward_layout(tensor: torch.Tensor) -> torch.Tensor:
    return _CausalConvBackwardLayout.apply(tensor)


def _causal_conv_segment(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    start: int,
    end: int,
    state: Optional[torch.Tensor],
    *,
    conv_kernel=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autograd-safe equivalent of Qwen's cached causal-conv update."""
    if activation not in (None, 'silu', 'swish'):
        raise NotImplementedError(f'unsupported causal-conv activation: {activation}')
    width = int(weight.shape[1])
    state_width = width - 1
    if state_width < 1:
        raise ValueError(f'causal-conv width must be >=2, got {width}')
    if state is None:
        # The CUDA kernel requires channel-last physical layout even though
        # the logical state shape remains [batch, channels, width - 1].
        state = x.new_zeros((x.shape[0], state_width, x.shape[1])).transpose(1, 2)
    if end <= start:
        return x[:, :, :0], state

    segment = x[:, :, start:end]
    if conv_kernel is not None:
        output, next_state = conv_kernel(
            x=segment,
            weight=weight,
            bias=bias,
            seq_idx=None,
            initial_states=state,
            return_final_states=True,
            final_states_out=None,
            activation=activation,
        )
        return (
            _causal_conv_backward_layout(output),
            _causal_conv_backward_layout(next_state),
        )
    extended = torch.cat((state, segment), dim=-1).to(weight.dtype)
    output = torch.nn.functional.conv1d(
        extended,
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=int(weight.shape[0]),
    )
    if activation in ('silu', 'swish'):
        output = torch.nn.functional.silu(output)
    next_state = extended[:, :, -state_width:].to(dtype=x.dtype)
    return output.to(dtype=x.dtype), next_state


def _inference_aligned_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    *,
    frame_starts: Sequence[int],
    b_spans: Sequence[Tuple[int, int]],
    a_spans: Sequence[Tuple[int, int]],
    sample_spans: Sequence[Tuple[int, int]],
    conv_kernel=None,
) -> torch.Tensor:
    """Replay causal-conv with the same A/B fork and frame boundaries.

    Production keeps independent convolution states for A and B and adopts
    A's tail after merging the recurrent state. Later trailer/frame tokens
    therefore see A, never the physically preceding B. This functional path
    retains that state graph for RL backward without calling the inference-
    only causal_conv1d_update kernel.
    """
    if x.shape[0] != 1:
        raise ValueError(f'inference-aligned causal conv expects batch size 1, got '
                         f'{x.shape[0]}')
    if len(b_spans) != len(a_spans):
        raise ValueError('B/A span count mismatch during causal-conv replay')

    seq_len = int(x.shape[-1])
    frame_starts = sorted(int(value) for value in frame_starts)
    b_spans = [(int(start), int(end)) for start, end in b_spans]
    a_spans = [(int(start), int(end)) for start, end in a_spans]
    output = torch.empty_like(x)
    branch_idx = 0

    for sample_start, sample_end in sample_spans:
        sample_start, sample_end = int(sample_start), int(sample_end)
        if not (0 <= sample_start < sample_end <= seq_len):
            raise ValueError(f'invalid causal-conv sample span '
                             f'{(sample_start, sample_end)}')
        state = None
        cursor = sample_start
        sample_branch_count = 0

        while branch_idx < len(b_spans) and b_spans[branch_idx][0] < sample_end:
            b_start, b_end = b_spans[branch_idx]
            a_start, a_end = a_spans[branch_idx]
            if not (sample_start <= b_start < b_end == a_start < a_end <= sample_end):
                raise ValueError(f'invalid causal-conv branch pair #{branch_idx}: '
                                 f'B={b_spans[branch_idx]} A={a_spans[branch_idx]} '
                                 f'sample={(sample_start, sample_end)}')
            if b_start < cursor:
                raise ValueError(f'causal-conv branch #{branch_idx} starts before '
                                 f'cursor={cursor}')

            if sample_branch_count > 0:
                next_frame_starts = [value for value in frame_starts if cursor < value < b_start]
                if len(next_frame_starts) != 1:
                    raise RuntimeError('inference-aligned causal conv expected exactly one '
                                       f'frame boundary in [{cursor}, {b_start}), got '
                                       f'{next_frame_starts}')
                next_frame_start = next_frame_starts[0]
                trailer_out, state = _causal_conv_segment(
                    x,
                    weight,
                    bias,
                    activation,
                    cursor,
                    next_frame_start,
                    state,
                    conv_kernel=conv_kernel,
                )
                output[:, :, cursor:next_frame_start] = trailer_out
                prefill_out, state = _causal_conv_segment(
                    x,
                    weight,
                    bias,
                    activation,
                    next_frame_start,
                    b_start,
                    state,
                    conv_kernel=conv_kernel,
                )
                output[:, :, next_frame_start:b_start] = prefill_out
            else:
                shared_out, state = _causal_conv_segment(
                    x,
                    weight,
                    bias,
                    activation,
                    cursor,
                    b_start,
                    state,
                    conv_kernel=conv_kernel,
                )
                output[:, :, cursor:b_start] = shared_out

            state_pre = state
            b_out, _ = _causal_conv_segment(
                x,
                weight,
                bias,
                activation,
                b_start,
                b_end,
                state_pre,
                conv_kernel=conv_kernel,
            )
            output[:, :, b_start:b_end] = b_out
            a_out, state = _causal_conv_segment(
                x,
                weight,
                bias,
                activation,
                a_start,
                a_end,
                state_pre,
                conv_kernel=conv_kernel,
            )
            output[:, :, a_start:a_end] = a_out
            cursor = a_end
            branch_idx += 1
            sample_branch_count += 1

        trailing_out, state = _causal_conv_segment(
            x,
            weight,
            bias,
            activation,
            cursor,
            sample_end,
            state,
            conv_kernel=conv_kernel,
        )
        output[:, :, cursor:sample_end] = trailing_out

    if branch_idx != len(b_spans):
        raise ValueError(f'{len(b_spans) - branch_idx} causal-conv branches lie outside '
                         'sample spans')
    return output


def fork_and_merge_scan(
    chunk_kernel,
    Q,
    K,
    V,
    g,
    beta,
    *,
    frame_starts: List[int],
    b_spans: List[Tuple[int, int]],
    a_spans: List[Tuple[int, int]],
    merge_module: Optional[GatedAdditiveMerge] = None,
    merge_mode: str = 'sequential',
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[Sequence[int] | torch.Tensor] = None,
    sample_spans: Optional[Sequence[Tuple[int, int]]] = None,
    recurrent_kernel=None,
    actor_linear_mode: str = 'training_chunk',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run fork+merge while resetting recurrent state at packed samples.

    Q/K/V are padding-free tensors with shape ``[1, total_length, ...]``.
    ``sample_spans`` (or ``cu_seqlens``) identifies the original logical
    samples. Each sample starts from an independent recurrent state, and a
    B/A fork is rejected if it crosses a logical sample boundary.

    ``inference_aligned`` preserves production numerical segmentation:
    frame prefill uses the chunk kernel, while branch tokens and the forced
    trailer use the recurrent kernel used by Q=1 decode.
    """
    if actor_linear_mode not in ('training_chunk', 'inference_aligned'):
        raise ValueError(f'unknown actor_linear_mode: {actor_linear_mode}')
    if actor_linear_mode == 'inference_aligned' and recurrent_kernel is None:
        raise ValueError('inference_aligned actor scan requires recurrent_kernel')
    frame_starts = sorted(int(value) for value in frame_starts)

    batch_size, seq_len, num_v_heads, head_v_dim = V.shape
    if batch_size != 1:
        raise ValueError(f'fork_and_merge_scan expects padding-free batch size 1, got {batch_size}')
    if not (Q.shape[1] == K.shape[1] == beta.shape[1] == g.shape[1] == seq_len):
        raise ValueError('Q/K/V/g/beta sequence lengths do not match')

    if sample_spans is None:
        sample_spans = _sample_spans_from_cu_seqlens(cu_seqlens, seq_len)
    else:
        sample_spans = [(int(start), int(end)) for start, end in sample_spans]
        if not sample_spans:
            raise ValueError('sample_spans must not be empty')
        if sample_spans[0][0] != 0 or sample_spans[-1][1] != seq_len:
            raise ValueError(f'sample_spans must cover [0, {seq_len}), got {sample_spans}')
        if any(start >= end for start, end in sample_spans):
            raise ValueError(f'sample_spans must be non-empty, got {sample_spans}')
        if any(left[1] != right[0] for left, right in zip(sample_spans, sample_spans[1:])):
            raise ValueError(f'sample_spans must be contiguous, got {sample_spans}')

    if len(b_spans) != len(a_spans):
        raise ValueError(f'len mismatch: b_spans={len(b_spans)}, a_spans={len(a_spans)}')
    b_spans = [(int(start), int(end)) for start, end in b_spans]
    a_spans = [(int(start), int(end)) for start, end in a_spans]
    previous_a_end = 0
    for branch_idx, ((b_start, b_end), (a_start, a_end)) in enumerate(zip(b_spans, a_spans)):
        if not (0 <= b_start < b_end == a_start < a_end <= seq_len):
            raise ValueError(f'invalid branch pair #{branch_idx}: B={b_spans[branch_idx]} '
                             f'A={a_spans[branch_idx]} seq_len={seq_len}')
        if b_start < previous_a_end:
            raise ValueError(f'branch spans overlap or are out of order at #{branch_idx}')
        previous_a_end = a_end

    num_samples = len(sample_spans)
    if initial_state is not None and initial_state.shape[0] != num_samples:
        raise ValueError(f'initial_state has {initial_state.shape[0]} states, expected {num_samples}')

    core_out = V.new_empty((batch_size, seq_len, num_v_heads, head_v_dim))
    final_states = []
    branch_idx = 0

    for sample_idx, (sample_start, sample_end) in enumerate(sample_spans):
        cur_state = (None if initial_state is None else initial_state[sample_idx:sample_idx + 1])
        cursor = sample_start
        sample_branch_count = 0

        while branch_idx < len(b_spans) and b_spans[branch_idx][0] < sample_end:
            b_start, b_end = b_spans[branch_idx]
            a_start, a_end = a_spans[branch_idx]
            if b_start < sample_start or a_end > sample_end:
                raise ValueError(f'branch pair #{branch_idx} crosses packed-sample boundary '
                                 f'[{sample_start}, {sample_end}): B={b_spans[branch_idx]} '
                                 f'A={a_spans[branch_idx]}')
            if b_start < cursor:
                raise ValueError(f'branch pair #{branch_idx} starts before scan cursor {cursor}')

            if (actor_linear_mode == 'inference_aligned' and sample_branch_count > 0):
                next_frame_starts = [value for value in frame_starts if cursor < value < b_start]
                if len(next_frame_starts) != 1:
                    raise RuntimeError('inference-aligned actor expected exactly one frame '
                                       f'boundary in shared span [{cursor}, {b_start}), got '
                                       f'{next_frame_starts}')
                next_frame_start = next_frame_starts[0]
                trailer_out, cur_state = _segment_kernel_call(
                    recurrent_kernel,
                    Q,
                    K,
                    V,
                    g,
                    beta,
                    cursor,
                    next_frame_start,
                    cur_state,
                )
                if trailer_out.numel() > 0:
                    core_out[:, cursor:next_frame_start] = trailer_out
                prefill_out, cur_state = _segment_kernel_call(
                    chunk_kernel,
                    Q,
                    K,
                    V,
                    g,
                    beta,
                    next_frame_start,
                    b_start,
                    cur_state,
                )
                if prefill_out.numel() > 0:
                    core_out[:, next_frame_start:b_start] = prefill_out
            else:
                shared_out, cur_state = _segment_kernel_call(
                    chunk_kernel,
                    Q,
                    K,
                    V,
                    g,
                    beta,
                    cursor,
                    b_start,
                    cur_state,
                )
                if shared_out.numel() > 0:
                    core_out[:, cursor:b_start] = shared_out
            state_pre = cur_state

            branch_kernel = (recurrent_kernel if actor_linear_mode == 'inference_aligned' else chunk_kernel)
            b_out, state_B_end = _segment_kernel_call(
                branch_kernel,
                Q,
                K,
                V,
                g,
                beta,
                b_start,
                b_end,
                state_pre,
            )
            core_out[:, b_start:b_end] = b_out

            a_out, state_A_end = _segment_kernel_call(
                branch_kernel,
                Q,
                K,
                V,
                g,
                beta,
                a_start,
                a_end,
                state_pre,
            )
            core_out[:, a_start:a_end] = a_out

            if merge_mode == 'sequential':
                _, cur_state = _segment_kernel_call(
                    branch_kernel,
                    Q,
                    K,
                    V,
                    g,
                    beta,
                    a_start,
                    a_end,
                    state_B_end,
                )
            elif merge_mode == 'additive':
                if merge_module is None:
                    raise ValueError('additive merge requires GatedAdditiveMerge')
                cur_state = merge_module(state_pre, state_B_end, state_A_end)
            else:
                raise ValueError(f'unknown merge_mode: {merge_mode}')

            cursor = a_end
            branch_idx += 1
            sample_branch_count += 1

        trailing_kernel = (
            recurrent_kernel if actor_linear_mode == 'inference_aligned' and sample_branch_count > 0 else chunk_kernel)
        trailing_out, cur_state = _segment_kernel_call(
            trailing_kernel,
            Q,
            K,
            V,
            g,
            beta,
            cursor,
            sample_end,
            cur_state,
        )
        if trailing_out.numel() > 0:
            core_out[:, cursor:sample_end] = trailing_out
        if cur_state is None:
            raise RuntimeError(f'no recurrent state produced for sample #{sample_idx}')
        final_states.append(cur_state)

    if branch_idx != len(b_spans):
        raise ValueError(f'{len(b_spans) - branch_idx} branch pairs lie outside sample spans')

    return core_out, torch.cat(final_states, dim=0)


# ---------------------------------------------------------------------------
# Setter used by the pre_forward_hook: stash segment info for the scan
# wrapper to consume during the model's forward.


def set_linear_fork_ctx(
    frame_starts,
    b_spans,
    a_spans,
    *,
    cu_seqlens: Optional[Sequence[int] | torch.Tensor] = None,
    seq_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Set branch metadata and padding-free sample boundaries for a forward."""
    if seq_len is None:
        if cu_seqlens is not None:
            if isinstance(cu_seqlens, torch.Tensor):
                seq_len = int(cu_seqlens[-1].detach().cpu().item())
            else:
                seq_len = int(list(cu_seqlens)[-1])
        elif a_spans:
            seq_len = int(a_spans[-1][1])
        else:
            raise ValueError('seq_len is required when no cu_seqlens or branch spans exist')

    sample_spans = _sample_spans_from_cu_seqlens(cu_seqlens, seq_len)
    if device is None:
        if isinstance(cu_seqlens, torch.Tensor):
            device = cu_seqlens.device
        else:
            device = torch.device('cpu')
    device = torch.device(device)
    boundaries = [start for start, _ in sample_spans] + [sample_spans[-1][1]]

    # In-place update is required for gradient-checkpoint recomputation.
    _linear_fork_ctx.clear()
    _linear_fork_ctx['frame_starts'] = [int(x) for x in frame_starts]
    _linear_fork_ctx['b_spans'] = [tuple(map(int, x)) for x in b_spans]
    _linear_fork_ctx['a_spans'] = [tuple(map(int, x)) for x in a_spans]
    _linear_fork_ctx['sample_spans'] = sample_spans
    _linear_fork_ctx['cu_seqlens'] = torch.tensor(boundaries, dtype=torch.int32, device=device)
    _linear_fork_ctx['seq_idx'] = _seq_idx_from_sample_spans(sample_spans, device)


def clear_linear_fork_ctx() -> None:
    _linear_fork_ctx.clear()


def get_linear_fork_ctx() -> Optional[dict]:
    return _linear_fork_ctx if _linear_fork_ctx else None


# ---------------------------------------------------------------------------
# Injection: replace each Qwen3_5GatedDeltaNet's ``chunk_gated_delta_rule``
# attribute with a wrapper that either
#   (a) falls through to the original kernel when no fork ctx is set, OR
#   (b) runs fork_and_merge_scan when fork ctx is populated.

_linear_fork_patched: dict = {'installed': False}
_padding_free_hook_patched: dict = {'installed': False}


def patch_pre_forward_hook_for_padding_free_linear_attn() -> None:
    """Populate sample boundaries for ordinary Qwen3.5 padding-free batches."""
    if _padding_free_hook_patched.get('installed'):
        return

    from swift.template import base as _tpl_base

    original_hook = _tpl_base.Template.pre_forward_hook

    def wrapped(self, model, args, kwargs):
        input_ids = kwargs.get('input_ids')
        inputs_embeds = kwargs.get('inputs_embeds')
        cu_seqlens = kwargs.get('cu_seq_lens_q')
        source = input_ids if input_ids is not None else inputs_embeds
        if source is not None and cu_seqlens is not None:
            set_linear_fork_ctx(
                frame_starts=[],
                b_spans=[],
                a_spans=[],
                cu_seqlens=cu_seqlens,
                seq_len=source.shape[-1] if input_ids is not None else source.shape[-2],
                device=source.device,
            )
        else:
            clear_linear_fork_ctx()
        return original_hook(self, model, args, kwargs)

    _tpl_base.Template.pre_forward_hook = wrapped
    _padding_free_hook_patched.update(installed=True, original_hook=original_hook)
    log.info('[linear_fork] wrapped Template.pre_forward_hook for padding-free boundaries')


def inject_dual_branch_linear_attn(model, merge_mode: str = 'sequential') -> int:
    """Make Qwen3.5 linear attention sample-aware and optionally fork B/A.

    The causal convolution receives ``seq_idx`` and the recurrent kernel
    receives either ``cu_seqlens`` or the explicit dual-branch scan. This
    keeps every logical padding-free sample independent even though Swift
    flattens a batch to physical shape ``[1, sum(lengths), ...]``.
    """
    if _linear_fork_patched.get('installed'):
        return 0

    merge_gate_mode = get_merge_gate_mode() if merge_mode == 'additive' else None
    n_patched = 0
    for _, module in model.named_modules():
        if type(module).__name__ != 'Qwen3_5GatedDeltaNet':
            continue

        num_v_heads = module.num_v_heads
        if merge_mode == 'additive':
            module.add_module('dual_merge', GatedAdditiveMerge(num_v_heads, gate_mode=merge_gate_mode))
            base_p = next(module.parameters())
            module.dual_merge.to(device=base_p.device)
        else:
            module.add_module('dual_merge', nn.Identity())

        original_kernel = module.chunk_gated_delta_rule
        original_causal_conv = module.causal_conv1d_fn
        kernel_supports_cu = ('cu_seqlens' in inspect.signature(original_kernel).parameters)

        def make_kernel_wrapper(mod, orig_kernel, supports_cu):

            def wrapped_kernel(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
                *args,
                **kwargs,
            ):
                ctx = get_linear_fork_ctx()
                if ctx is None:
                    return orig_kernel(
                        q,
                        k,
                        v,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        *args,
                        **kwargs,
                    )

                sample_spans = ctx['sample_spans']
                has_branches = bool(ctx['b_spans'])
                is_varlen = len(sample_spans) > 1
                if not has_branches and not is_varlen:
                    return orig_kernel(
                        q,
                        k,
                        v,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        *args,
                        **kwargs,
                    )

                if not has_branches and supports_cu:
                    kernel_kwargs = dict(kwargs)
                    kernel_kwargs['cu_seqlens'] = ctx['cu_seqlens'].to(device=q.device)
                    return orig_kernel(
                        q,
                        k,
                        v,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        *args,
                        **kernel_kwargs,
                    )

                merge_mod = mod.dual_merge if merge_mode == 'additive' else None
                actor_linear_mode = get_actor_linear_mode()
                recurrent_kernel = None
                if actor_linear_mode == 'inference_aligned':
                    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

                    def recurrent_kernel(
                        q_seg,
                        k_seg,
                        v_seg,
                        *,
                        g,
                        beta,
                        initial_state,
                        output_final_state,
                        use_qk_l2norm_in_kernel,
                    ):
                        return _production_recurrent_with_torch_backward(
                            q_seg,
                            k_seg,
                            v_seg,
                            g=g,
                            beta=beta,
                            initial_state=initial_state,
                            output_final_state=output_final_state,
                            use_qk_l2norm_in_kernel=(use_qk_l2norm_in_kernel),
                            forward_kernel=mod.recurrent_gated_delta_rule,
                            backward_kernel=torch_recurrent_gated_delta_rule,
                        )

                return fork_and_merge_scan(
                    orig_kernel,
                    q,
                    k,
                    v,
                    g,
                    beta,
                    frame_starts=ctx['frame_starts'],
                    b_spans=ctx['b_spans'],
                    a_spans=ctx['a_spans'],
                    merge_module=merge_mod,
                    merge_mode=merge_mode,
                    initial_state=initial_state,
                    sample_spans=sample_spans,
                    recurrent_kernel=recurrent_kernel,
                    actor_linear_mode=actor_linear_mode,
                )

            return wrapped_kernel

        def make_causal_conv_wrapper(orig_causal_conv):

            def wrapped_causal_conv(
                x,
                weight,
                bias=None,
                seq_idx=None,
                initial_states=None,
                return_final_states=False,
                final_states_out=None,
                activation=None,
            ):
                ctx = get_linear_fork_ctx()
                if (ctx is not None and seq_idx is None and ctx.get('seq_idx') is not None):
                    seq_idx = ctx['seq_idx'].to(device=x.device)

                if (ctx is not None and ctx.get('b_spans') and get_actor_linear_mode() == 'inference_aligned'):
                    if (initial_states is not None or return_final_states or final_states_out is not None):
                        raise NotImplementedError('inference-aligned actor causal conv expects a '
                                                  'stateless full-sequence training call')
                    return _inference_aligned_causal_conv1d(
                        x,
                        weight,
                        bias,
                        activation,
                        frame_starts=ctx['frame_starts'],
                        b_spans=ctx['b_spans'],
                        a_spans=ctx['a_spans'],
                        sample_spans=ctx['sample_spans'],
                        conv_kernel=orig_causal_conv,
                    )
                if orig_causal_conv is not None:
                    result = orig_causal_conv(
                        x=x,
                        weight=weight,
                        bias=bias,
                        seq_idx=seq_idx,
                        initial_states=initial_states,
                        return_final_states=return_final_states,
                        final_states_out=final_states_out,
                        activation=activation,
                    )
                else:
                    if initial_states is not None or return_final_states:
                        raise NotImplementedError('torch causal-conv fallback does not support cached states')
                    sample_spans = (ctx['sample_spans'] if ctx is not None else [(0, x.shape[-1])])
                    result = _segmented_causal_conv1d(x, weight, bias, activation, sample_spans)

                if ctx is None or not ctx.get('b_spans'):
                    return result
                if return_final_states:
                    output, final_states = result
                else:
                    output, final_states = result, None
                output = _correct_branch_causal_conv(
                    output,
                    x,
                    weight,
                    bias,
                    activation,
                    ctx['b_spans'],
                    ctx['a_spans'],
                    ctx['sample_spans'],
                )
                return (output, final_states) if return_final_states else output

            return wrapped_causal_conv

        module.chunk_gated_delta_rule = make_kernel_wrapper(module, original_kernel, kernel_supports_cu)
        module.causal_conv1d_fn = make_causal_conv_wrapper(original_causal_conv)
        n_patched += 1

    _linear_fork_patched.update(
        installed=True,
        n_patched=n_patched,
        merge_mode=merge_mode,
        merge_gate_mode=merge_gate_mode,
    )
    log.info(f'[linear_fork] patched recurrent scan and causal conv on {n_patched} '
             f'Qwen3_5GatedDeltaNet layers (merge_mode={merge_mode}, '
             f'merge_gate_mode={merge_gate_mode})')
    return n_patched


def enable_padding_free_linear_attn(model) -> int:
    """Enable sample isolation for non-dual Qwen3.5 padding-free training."""
    patch_pre_forward_hook_for_padding_free_linear_attn()
    return inject_dual_branch_linear_attn(model, merge_mode='sequential')
