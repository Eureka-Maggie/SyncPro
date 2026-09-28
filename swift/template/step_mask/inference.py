"""Streaming-inference helper for the step+window mask.

At training time, the patched ``create_causal_mask`` reads input_ids from a
contextvar populated by swift's ``pre_forward_hook``. That hook fires once per
forward with the full prompt, which is sufficient for SFT.

At inference time we need finer control: each ``model.forward`` during
``generate()`` only sees the next 1-N tokens (with growing KV cache), and our
mask predicate needs the FULL cumulative sequence to compute is_visual /
frame_id / step_close. We solve this by maintaining a
``StreamingStepMaskState`` that holds the cumulative input_ids; the patched
``create_causal_mask`` recognises a streaming-mode contextvar and dispatches
through :func:`build_streaming_block_mask`.

This file also exposes :class:`StreamingMaskedRunner`, an inference driver
that:

* Runs a per-chunk loop over video frames matching the training format
  (``Query: <q><image>`` for frame 0, ``<image>`` thereafter, then a `<silent>`
  / ``<inner>...</inner>`` / ``<speak>...</speak>`` assistant turn).
* Reuses the KV cache across frames (only the new turn's tokens are prefilled).
* Optionally evicts the KV cache physically at step boundaries (window-bounded
  KV memory, like ThinkStream's decode-and-prune).
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from dataclasses import dataclass, field
from torch.nn.attention.flex_attention import create_block_mask
from typing import Optional, Sequence, Tuple

from .meta import SPECIAL_TOKEN_IDS, build_step_mask_meta
from .patch import _ctx

log = logging.getLogger(__name__)

_VIS_START = SPECIAL_TOKEN_IDS['vision_start']
_VIS_END = SPECIAL_TOKEN_IDS['vision_end']
_IM_END = SPECIAL_TOKEN_IDS['im_end']
_INNER_CLOSE = SPECIAL_TOKEN_IDS['inner_close']
_SPEAK_CLOSE = SPECIAL_TOKEN_IDS['speak_close']


@dataclass
class StreamingStepMaskState:
    """Cumulative state for one streaming-inference request.

    Attributes
    ----------
    window_frames     : W (must be > 0; 0 disables the patch entirely).
    assistant_role_id : single-token id for the literal "assistant" (74455
                        for the Qwen3.5 tokenizer).
    input_ids_np      : np.int64 array of every token in the request so far
                        (prompt + decoded). Always equals the model's KV cache
                        length when physical eviction is off.
    n_evicted         : number of leading positions that have been physically
                        evicted from the KV cache. Position p in the original
                        sequence sits at slot ``p - n_evicted`` in the cache.
                        Always 0 until ``evict_to(...)`` is called.
    """

    window_frames: int
    assistant_role_id: int
    input_ids_np: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    n_evicted: int = 0

    # Inference-only marker for an independently decoded A/B branch. A
    # branch-local history contains only one half of the current physical
    # dual turn, so ``build_step_mask_meta`` would otherwise classify it as a
    # single-branch turn and emit a premature step_close. While this marker
    # is set, closes at/after the branch source are suppressed. The main
    # (post-merge) state never carries the marker and therefore derives the
    # real close from the complete ``B ... im_end \n A ... im_end`` layout.
    pending_dual_turn_start: Optional[int] = None

    # Internal: cached derived metadata, invalidated on each extend().
    _meta_cache_len: int = 0
    _is_visual: Optional[np.ndarray] = None
    _frame_id: Optional[np.ndarray] = None
    _step_close: Optional[np.ndarray] = None
    _seg_id: Optional[np.ndarray] = None
    _cur_frame: Optional[np.ndarray] = None

    @property
    def total_len(self) -> int:
        return int(self.input_ids_np.shape[0])

    @property
    def cache_len(self) -> int:
        return self.total_len - self.n_evicted

    def extend(self, new_token_ids) -> int:
        """Append new token IDs (1-D iterable / tensor / array). Returns the
        original starting position of the appended block."""
        if torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.detach().cpu().to(torch.int64).numpy()
        else:
            new_token_ids = np.asarray(new_token_ids, dtype=np.int64).ravel()
        if new_token_ids.size == 0:
            return self.total_len
        start = self.total_len
        self.input_ids_np = np.concatenate([self.input_ids_np, new_token_ids])
        # invalidate cache (any append at end requires re-deriving running-max
        # arrays; cheap so we re-derive on demand).
        self._meta_cache_len = 0
        return start

    def begin_pending_dual_turn(self, start_position: Optional[int] = None) -> int:
        """Mark the current branch-local dual turn as not yet merged.

        ``start_position`` is expressed in original physical coordinates and
        defaults to the next token appended to this state (the duplicated
        branch source P_A/P_B). Historical closes remain authoritative; only
        temporary closes in the incomplete current turn are suppressed.
        """
        if self.pending_dual_turn_start is not None:
            raise RuntimeError('a pending dual turn is already active at position '
                               f'{self.pending_dual_turn_start}')
        start = self.total_len if start_position is None else int(start_position)
        if not 0 <= start <= self.total_len:
            raise ValueError(f'pending dual start {start} is outside [0, {self.total_len}]')
        self.pending_dual_turn_start = start
        self._meta_cache_len = 0
        return start

    def _refresh_meta(self) -> None:
        if self._meta_cache_len == self.total_len and self._cur_frame is not None:
            return
        meta = build_step_mask_meta(self.input_ids_np, assistant_role_id=self.assistant_role_id)
        self._is_visual = meta.is_visual
        self._frame_id = meta.frame_id
        self._step_close = meta.step_close.copy()
        if self.pending_dual_turn_start is not None:
            self._step_close[self.pending_dual_turn_start:] = False
        # seg_id[t] = # of step closes strictly before t.
        self._seg_id = (np.cumsum(self._step_close.astype(np.int32)) - self._step_close.astype(np.int32))
        # cur_frame[t] = running max of frame_id, carrying forward through text.
        self._cur_frame = np.maximum.accumulate(np.where(self._frame_id >= 0, self._frame_id, -1))
        self._meta_cache_len = self.total_len

    def get_meta_tensors(self, device) -> Tuple[torch.Tensor, ...]:
        """Return (is_visual, frame_id, seg_id, cur_frame) bool/int32 tensors
        for the FULL cumulative sequence."""
        self._refresh_meta()
        return (
            torch.from_numpy(self._is_visual).to(device=device),
            torch.from_numpy(self._frame_id).to(device=device, dtype=torch.int32),
            torch.from_numpy(self._seg_id).to(device=device, dtype=torch.int32),
            torch.from_numpy(self._cur_frame).to(device=device, dtype=torch.int32),
        )

    def last_step_close_pos(self) -> Optional[int]:
        """Position of the most recent ``</inner>`` / ``</speak>`` token in
        the *original* coordinate system (pre-eviction). None if no step
        close has occurred yet."""
        self._refresh_meta()
        pos = np.flatnonzero(self._step_close)
        return int(pos[-1]) if pos.size else None

    def positions_in_segment(self, segment_id: int) -> np.ndarray:
        """Positions belonging to a given segment id."""
        self._refresh_meta()
        return np.flatnonzero(self._seg_id == segment_id)

    def mark_evicted(self, n_new_evicted_positions: int) -> None:
        """Inform the state that the inference driver has physically dropped
        the leading ``n_new_evicted_positions`` positions (after the current
        ``n_evicted``) from the model's KV cache."""
        self.n_evicted += int(n_new_evicted_positions)
        if self.n_evicted > self.total_len:
            raise ValueError(f'evicted {self.n_evicted} positions but only {self.total_len} exist')


def build_streaming_block_mask(
    state: StreamingStepMaskState,
    q_token_positions: torch.Tensor,
    kv_token_positions: torch.Tensor,
    *,
    num_heads: int = 1,
    valid_kv_len: Optional[int] = None,
    static_meta: Optional[dict] = None,
):
    """Build a flex_attention ``BlockMask`` for a streaming forward.

    Parameters
    ----------
    state               : the per-request cumulative state.
    q_token_positions   : 1-D LongTensor of length Q_LEN, listing the *original*
                          (pre-eviction) sequence positions of each query token
                          in the forward call.
    kv_token_positions  : 1-D LongTensor of length KV_LEN listing the *original*
                          positions of each KV entry in the cache. KV_LEN may
                          exceed ``valid_kv_len`` when using a pre-allocated
                          fixed-shape cache (StreamingCache); the trailing
                          ``KV_LEN - valid_kv_len`` slots are unused buffer
                          and are excluded by the ``kv_idx < valid_kv_len``
                          predicate.
    valid_kv_len        : number of populated KV slots in the leading prefix
                          of ``kv_token_positions``. Defaults to ``KV_LEN``
                          (legacy DynamicCache path: every slot is valid).
    static_meta         : optional dict with pre-allocated
                          ``{is_visual, frame_id, seg_id, cur_frame}`` tensors
                          of a fixed size >= state.total_len. When provided,
                          the resulting BlockMask's ``mask_mod`` closure
                          references THESE tensors (not fresh ones derived
                          from state). Used by GraphDecoder so that across
                          replays the captured mask_mod reads UPDATED meta
                          content rather than stale capture-time tensors.
                          The caller must update the static buffers' contents
                          (e.g. via :func:`copy_state_meta_into_static`) BEFORE
                          calling this function each step.

    Returns
    -------
    BlockMask of shape (Q_LEN, KV_LEN), consumed by flex_attention.
    """
    device = q_token_positions.device
    if static_meta is not None:
        is_visual = static_meta['is_visual']
        frame_id = static_meta['frame_id']
        seg_id = static_meta['seg_id']
        cur_frame = static_meta['cur_frame']
    else:
        is_visual, frame_id, seg_id, cur_frame = state.get_meta_tensors(device)
    W = int(state.window_frames)
    Q_LEN = int(q_token_positions.shape[0])
    KV_LEN = int(kv_token_positions.shape[0])
    # V_LEN must be a 0-D LongTensor when the caller wants the mask_mod to
    # read its value at RUNTIME (e.g. CUDA-graph replay where capture-time
    # V_LEN differs from replay-time). When passed as a Python int it gets
    # folded into the mask_mod closure as a constant.
    if valid_kv_len is None:
        V_LEN = KV_LEN
    elif isinstance(valid_kv_len, torch.Tensor):
        V_LEN = valid_kv_len  # tensor ref; read at runtime
    else:
        V_LEN = int(valid_kv_len)

    # Move the position lookups onto device once.
    q_pos = q_token_positions.to(device=device, dtype=torch.long)
    k_pos = kv_token_positions.to(device=device, dtype=torch.long)

    def mask_mod(b, h, q_idx, kv_idx):
        # ``in_buffer``: reject trailing unused slots in a pre-allocated cache.
        # For the legacy DynamicCache path V_LEN == KV_LEN so this is always
        # true and flex_attention's BlockMask compiler folds it away. With a
        # StreamingCache, V_LEN < KV_LEN and entire trailing BLOCK_N-blocks
        # (32 slots) are full-False -> block-level pruning skips them at zero
        # FLOPs cost.
        in_buffer = kv_idx < V_LEN
        q = q_pos[q_idx]
        k = k_pos[kv_idx]
        causal = q >= k
        text_key = ~is_visual[k]
        same_seg = seg_id[q] == seg_id[k]
        in_window = (cur_frame[q] - frame_id[k]) < W
        return in_buffer & causal & (text_key | (same_seg & in_window))

    # _compile=False at inference: (Q_LEN, KV_LEN) changes every forward
    # (prefill is variable, decode is 1, KV grows each step), and with
    # _compile=True torch._dynamo recompiles each new shape -- catastrophic.
    return create_block_mask(
        mask_mod,
        B=1,
        H=num_heads,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
        _compile=False,
    )


def build_batched_streaming_block_mask(
    states: Sequence[StreamingStepMaskState],
    q_token_positions: torch.Tensor,
    kv_token_positions: torch.Tensor,
    *,
    num_heads: int = 1,
    valid_kv_len: Optional[int | torch.Tensor] = None,
    valid_kv_mask: Optional[torch.Tensor] = None,
):
    """Build a per-sample streaming mask for a real tensor batch.

    Dual-branch decoding starts A and B from the same cache snapshot, but the
    two token streams can encounter ``</inner>`` / ``</speak>`` at different
    decode steps. Those close tokens change ``seg_id`` and therefore visual
    visibility, so broadcasting a single bs=1 BlockMask would be subtly
    incorrect. This builder stacks each active branch's metadata and lets
    ``mask_mod`` index it with FlexAttention's batch coordinate.

    Position tensors accept shared ``[L]`` values or explicit ``[B, L]``
    values. ``valid_kv_mask[B, KV]`` supports arbitrary holes introduced when
    different-query prefixes are padded into one cache tensor; it is mutually
    exclusive with the legacy leading-prefix ``valid_kv_len`` argument.
    """
    if not states:
        raise ValueError('states must contain at least one active stream')
    batch_size = len(states)
    device = q_token_positions.device
    window_frames = int(states[0].window_frames)
    for index, state in enumerate(states):
        if int(state.window_frames) != window_frames:
            raise ValueError('all batched states must use the same window: '
                             f'row0={window_frames}, row{index}={state.window_frames}')

    def _as_batched_positions(tensor: torch.Tensor, name: str) -> torch.Tensor:
        tensor = tensor.to(device=device, dtype=torch.long)
        if tensor.ndim == 1:
            return tensor.unsqueeze(0).expand(batch_size, -1)
        if tensor.ndim == 2 and int(tensor.shape[0]) == batch_size:
            return tensor
        raise ValueError(f'{name} must be [L] or [B, L] with B={batch_size}, '
                         f'got {tuple(tensor.shape)}')

    q_pos = _as_batched_positions(q_token_positions, 'q_token_positions')
    k_pos = _as_batched_positions(kv_token_positions, 'kv_token_positions')
    q_len = int(q_pos.shape[1])
    kv_len = int(k_pos.shape[1])

    meta_rows = [state.get_meta_tensors(device) for state in states]
    max_total_len = max(int(state.total_len) for state in states)

    def _pad_meta(index: int, *, fill: int | bool) -> torch.Tensor:
        rows = []
        for meta in meta_rows:
            tensor = meta[index]
            if int(tensor.shape[0]) < max_total_len:
                tensor = torch.nn.functional.pad(
                    tensor,
                    (0, max_total_len - int(tensor.shape[0])),
                    value=fill,
                )
            rows.append(tensor)
        return torch.stack(rows, dim=0)

    is_visual = _pad_meta(0, fill=False)
    frame_id = _pad_meta(1, fill=-1)
    seg_id = _pad_meta(2, fill=0)
    cur_frame = _pad_meta(3, fill=-1)

    if valid_kv_mask is not None and valid_kv_len is not None:
        raise ValueError('pass only one of valid_kv_len and valid_kv_mask')
    if valid_kv_mask is not None:
        valid_mask = valid_kv_mask.to(device=device, dtype=torch.bool)
        if tuple(valid_mask.shape) != (batch_size, kv_len):
            raise ValueError('valid_kv_mask must have shape '
                             f'{(batch_size, kv_len)}, got {tuple(valid_mask.shape)}')
        valid_len = None
    elif valid_kv_len is None:
        valid_len: int | torch.Tensor = kv_len
    elif isinstance(valid_kv_len, torch.Tensor):
        valid_len = valid_kv_len.to(device=device, dtype=torch.long)
        if valid_len.ndim not in (0, 1):
            raise ValueError('valid_kv_len tensor must be scalar or [B], got '
                             f'{tuple(valid_len.shape)}')
        if valid_len.ndim == 1 and int(valid_len.shape[0]) != batch_size:
            raise ValueError(f'valid_kv_len has B={valid_len.shape[0]}, expected '
                             f'{batch_size}')
    else:
        valid_len = int(valid_kv_len)

    def mask_mod(b, h, q_idx, kv_idx):
        del h
        if valid_kv_mask is not None:
            in_buffer = valid_mask[b, kv_idx]
        else:
            row_valid_len = valid_len[b] if (isinstance(valid_len, torch.Tensor) and valid_len.ndim == 1) else valid_len
            in_buffer = kv_idx < row_valid_len
        q = q_pos[b, q_idx]
        k = k_pos[b, kv_idx]
        causal = q >= k
        text_key = ~is_visual[b, k]
        same_seg = seg_id[b, q] == seg_id[b, k]
        in_window = (cur_frame[b, q] - frame_id[b, k]) < window_frames
        return in_buffer & causal & (text_key | (same_seg & in_window))

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=num_heads,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
        _compile=False,
    )


def copy_state_meta_into_static(state: StreamingStepMaskState, static_meta: dict) -> None:
    """Refresh ``state`` derived metadata and copy into pre-allocated static
    buffers. Buffers must be at least ``state.total_len`` long along dim 0;
    tail entries beyond ``state.total_len`` are zeroed (frame_id/cur_frame
    set to -1 to mark "not a visual position").

    Used by :class:`GraphDecoder` so the captured ``mask_mod`` closure can
    keep reading the SAME tensor address while the contents reflect the
    latest cumulative state. Without this, the captured closure binds to
    the capture-time fresh tensors (which have length = state.total_len at
    capture) and goes out-of-bounds for any later position.
    """
    state._refresh_meta()
    L = int(state.input_ids_np.shape[0])
    iv_src = torch.from_numpy(state._is_visual)
    fi_src = torch.from_numpy(state._frame_id).to(torch.int32)
    si_src = torch.from_numpy(state._seg_id).to(torch.int32)
    cf_src = torch.from_numpy(state._cur_frame).to(torch.int32)
    iv = static_meta['is_visual']
    fi = static_meta['frame_id']
    si = static_meta['seg_id']
    cf = static_meta['cur_frame']
    META_LEN = int(iv.shape[0])
    if L > META_LEN:
        raise RuntimeError(f'state.total_len={L} > static_meta capacity {META_LEN}; '
                           f'increase GraphDecoder.meta_len.')
    iv[:L].copy_(iv_src.to(device=iv.device, non_blocking=True))
    fi[:L].copy_(fi_src.to(device=fi.device, non_blocking=True))
    si[:L].copy_(si_src.to(device=si.device, non_blocking=True))
    cf[:L].copy_(cf_src.to(device=cf.device, non_blocking=True))
    if L < META_LEN:
        iv[L:].zero_()
        # frame_id / cur_frame use -1 sentinel for "not a frame token"; the
        # mask_mod predicate's in_window = (cur_frame[q] - frame_id[k]) < W
        # then evaluates to False for tail positions (kv_pos_static[i]=0 for
        # padding slots already excludes them via in_buffer, but defensive).
        fi[L:].fill_(-1)
        si[L:].zero_()
        cf[L:].fill_(-1)


def streaming_ctx_token(state: StreamingStepMaskState,
                        q_token_positions: torch.Tensor,
                        kv_token_positions: torch.Tensor,
                        valid_kv_len: Optional[int] = None,
                        precomputed_block_mask=None):
    """Set the patch contextvar to dispatch into streaming mode. Returns the
    contextvar reset token; pass it to ``_ctx.reset(token)`` when done.

    ``valid_kv_len`` lets a pre-allocated fixed-shape cache (StreamingCache)
    inform mask_mod about the populated prefix; trailing slots in
    ``kv_token_positions`` are excluded from attention. Defaults to None
    (legacy DynamicCache: every position is populated).

    ``precomputed_block_mask`` short-circuits BlockMask construction inside
    the patched create_causal_mask -- used by GraphDecoder which keeps a
    static BlockMask and updates its tensors in-place outside the captured
    graph."""
    return _ctx.set({
        'window_frames': state.window_frames,
        'assistant_role_id': state.assistant_role_id,
        'streaming': True,
        'state': state,
        'q_token_positions': q_token_positions,
        'kv_token_positions': kv_token_positions,
        'valid_kv_len': valid_kv_len,
        'precomputed_block_mask': precomputed_block_mask,
    })


def reset_ctx(token):
    _ctx.reset(token)


# Default Q_LEN values to warm up flex_attention compilation. Empirically
# (probe_flex_dynamic_v2.py) torch.compile with dynamic=True specialises by
# Q_LEN bucket, so the first prefill at each new bucket stalls 1-7 seconds.
# These three values cover the practical range for our streaming pipeline:
#
#   1     -- every decode step
#   128   -- subsequent frame prefills (delta ~ 60-150 tokens for one user
#            turn + assistant prefix). Q=128 compile is observed to cover
#            Q in [75, 256].
#   1280  -- first-frame prefill (system prompt + Query + first image +
#            assistant prefix, typically 1200-1300 tokens). The Q=1280
#            compile covers Q values >= 512.
DEFAULT_WARMUP_Q_LENS = (1, 128, 1280)


@torch.no_grad()
def warmup_flex_attention(
    model,
    window_frames: int,
    assistant_role_id: int,
    *,
    q_lens=DEFAULT_WARMUP_Q_LENS,
    device='cuda:0',
    verbose: bool = True,
) -> None:
    """Pre-trigger flex_attention compilation at representative Q_LEN values.

    Why: with our :func:`force_dynamic_flex_compile` patch enabled, the
    singleton uses ``torch.compile(flex_attention, dynamic=True)`` which is
    fast at runtime but still recompiles when a new Q_LEN bucket is seen for
    the first time (Inductor specialises per-shape despite ``dynamic=True``).
    Without warmup, the first real prefill stalls for 1-7s on each new Q_LEN.
    Warming up at init amortises this so user-facing inference is steady-state.

    Linear-attention state safety: each warmup forward uses a **fresh**
    ``Qwen3_5DynamicCache`` that is discarded afterwards. The model's parameter
    state is untouched (forward only). After this returns, the caller can
    start real inference with a fresh cache; the linear-attention recurrent
    state has not been polluted.

    Parameters
    ----------
    model              : the patched Qwen3_5ForConditionalGeneration.
    window_frames      : same value used by the production runner; affects
                          mask_mod predicate but not its compiled shape.
    assistant_role_id  : tokenizer's id for "assistant".
    q_lens             : iterable of Q_LEN values to warm up.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

    text_cfg = getattr(model.config, 'text_config', model.config)
    if verbose:
        log.info(f'[warmup] triggering flex compile for Q_LEN={tuple(q_lens)}...')
    import time as _time
    total_t0 = _time.perf_counter()

    for q_len in q_lens:
        # Throwaway state + cache so nothing leaks.
        warmup_state = StreamingStepMaskState(
            window_frames=window_frames,
            assistant_role_id=assistant_role_id,
        )
        # Synthetic tokens: avoid special-token IDs to keep is_visual/frame_id
        # all-False. Token id 100 is a normal text token in the Qwen3 vocab.
        synthetic_ids = torch.full((1, q_len), 100, dtype=torch.long, device=device)
        warmup_state.extend(synthetic_ids[0])
        kept_positions = list(range(q_len))

        q_pos = torch.arange(q_len, device=device, dtype=torch.long)
        kv_pos = torch.tensor(kept_positions, device=device, dtype=torch.long)
        cache = Qwen3_5DynamicCache(config=text_cfg)

        token = streaming_ctx_token(warmup_state, q_pos, kv_pos)
        t0 = _time.perf_counter()
        try:
            model(
                input_ids=synthetic_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                cache_position=q_pos,
                # Must match the kernel_options used at real inference, else
                # the warmup compile and the real-inference compile diverge
                # and the warmup buys us nothing. See infer_stream_masked.py
                # for the rationale of each flag.
                kernel_options={
                    'BLOCK_M': 32,
                    'BLOCK_N': 32,
                    'FORCE_USE_FLEX_ATTENTION': True,
                },
            )
        finally:
            reset_ctx(token)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        dt = _time.perf_counter() - t0
        if verbose:
            log.info(f'[warmup] Q_LEN={q_len}: {dt:.2f}s')
        # Drop the throwaway cache so memory is freed before next warmup.
        del cache
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    if verbose:
        log.info(f'[warmup] done in {(_time.perf_counter()-total_t0):.1f}s '
                 f'across {len(q_lens)} Q_LEN bucket(s)')
