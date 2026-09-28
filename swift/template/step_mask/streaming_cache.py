"""Pre-allocated fixed-shape KV cache for streaming inference.

The HF ``Qwen3_5DynamicCache`` grows its K/V tensors along ``seq_len`` (dim=2)
on every forward via ``torch.cat``. That precludes CUDA graph capture: a
graph requires every tensor shape to be fixed at capture time, and re-capturing
the graph for each KV_LEN is prohibitive.

:class:`StreamingCache` is a drop-in replacement that:

1. Pre-allocates ``key_cache[layer]`` / ``value_cache[layer]`` at the full
   ``max_kv_len`` shape per full-attention layer (8 of 32 for Qwen3.5).
2. Tracks a single ``cur_len`` write pointer — ``update()`` writes the new
   ``q`` slots at ``[cur_len, cur_len + q)`` and returns the FULL buffer.
   ``cur_len`` is advanced by the runner via :meth:`advance` AFTER all layers
   in this forward have written.
3. Lets the BlockMask / ``mask_mod`` rule out positions ``>= cur_len`` (and
   evicted positions). flex_attention's block-level mask culling means the
   trailing-invalid tail is computationally free.
4. Implements eviction via :meth:`evict` — an in-place compact (gather
   surviving slots to the front, zero the tail). The buffer shape never
   changes, so the gather kernel is graph-capturable in a later phase.

The 24 linear-attention layers (Qwen3_5GatedDeltaNet) keep the original lazy
allocation — they store ``conv_states`` / ``recurrent_states`` of shape O(d),
independent of ``seq_len``, and don't need pre-allocation.

This is Phase 2 of the ThinkStream-style streaming inference roadmap. Phase 3
will capture the decode forward as a CUDA graph (requires the fixed shapes
established here). Phase 4 will capture eviction as a separate CUDA graph.

Memory footprint (Qwen3.5-4B, ``max_kv_len=8192``, bf16):
    8 full-attn layers × 2 (K+V) × 1 batch × 4 kv_heads × 8192 × 128 × 2 bytes
    ≈ 134 MB. Comfortable.
"""

from __future__ import annotations

import logging
import torch
from typing import Any, Optional

log = logging.getLogger(__name__)


class StreamingCache:
    """Pre-allocated fixed-shape KV cache. API-compatible with
    :class:`transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DynamicCache`
    (the parts the Qwen3.5 attention layers actually call).
    """

    is_compileable = False

    def __init__(
        self,
        config,
        max_kv_len: int,
        *,
        batch_size: int = 1,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.layer_types = list(config.layer_types)
        self.transformer_layers = [i for i, t in enumerate(self.layer_types) if t == 'full_attention']
        self.last_linear_layer = (len(self.layer_types) - 1 - self.layer_types[::-1].index('linear_attention'))
        num_layers = len(self.layer_types)
        self.max_kv_len = int(max_kv_len)
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else None

        # All four lists are length-``num_layers``. Full-attn layers fill
        # ``key_cache[i]`` / ``value_cache[i]`` at first ``update()``; linear
        # layers fill ``conv_states[i]`` / ``recurrent_states[i]`` at first
        # ``update()`` from the GatedDeltaNet layer. The "wrong" two are
        # ``None`` per layer.
        self.key_cache: list[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: list[Optional[torch.Tensor]] = [None] * num_layers
        self.conv_states: list[Optional[torch.Tensor]] = [None] * num_layers
        self.recurrent_states: list[Optional[torch.Tensor]] = [None] * num_layers

        # Write pointer. All full-attn layers see the same Q per forward, so a
        # single ``cur_len`` shared across layers is correct. The runner must
        # call ``advance(q_len)`` AFTER ``model.forward()`` returns -- update()
        # cannot self-advance because it is called once per layer.
        self.cur_len: int = 0

        # Graph-friendly write position. When ``graph_mode`` is enabled, update()
        # writes via ``index_copy_(2, cur_len_tensor, ...)`` so the K/V slot
        # is determined at REPLAY time from the tensor's current value, not at
        # CAPTURE time from a Python int. The runner advances cur_len_tensor
        # in-place outside the graph between replays.
        self.graph_mode: bool = False
        self.cur_len_tensor: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ core

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``q`` new K/V rows at slots ``[cur_len, cur_len + q)`` and
        return the FULL pre-allocated buffer.

        The returned tensor has shape ``(B, H, max_kv_len, D)`` — caller's
        flex_attention BlockMask MUST also be sized ``max_kv_len`` along the
        KV axis, with ``mask_mod`` rejecting ``kv_idx >= cur_len + q``.
        """
        if self.key_cache[layer_idx] is None:
            self._lazy_allocate_layer(layer_idx, key_states, value_states)

        if self.graph_mode:
            # Graph-friendly path: write via ``index_copy_`` whose index is a
            # TENSOR. At capture time the kernel binds to cur_len_tensor's
            # address; at REPLAY time it reads the current value. Caller has
            # arranged cur_len_tensor[0] = correct slot for this forward.
            # We assume q == 1 in graph mode (decode); prefill is not
            # graph-captured because Q varies.
            self.key_cache[layer_idx].index_copy_(2, self.cur_len_tensor, key_states)
            self.value_cache[layer_idx].index_copy_(2, self.cur_len_tensor, value_states)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        q = int(key_states.shape[2])
        write_start = self.cur_len
        write_end = write_start + q
        if write_end > self.max_kv_len:
            raise RuntimeError(f'StreamingCache overflow: cur_len={self.cur_len} + q={q} = '
                               f'{write_end} > max_kv_len={self.max_kv_len}. Raise '
                               f'``max_kv_len`` or compact via ``evict``.')

        # In-place write. Source dtype/device match dest by the lazy-allocate
        # invariant (we allocate using the first key_states' dtype/device).
        self.key_cache[layer_idx][:, :, write_start:write_end, :] = key_states
        self.value_cache[layer_idx][:, :, write_start:write_end, :] = value_states
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def advance(self, q: int) -> None:
        """Advance ``cur_len`` after all layers in this forward have written.

        Called by the runner AFTER ``model.forward()`` returns. update() cannot
        self-advance because each full-attention layer's update() must see the
        same starting ``cur_len`` (they all write at the same slot range).
        """
        new_len = self.cur_len + int(q)
        if new_len > self.max_kv_len:
            raise RuntimeError(f'StreamingCache overflow at advance: '
                               f'{self.cur_len} + {q} > {self.max_kv_len}')
        self.cur_len = new_len
        if self.cur_len_tensor is not None:
            self.cur_len_tensor.fill_(self.cur_len)

    def enable_graph_mode(self, device) -> None:
        """Switch update() to the graph-capturable index_copy_ path.

        Allocates ``cur_len_tensor`` at a stable address; the captured graph
        binds to this tensor so each replay reads the current write slot.
        """
        if self.cur_len_tensor is None:
            self.cur_len_tensor = torch.tensor([self.cur_len], device=device, dtype=torch.long)
        else:
            self.cur_len_tensor.fill_(self.cur_len)
        self.graph_mode = True

    def disable_graph_mode(self) -> None:
        """Return to the slice-assignment update path. cur_len_tensor stays
        allocated for cheap re-enable."""
        self.graph_mode = False

    def evict(self, keep_indices: torch.Tensor) -> None:
        """Compact the cache: keep only the slots listed in ``keep_indices``,
        shift them to the front, zero the tail, set ``cur_len = len(keep_indices)``.

        ``keep_indices`` are physical slot indices in ``[0, cur_len)``. After
        eviction the buffer is still ``max_kv_len``-shaped; only ``cur_len``
        decreased. Caller must update its own original-position bookkeeping
        (``kept_positions`` in the runner) in lockstep.
        """
        if keep_indices.numel() == 0:
            # Eviction would drop everything; clear cur_len to 0. Zero the
            # used prefix to avoid stale RoPE'd K/V poisoning future attends
            # (mask_mod should reject these too, but be defensive).
            self._zero_tail(0)
            self.cur_len = 0
            if self.cur_len_tensor is not None:
                self.cur_len_tensor.fill_(0)
            return

        n_keep = int(keep_indices.shape[0])
        if n_keep == self.cur_len:
            # Nothing dropped.
            return

        for li in self.transformer_layers:
            k = self.key_cache[li]
            v = self.value_cache[li]
            if k is None or v is None:
                continue
            # Gather surviving slots into a temporary, write back compactly,
            # zero the tail. ``index_select`` allocates a new (B, H, n_keep, D)
            # tensor; we then copy it into the leading ``n_keep`` slots of the
            # pre-allocated buffer.
            k_keep = k.index_select(2, keep_indices)
            v_keep = v.index_select(2, keep_indices)
            self.key_cache[li][:, :, :n_keep, :].copy_(k_keep)
            self.value_cache[li][:, :, :n_keep, :].copy_(v_keep)
            # Zero the tail. Necessary because the mask_mod for partially-
            # populated buffer relies on ``kv_idx < cur_len``, but a defensive
            # zero protects against accidental attend-to-stale-K bugs.
            self.key_cache[li][:, :, n_keep:, :].zero_()
            self.value_cache[li][:, :, n_keep:, :].zero_()
        self.cur_len = n_keep
        if self.cur_len_tensor is not None:
            self.cur_len_tensor.fill_(n_keep)

    # ------------------------------------------------------------------ meta

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Logical sequence length (= ``cur_len``), NOT the buffer size. The
        attention kernels call this to decide how to slice positions for
        causal masking — they expect the number of past+current tokens, not
        the storage capacity.
        """
        return self.cur_len

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """Return ``(kv_length, kv_offset)``.

        For our streaming pipeline this getter is NOT consulted -- our patched
        ``create_causal_mask`` in :mod:`step_mask.patch` returns a BlockMask
        sized at ``max_kv_len`` regardless. We still implement it consistently
        for any defensive callers.
        """
        return self.max_kv_len, 0

    @property
    def has_previous_state(self) -> bool:
        """Mirror of Qwen3_5DynamicCache: have we initialised the linear-
        attention conv state yet?"""
        return self.conv_states[self.last_linear_layer] is not None

    def __len__(self) -> int:
        return len(self.layer_types)

    # ---------------------------------------------------------------- helpers

    def _lazy_allocate_layer(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """First-touch allocation. We size to (B, num_kv_heads, max_kv_len, D)
        from the first call's tensors -- this avoids hard-coding head counts
        from config and works uniformly across hybrid layer types."""
        B, H, _, D = key_states.shape
        if B != self.batch_size:
            log.warning(f'[StreamingCache] update called with B={B}, expected '
                        f'batch_size={self.batch_size}; using provided B')
            self.batch_size = B
        # Inherit dtype/device from the incoming K (not from __init__'s hints)
        # so the buffer matches whatever the attention layer produces.
        target_dtype = key_states.dtype
        target_device = key_states.device
        self.key_cache[layer_idx] = torch.zeros(B, H, self.max_kv_len, D, dtype=target_dtype, device=target_device)
        self.value_cache[layer_idx] = torch.zeros(
            B, H, self.max_kv_len, D, dtype=value_states.dtype, device=target_device)
        # Record device for downstream (e.g., evict's keep_indices).
        if self.device is None:
            self.device = target_device

    def _zero_tail(self, from_slot: int) -> None:
        """Zero K/V buffers from slot ``from_slot`` onward. Used by evict
        when the entire cache is dropped."""
        for li in self.transformer_layers:
            if self.key_cache[li] is not None:
                self.key_cache[li][:, :, from_slot:, :].zero_()
                self.value_cache[li][:, :, from_slot:, :].zero_()

    # ------------------------- HF-compat (rarely used in our pipeline) -------

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Beam search reorder. Not used in single-stream inference; kept
        for API parity."""
        for li in range(len(self.key_cache)):
            if self.key_cache[li] is not None:
                bi = beam_idx.to(self.key_cache[li].device)
                self.key_cache[li] = self.key_cache[li].index_select(0, bi)
                self.value_cache[li] = self.value_cache[li].index_select(0, bi)
            if self.conv_states[li] is not None:
                bi = beam_idx.to(self.conv_states[li].device)
                self.conv_states[li] = self.conv_states[li].index_select(0, bi)
                self.recurrent_states[li] = self.recurrent_states[li].index_select(0, bi)
