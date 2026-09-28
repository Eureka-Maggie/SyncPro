"""Phase 4: CUDA-graph-captured cache eviction.

Eviction in Phase 2 does, for each full-attention layer:
  k_keep = K_buf.index_select(2, keep_indices)
  K_buf[:, :, :n_keep, :].copy_(k_keep)
  K_buf[:, :, n_keep:, :].zero_()
  (same for V)

The only variables across evictions are ``keep_indices`` (variable length)
and ``n_keep`` (variable int). Capturing the gather + copy + zero loop in a
CUDA graph requires fixed shapes.

Strategy:
1. Pad ``keep_indices`` to ``max_kv_len`` with self-references (unused slots
   point at slot 0 -- a valid index that produces benign garbage, masked out
   by step 3).
2. Track ``n_keep`` in a 1-element GPU tensor.
3. Captured ops:
     - K_gathered = K_buf.index_select(2, padded_keep_indices)  # full MAX shape
     - K_buf.copy_(K_gathered)
     - K_buf *= valid_mask_b   # zeros slots i >= n_keep
   where ``valid_mask = arange(max_kv_len) < n_keep`` is computed in-graph.
4. Update ``cache.cur_len_tensor`` from ``n_keep_tensor`` as part of the
   captured ops.

Linear-attn state (conv_states / recurrent_states) is NOT evicted -- it's
O(d), not per-position. Phase 4 only graphs the full-attn K/V eviction.
"""

from __future__ import annotations

import logging
import torch
from typing import Optional

log = logging.getLogger(__name__)


class EvictionGraph:
    """Captured K/V compaction graph. One instance per StreamingCache.

    Lifecycle:
      1. ``capture()`` -- builds the static tensors and captures the graph.
      2. ``replay(keep_indices)`` -- updates the static keep_indices and
         n_keep_tensor in place, then replays the captured graph. Returns
         the new ``cur_len``.
    """

    def __init__(self, cache, device: str | torch.device):
        self.cache = cache
        self.max_kv_len = cache.max_kv_len
        self.device = torch.device(device)
        self.graph: Optional[torch.cuda.CUDAGraph] = None

        # Static tensors at fixed addresses.
        self.padded_keep_indices_static = torch.zeros(self.max_kv_len, dtype=torch.long, device=self.device)
        self.n_keep_tensor_static = torch.zeros(1, dtype=torch.long, device=self.device)
        # Pre-computed arange(max_kv_len) used inside the captured graph
        # to build valid_mask. Materialised once; address stable.
        self._arange_max = torch.arange(self.max_kv_len, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def capture(self) -> None:
        """Build and capture the eviction graph. The cache must have all
        full-attention K/V buffers already allocated (call after at least
        one prefill so ``cache.key_cache[i]`` is not None).
        """
        # Initialise the static buffers with a benign no-op eviction:
        # keep all current slots.
        cur = int(self.cache.cur_len)
        self.padded_keep_indices_static[:cur].copy_(torch.arange(cur, device=self.device, dtype=torch.long))
        self.padded_keep_indices_static[cur:].zero_()
        self.n_keep_tensor_static.fill_(cur)

        # Warmup: run the eager eviction sequence twice so cublas/cudnn
        # workspaces are allocated. Use the actual buffers so dtype/shape
        # are correct.
        for _ in range(2):
            self._evict_eager(self.padded_keep_indices_static, self.n_keep_tensor_static)

        # Capture.
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        torch.cuda.current_stream(self.device).wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            self._evict_eager(self.padded_keep_indices_static, self.n_keep_tensor_static)

        log.info(f'[EvictionGraph] captured K/V compaction graph '
                 f'(max_kv_len={self.max_kv_len}, '
                 f'n_layers={len(self.cache.transformer_layers)})')

    def _evict_eager(self, padded_keep_indices: torch.Tensor, n_keep_tensor: torch.Tensor) -> None:
        """The captured op sequence. Runs eagerly during warmup + captured
        during ``with torch.cuda.graph``."""
        # valid_mask: (max_kv_len,) bool -> (1, 1, max_kv_len, 1) for broadcast.
        valid_mask = (self._arange_max < n_keep_tensor).to(torch.bfloat16)
        valid_mask_b = valid_mask.view(1, 1, -1, 1)

        for li in self.cache.transformer_layers:
            k = self.cache.key_cache[li]
            v = self.cache.value_cache[li]
            if k is None or v is None:
                continue
            # Gather + write back + zero the tail.
            k_gathered = k.index_select(2, padded_keep_indices)
            v_gathered = v.index_select(2, padded_keep_indices)
            k.copy_(k_gathered)
            v.copy_(v_gathered)
            k.mul_(valid_mask_b)
            v.mul_(valid_mask_b)

        # Update cache.cur_len_tensor in place.
        if self.cache.cur_len_tensor is not None:
            self.cache.cur_len_tensor.copy_(n_keep_tensor)

    @torch.no_grad()
    def replay(self, keep_indices: torch.Tensor) -> int:
        """Update static inputs and replay the captured eviction graph.

        Returns the new ``cur_len`` (= len(keep_indices)). The caller
        should sync its Python-side ``cache.cur_len`` to match.
        """
        assert self.graph is not None, 'capture() must be called first'
        n_keep = int(keep_indices.shape[0])

        # Update static inputs in place.
        self.padded_keep_indices_static[:n_keep].copy_(keep_indices)
        # Fill the rest with 0 (slot 0 is always valid; the post-gather
        # multiply by valid_mask zeros these slots anyway).
        self.padded_keep_indices_static[n_keep:].zero_()
        self.n_keep_tensor_static.fill_(n_keep)

        self.graph.replay()
        return n_keep
