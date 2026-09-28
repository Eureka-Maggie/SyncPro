"""Phase 3: CUDA-graph-captured decode for streaming inference.

After Phase 2 made the attention compute fixed-shape, the decode forward
(Q_LEN=1, KV_LEN=max_kv_len, BLOCK fixed) consists of a fixed sequence of
CUDA kernels with stable input/output addresses. ``GraphDecoder`` captures
that sequence once and replays it for every decode step, eliminating
per-step Python overhead + kernel-launch overhead.

What gets captured
------------------
1. ``model.forward(input_ids=static_in, past_key_values=cache, ...)``
   - Internal flex_attention call uses ``ctx['precomputed_block_mask']``
     short-circuit to read the static BlockMask without rebuilding it.
   - StreamingCache.update() runs in ``graph_mode``: the K/V write uses
     ``index_copy_`` with ``cur_len_tensor`` as the GPU index, so the
     replay reads the current slot value from the tensor.
   - cur_len advances in-place via ``cur_len_tensor.add_(1)``.

What runs OUTSIDE the captured graph (every step)
-------------------------------------------------
* Build a fresh BlockMask via ``build_streaming_block_mask`` -- has host
  numpy ops + create_block_mask Python loop, disallowed during capture.
* Copy the new BlockMask's tensors into the static BlockMask in-place.
* Update ``input_ids_static[0, 0]`` to the new token.
* Update ``cache_position_static[0]`` to the new position.

Argmax sampling happens after replay using the static logits buffer.

Why this works
--------------
PyTorch CUDA graphs capture kernel calls + the tensor addresses they bind
to. They do NOT re-execute Python code on replay. Since:
  - All tensor SHAPES are fixed (Phase 2 prereq)
  - All BlockMask tensor shapes are fixed (verified by probe)
  - cur_len position is read from a tensor (not Python int)
the captured kernels operate on the latest tensor *contents* every replay,
which is exactly what we need for autoregressive decode.
"""

from __future__ import annotations

import logging
import torch
from typing import Optional

from .inference import (StreamingStepMaskState, build_streaming_block_mask, copy_state_meta_into_static, reset_ctx,
                        streaming_ctx_token)

log = logging.getLogger(__name__)

# Kernel options must match those used in MaskedStreamingRunner._forward_with_mask
# and warmup_flex_attention. Diverging produces a separate triton compile and
# the warmup buys us nothing.
DECODE_KERNEL_OPTS = {
    'BLOCK_M': 32,
    'BLOCK_N': 32,
    'FORCE_USE_FLEX_ATTENTION': True,
}


class _patch_linear_attn_mask_to_none:
    """Context manager that replaces the language_model's
    ``_update_linear_attn_mask`` with a no-op returning None.

    The original implementation does ``cache_position[0] > 0`` which is a
    host-side comparison disallowed during ``torch.cuda.graph`` capture.
    During decode (cache_position > 0) the method always returns None
    anyway, so short-circuiting is safe.
    """

    def __init__(self, model):
        # Navigate the wrapper chain to find the qwen3_5 text model that
        # owns _update_linear_attn_mask. Qwen3_5ForConditionalGeneration
        # -> .model (Qwen3_5Model VLM wrapper) -> .language_model
        # (Qwen3_5TextModel that owns _update_linear_attn_mask).
        for path in [
            ('model', 'language_model'),
            ('model', ),
            (),
        ]:
            cur = model
            ok = True
            for attr in path:
                cur = getattr(cur, attr, None)
                if cur is None:
                    ok = False
                    break
            if ok and hasattr(cur, '_update_linear_attn_mask'):
                self.target = cur
                break
        else:
            raise RuntimeError('Could not locate _update_linear_attn_mask on model')
        self.original = None

    def __enter__(self):
        self.original = self.target._update_linear_attn_mask
        # Replace with a no-op that returns None (decode-path equivalent).
        self.target._update_linear_attn_mask = lambda *a, **kw: None
        return self

    def __exit__(self, *exc):
        self.target._update_linear_attn_mask = self.original


def copy_block_mask_(dst, src) -> None:
    """In-place copy of all tensor attributes from ``src`` BlockMask into
    ``dst``. Both BlockMasks must have IDENTICAL shapes for every attribute
    (guaranteed by fixed (Q_LEN, KV_LEN, BLOCK_SIZE) across both).
    """
    # Direct sparse-layout tensors.
    dst.kv_num_blocks.copy_(src.kv_num_blocks)
    dst.kv_indices.copy_(src.kv_indices)
    # Transposed Q-side layout (auto-derived by from_kv_blocks).
    dst.q_num_blocks.copy_(src.q_num_blocks)
    dst.q_indices.copy_(src.q_indices)
    # full_* are optional but always present in our streaming masks (because
    # mask_mod returns False for some positions -> partial blocks).
    if dst.full_kv_num_blocks is not None and src.full_kv_num_blocks is not None:
        dst.full_kv_num_blocks.copy_(src.full_kv_num_blocks)
        dst.full_kv_indices.copy_(src.full_kv_indices)
        dst.full_q_num_blocks.copy_(src.full_q_num_blocks)
        dst.full_q_indices.copy_(src.full_q_indices)


class GraphDecoder:
    """Owns the captured decode graph + static tensors. One instance per
    runner; capture happens lazily on the first ``decode_step``.
    """

    def __init__(self,
                 model,
                 cache,
                 state: StreamingStepMaskState,
                 max_kv_len: int,
                 device: str | torch.device,
                 meta_len: int = 200_000):
        self.model = model
        self.cache = cache
        self.state = state
        self.max_kv_len = max_kv_len
        self.device = torch.device(device)

        # Static input tensors at fixed addresses. Their CONTENTS get updated
        # before each replay; the captured kernels read whatever's there.
        self.input_ids_static = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.cache_position_static = torch.zeros((1, ), dtype=torch.long, device=self.device)
        self.q_pos_static = torch.zeros((1, ), dtype=torch.long, device=self.device)
        # kv_token_positions is sized at max_kv_len; leading valid_kv_len
        # entries are kept_positions, rest is zero (mask blocks them).
        self.kv_pos_static = torch.zeros((max_kv_len, ), dtype=torch.long, device=self.device)
        # valid_kv_len as a 0-D LongTensor so the mask_mod closure captured at
        # capture time reads its current value at REPLAY time. Without this the
        # int valid_kv_len gets folded into mask_mod as a Python constant
        # (= n_kept at capture), so subsequent replays mask out every slot
        # written AFTER capture (including the first decode token's own KV at
        # slot[n_kept_at_capture]) — directly causing the L∞=2.0 logit drift
        # we observed at the very first replay.
        self.valid_kv_len_static = torch.zeros((), dtype=torch.long, device=self.device)

        # Static meta tensors for mask_mod (CRITICAL — see docstring of
        # ``copy_state_meta_into_static``). Without these the captured
        # mask_mod closure binds to capture-time ``state.get_meta_tensors()``
        # output (fresh tensors of length L_capture). As ``state.total_len``
        # grows past L_capture in subsequent frames, the captured closure
        # reads out-of-bounds → wrong mask → cumulative drift starting
        # around frame 8 of a PVQA-like trace.
        #
        # We pre-allocate at ``meta_len`` (default 200k, covers PVQA's
        # 360-frame queries with safety margin) and copy state.input_ids_np-
        # derived metadata into these buffers IN PLACE before each
        # build_streaming_block_mask call (both at capture and at every
        # decode_step). Memory cost: 200k × (1 bool + 3 int32) ≈ 2.6 MB.
        self.meta_len = int(meta_len)
        self.static_meta = {
            'is_visual': torch.zeros((self.meta_len, ), dtype=torch.bool, device=self.device),
            'frame_id': torch.full((self.meta_len, ), -1, dtype=torch.int32, device=self.device),
            'seg_id': torch.zeros((self.meta_len, ), dtype=torch.int32, device=self.device),
            'cur_frame': torch.full((self.meta_len, ), -1, dtype=torch.int32, device=self.device),
        }

        # Static BlockMask: built once during warmup; contents updated each step.
        self.static_bm = None
        # Captured graph + static logits output.
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.logits_static: Optional[torch.Tensor] = None

        # State-update bookkeeping. Track our own python int cur_len that
        # mirrors cache.cur_len_tensor[0]. Used to compose mask_mod each step.
        self._next_position: int = 0

        # Linear-attention READ tensor references. These are the tensors the
        # captured graph BINDS to for reading conv_state / recurrent_state.
        # We hold Python refs so they stay alive across frames. The
        # bridge_post_prefill / bridge_post_decode methods keep their values
        # in sync with cache.conv_states[i] / recurrent_states[i] across
        # the non-graph prefill <-> captured decode boundary.
        self.captured_read_conv_states: list = []
        self.captured_read_recur_states: list = []

        # Linear-attention WRITE tensor references for recurrent_states only.
        # The Qwen3_5 GatedDeltaNet does an UNCONDITIONAL rebind at every
        # forward (line 618 of modeling_qwen3_5.py:
        # ``cache_params.recurrent_states[layer_idx] = last_recurrent_state``).
        # During capture this binds the captured-graph WRITE address to an
        # in-graph mempool tensor T_lrc. After capture, T_lrc's only Python
        # ref is ``cache.recurrent_states[i]`` — which the FIRST eager-mode
        # prefill (next frame) overwrites with a fresh tensor, dropping the
        # last ref to T_lrc. The mempool then reuses T_lrc's address, and
        # subsequent graph replays clobber/read garbage at that address →
        # cumulative floating-point drift that starts diverging from LEG
        # around frame 8 of a typical PVQA-like trace.
        #
        # Fix: snapshot ``cache.recurrent_states`` immediately after capture
        # so we hold Python refs to T_lrc and prevent GC + mempool reuse.
        # conv_states does NOT need this because its decode path is in-place
        # (causal_conv1d_update), so no rebind during graph capture; its
        # captured_read refs already pin everything.
        self.captured_write_recur_states: list = []

    @torch.no_grad()
    def capture(self, initial_position: int, initial_kept_positions: list[int]) -> None:
        """Build the static BlockMask, warm up kernels, capture the decode
        forward. Must be called once after prefill and BEFORE the decode
        loop starts.

        ``initial_position``    : the position of the FIRST decode-token query
                                  (= state.total_len at end of prefill).
        ``initial_kept_positions``: kept_positions list AFTER prefill (i.e.
                                    the original positions occupying cache
                                    slots [0, len(kept_positions))).
        """
        # 1. Fill static input/position tensors. We must use a PLACEHOLDER q
        # position INSIDE state's metadata-array bounds for the capture-time
        # mask build; state arrays have length `state.total_len` = the count
        # of tokens already extended, so the last valid index is total_len-1.
        # The FIRST real decode_step will overwrite these with the actual
        # forwarded position.
        placeholder_q = max(0, self.state.total_len - 1)
        self.input_ids_static[0, 0] = 0  # placeholder; runner overwrites
        self.cache_position_static[0] = initial_position
        self.q_pos_static[0] = placeholder_q
        n_kept = len(initial_kept_positions)
        self.kv_pos_static[:n_kept].copy_(torch.tensor(initial_kept_positions, device=self.device, dtype=torch.long))
        self.kv_pos_static[n_kept:].zero_()

        # 2. Build the initial BlockMask. We need this BEFORE switching cache
        # to graph_mode so the build doesn't conflict with index_copy_ paths.
        # cur_len_tensor was set up by enable_graph_mode previously.
        #
        # Populate static meta buffers from the current state. The resulting
        # static_bm's mask_mod closure binds to ``self.static_meta`` tensor
        # addresses (which persist forever) — subsequent decode_step calls
        # refresh static_meta CONTENTS in-place, so the captured kernel
        # reads up-to-date metadata every replay.
        copy_state_meta_into_static(self.state, self.static_meta)
        # Fill the 0-D tensor so mask_mod (closing over self.valid_kv_len_static
        # as a TENSOR ref) sees the right initial value during sparse-layout
        # materialization here, and reads it dynamically at each replay below.
        self.valid_kv_len_static.fill_(n_kept)
        valid_kv_len = n_kept
        self.static_bm = build_streaming_block_mask(
            state=self.state,
            q_token_positions=self.q_pos_static,
            kv_token_positions=self.kv_pos_static,
            num_heads=1,
            valid_kv_len=self.valid_kv_len_static,
            static_meta=self.static_meta,
        )

        # 3. Snapshot POST-PREFILL VALUES of conv/recurrent state. These
        # are correct evolution values we'll restore both at the post-warmup
        # READ tensor and the post-capture WRITE tensor below.
        saved_conv_values = [t.clone() if t is not None else None for t in self.cache.conv_states]
        saved_recur_values = [t.clone() if t is not None else None for t in self.cache.recurrent_states]

        # 4. Warmup forwards. These allocate cublas/cudnn workspaces +
        # prime kernel paths so the capture below doesn't hit any
        # host-sync code path during streamCapture. Each warmup rebinds
        # cache.conv_states[i] to a fresh tensor (T2,T3,T4 in graph
        # mempool? regular mempool actually, since we're outside capture
        # context here). After warmup, cache.conv_states[i] points to T4.
        warm_init = int(self.cache.cur_len)
        with _patch_linear_attn_mask_to_none(self.model):
            for _ in range(3):
                self.cache.cur_len_tensor.fill_(warm_init)
                token = streaming_ctx_token(
                    self.state,
                    self.q_pos_static,
                    self.kv_pos_static,
                    valid_kv_len=valid_kv_len,
                    precomputed_block_mask=self.static_bm,
                )
                try:
                    _ = self.model(
                        input_ids=self.input_ids_static,
                        past_key_values=self.cache,
                        use_cache=True,
                        cache_position=self.cache_position_static,
                        kernel_options=DECODE_KERNEL_OPTS,
                    )
                finally:
                    reset_ctx(token)
            self.cache.cur_len_tensor.fill_(warm_init)
            self.cache.cur_len = warm_init

            # 4a. Restore the POST-PREFILL conv/recurrent values into the
            # post-warmup tensors (T4). The captured graph's forward will
            # READ from T4's address -- it needs correct uncontaminated
            # values there.
            for i, saved in enumerate(saved_conv_values):
                cur = self.cache.conv_states[i]
                if saved is None or cur is None:
                    continue
                cur.copy_(saved)
            for i, saved in enumerate(saved_recur_values):
                cur = self.cache.recurrent_states[i]
                if saved is None or cur is None:
                    continue
                cur.copy_(saved)

            # 4b. Snapshot the REFERENCES of these post-warmup tensors NOW
            # -- this is what the captured forward will READ. We must hold
            # Python references so they're not GC'd between captures.
            self.captured_read_conv_states = list(self.cache.conv_states)
            self.captured_read_recur_states = list(self.cache.recurrent_states)
            captured_read_conv_states = self.captured_read_conv_states
            captured_read_recur_states = self.captured_read_recur_states

            # 5. Capture.
            s = torch.cuda.Stream(device=self.device)
            s.wait_stream(torch.cuda.current_stream(self.device))
            torch.cuda.current_stream(self.device).wait_stream(s)
            token = streaming_ctx_token(
                self.state,
                self.q_pos_static,
                self.kv_pos_static,
                valid_kv_len=valid_kv_len,
                precomputed_block_mask=self.static_bm,
            )
            try:
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=s):
                    out = self.model(
                        input_ids=self.input_ids_static,
                        past_key_values=self.cache,
                        use_cache=True,
                        cache_position=self.cache_position_static,
                        kernel_options=DECODE_KERNEL_OPTS,
                    )
                    # Capture the cur_len advance as part of the graph too.
                    self.cache.cur_len_tensor.add_(1)
                    # Bridge linear-attn state: at this point, each
                    # cache.conv_states[i] / recurrent_states[i] points to
                    # a freshly rebound tensor (the WRITE target). The
                    # captured READ tensor (held in captured_read_*_states)
                    # has fixed address but stale content -- copy the WRITE
                    # values into the READ tensor IN-GRAPH so the next
                    # replay's forward reads correct evolved state.
                    for i, rd in enumerate(captured_read_conv_states):
                        wr = self.cache.conv_states[i]
                        if rd is None or wr is None:
                            continue
                        if rd.data_ptr() == wr.data_ptr():
                            continue
                        rd.copy_(wr)
                    for i, rd in enumerate(captured_read_recur_states):
                        wr = self.cache.recurrent_states[i]
                        if rd is None or wr is None:
                            continue
                        if rd.data_ptr() == wr.data_ptr():
                            continue
                        rd.copy_(wr)
                    self.logits_static = out.logits
            finally:
                reset_ctx(token)

        # The capture itself ran the forward + advance once, so
        # cur_len_tensor is now warm_init+1. Reset to warm_init so the FIRST
        # real decode_step replay writes to the correct (warm_init) slot.
        self.cache.cur_len_tensor.fill_(warm_init)
        self.cache.cur_len = warm_init

        # Zero-contamination restore: copy the pre-capture state values
        # into whatever tensors cache.conv_states[i] / recurrent_states[i]
        # currently point to (these were rebound to graph-mempool tensors
        # during the warmup + capture forwards). The captured graph
        # references those addresses; writing the correct values there
        # means the next decode replay reads UNCONTAMINATED state.
        for i, saved in enumerate(saved_conv_values):
            cur = self.cache.conv_states[i]
            if saved is None or cur is None:
                continue
            cur.copy_(saved)
            rd = self.captured_read_conv_states[i] \
                if i < len(self.captured_read_conv_states) else None
            if rd is not None and rd.data_ptr() != cur.data_ptr():
                rd.copy_(saved)
        for i, saved in enumerate(saved_recur_values):
            cur = self.cache.recurrent_states[i]
            if saved is None or cur is None:
                continue
            cur.copy_(saved)
            # CRITICAL: the in-graph bridge `rd.copy_(wr)` captured at
            # line ~362 ran DURING the capture forward, after line 618 of
            # modeling_qwen3_5.py rebound cache.recurrent_states[i] to a
            # fresh in-graph tensor (wr). That bridge wrote wr's
            # post-capture-decode values into rd (= captured_read_recur_states[i]
            # = the warmup-rebound T_lrc_warm3 tensor in eager mempool).
            # We MUST overwrite rd's content with the saved post-prefill
            # values too, otherwise the first replay's forward (which reads
            # from rd at line 534) sees stale capture-forward output and
            # the very first decode step diverges from eager.
            rd = self.captured_read_recur_states[i] \
                if i < len(self.captured_read_recur_states) else None
            if rd is not None and rd.data_ptr() != cur.data_ptr():
                rd.copy_(saved)

        # Pin the in-graph WRITE tensor for recurrent_states (see member
        # docstring). At this point cache.recurrent_states[i] is the in-graph
        # tensor that line 618 of modeling_qwen3_5.py rebound to during the
        # capture forward. Holding Python refs keeps these tensors alive
        # forever, preventing the next eager prefill from triggering GC +
        # mempool reuse of their addresses.
        self.captured_write_recur_states = list(self.cache.recurrent_states)

        log.info(f'[GraphDecoder] captured decode forward '
                 f'(input shape={tuple(self.input_ids_static.shape)}, '
                 f'kv_len={self.max_kv_len})')

    def invalidate(self) -> None:
        """Drop the captured graph (forces ``capture()`` next decode_step).
        Use sparingly -- ``bridge_post_prefill`` lets the existing graph
        survive across frames so we don't pay capture cost per frame.
        """
        self.graph = None
        self.logits_static = None
        self.static_bm = None
        self.captured_read_conv_states = []
        self.captured_read_recur_states = []
        self.captured_write_recur_states = []

    @torch.no_grad()
    def bridge_post_prefill(self) -> None:
        """Bring the captured READ tensors in sync with the post-prefill
        conv/recurrent state values, then rebind ``cache.conv_states[i]``
        to point at the READ tensor.

        Call AFTER each prefill (which rebound cache.conv_states[i] to a
        fresh tensor with post-prefill values) and BEFORE the captured
        decode replays. Without this, the first replay reads stale
        end-of-previous-decode values.
        """
        for i, rd in enumerate(self.captured_read_conv_states):
            cur = self.cache.conv_states[i]
            if rd is None or cur is None:
                continue
            if rd.data_ptr() == cur.data_ptr():
                continue
            rd.copy_(cur)
            # Rebind cache list slot to the READ tensor so the NEXT prefill
            # (frame N+2) reads correct post-decode values directly.
            self.cache.conv_states[i] = rd
        for i, rd in enumerate(self.captured_read_recur_states):
            cur = self.cache.recurrent_states[i]
            if rd is None or cur is None:
                continue
            if rd.data_ptr() == cur.data_ptr():
                continue
            rd.copy_(cur)
            self.cache.recurrent_states[i] = rd

    @torch.no_grad()
    def decode_step(self, next_token_id: int, next_position: int, kept_positions: list[int], sampler=None):
        """Run one captured decode step.

        Returns:
            - if ``sampler`` is None (default, greedy argmax): the sampled
              token id (int). EXACT same path as before this hook -- no
              extra tensor ops, no logp computation.
            - if ``sampler`` is provided: a tuple ``(token_id, log_prob)``.

        Caller is responsible for keeping ``state``, ``kept_positions``,
        and the cache's cur_len in sync. This method only handles the
        replay machinery.
        """
        assert self.graph is not None, 'capture() must be called first'

        # 1. Update static inputs in place.
        self.input_ids_static[0, 0] = next_token_id
        self.cache_position_static[0] = next_position
        self.q_pos_static[0] = next_position
        n_kept = len(kept_positions)
        self.kv_pos_static[:n_kept].copy_(torch.tensor(kept_positions, device=self.device, dtype=torch.long))
        if n_kept < self.max_kv_len:
            self.kv_pos_static[n_kept:].zero_()

        # 2. Refresh static meta tensors so static_bm.mask_mod (captured at
        # capture time, references self.static_meta addresses) sees current
        # cumulative state. Then rebuild BlockMask outside graph to get
        # current sparse layout, copy into static.
        copy_state_meta_into_static(self.state, self.static_meta)
        # CRITICAL: update the tensor BEFORE building new_bm. The captured
        # static_bm.mask_mod closure refs self.valid_kv_len_static (the SAME
        # tensor), so this update propagates to replay too.
        self.valid_kv_len_static.fill_(n_kept)
        new_bm = build_streaming_block_mask(
            state=self.state,
            q_token_positions=self.q_pos_static,
            kv_token_positions=self.kv_pos_static,
            num_heads=1,
            valid_kv_len=self.valid_kv_len_static,
            static_meta=self.static_meta,
        )
        copy_block_mask_(self.static_bm, new_bm)

        # 3. Replay.
        self.graph.replay()

        # 4. Sample. Default greedy path is EXACTLY the original line.
        if sampler is None:
            next_token = int(self.logits_static[0, -1].argmax().item())
            self.cache.cur_len += 1
            return next_token
        next_token, log_prob = sampler(self.logits_static[0, -1])
        self.cache.cur_len += 1
        return next_token, log_prob
