"""Trajectory-replay helper for RL ref_logp / behaviour_logp computation.

When training (GRPO / PPO), the trainer needs ``ref_logp(action_t | state_t)``
for each action token in a rolled-out trajectory. ``compute_logps_for_trajectory``
provides this in a way that goes through the SAME mask / cache / dispatch
infrastructure as rollout, so the bf16 numerics line up.

Key design choice
-----------------
We do ONE forward over the full trajectory ``input_ids_np`` (not a frame-by-
frame replay). The streaming BlockMask (``build_streaming_block_mask``) is
defined for arbitrary (q, kv) position pairs, so we just feed it q=arange(N)
and kv=arange(N), valid_kv_len=N. The mask predicate is the same one rollout
used, so the per-position attention pattern is identical.

This is much faster than replaying frame-by-frame (one forward vs N forwards)
and produces logits at every position. We then extract logp at the
``sampled_positions`` from the trajectory.

API
---
    compute_logps_for_trajectory(
        model,
        input_ids_np,             # state.input_ids_np from rollout
        sampled_positions,        # frame_callback's sampled_positions
        sampled_tokens,           # frame_callback's sampled_tokens
        window_frames=30,
        assistant_role_id=74455,
        kernel_options=DEFAULT_KERNEL_OPTS,
    ) -> list[float]

Returns one logp per (position, token) pair. Caller can compare these
against the trajectory's ``sampled_logps`` to get the importance ratio.
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from typing import Sequence

from .inference import StreamingStepMaskState, reset_ctx, streaming_ctx_token

log = logging.getLogger(__name__)

DEFAULT_KERNEL_OPTS = {
    'BLOCK_M': 32,
    'BLOCK_N': 32,
    'FORCE_USE_FLEX_ATTENTION': True,
}


@torch.no_grad()
def compute_logps_for_trajectory(
    model,
    input_ids_np: np.ndarray,
    sampled_positions: Sequence[int],
    sampled_tokens: Sequence[int],
    *,
    window_frames: int = 30,
    assistant_role_id: int = 74455,
    kernel_options: dict | None = None,
    device: str | torch.device = 'cuda:0',
) -> list[float]:
    """Replay a trajectory through ``model`` and return per-action logp.

    Parameters
    ----------
    model
        Patched Qwen3.5 model (must already have ``patch_qwen3_5_for_step_mask``
        applied so its create_causal_mask returns our streaming BlockMask).
    input_ids_np
        Full token sequence from the rollout's ``state.input_ids_np``. May
        include both the prefix and all decoded action tokens.
    sampled_positions
        For each action token, the position it OCCUPIES in ``input_ids_np``
        (= the value in the rollout dump's ``sampled_positions[i]``). The
        logits used to predict this token live at position
        ``sampled_positions[i] - 1`` (causal LM convention).
    sampled_tokens
        The actual token ids that were sampled. We compute log P(this token
        | preceding context) at the corresponding logits position.
    window_frames
        Same value used at rollout (for the visibility mask).
    assistant_role_id
        Same value used at rollout.
    kernel_options
        Kernel options for flex_attention. Must match rollout (so the same
        compiled kernel is used).
    device
        Device for the forward.

    Returns
    -------
    List of floats, one per (position, token) pair. Index i corresponds to
    ``sampled_tokens[i]`` at ``sampled_positions[i]``.
    """
    if kernel_options is None:
        kernel_options = DEFAULT_KERNEL_OPTS
    device = torch.device(device)
    if len(sampled_positions) != len(sampled_tokens):
        raise ValueError(f'sampled_positions / sampled_tokens length mismatch: '
                         f'{len(sampled_positions)} vs {len(sampled_tokens)}')
    if not len(sampled_positions):
        return []

    # Rebuild a streaming state from the trajectory tokens so the mask_mod
    # predicate (segment / frame_id / cur_frame) lines up byte-for-byte with
    # what the rollout used.
    state = StreamingStepMaskState(
        window_frames=window_frames,
        assistant_role_id=assistant_role_id,
    )
    state.extend(input_ids_np)
    N = int(input_ids_np.shape[0])

    q_pos = torch.arange(N, device=device, dtype=torch.long)
    # kv_token_positions has the same length as q for full-trajectory forward.
    kv_pos = q_pos.clone()

    in_ids = torch.from_numpy(input_ids_np).to(device=device, dtype=torch.long)
    in_ids = in_ids.unsqueeze(0)  # (1, N)

    # Streaming ctx so patched create_causal_mask dispatches through
    # build_streaming_block_mask with the rollout-compatible predicate.
    # No precomputed_block_mask -> builder runs each time (we only do this
    # once per trajectory anyway).
    token = streaming_ctx_token(
        state=state,
        q_token_positions=q_pos,
        kv_token_positions=kv_pos,
        valid_kv_len=N,
    )
    try:
        out = model(
            input_ids=in_ids,
            use_cache=False,  # we want logits for every position, not a cache.
            cache_position=q_pos,
            kernel_options=kernel_options,
        )
    finally:
        reset_ctx(token)

    # logits shape: (1, N, vocab). logits at position p predict position p+1.
    logits = out.logits[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    logps = []
    for pos, tok in zip(sampled_positions, sampled_tokens):
        # The token at position ``pos`` is predicted by the logits at
        # position ``pos - 1`` (causal LM). Edge case: pos == 0 has no
        # predicting logits; we'd skip those, but trajectories always start
        # after a prefix so pos >= 1 in practice.
        logits_pos = pos - 1
        if logits_pos < 0 or logits_pos >= N:
            logps.append(float('nan'))
            continue
        logps.append(float(log_probs[logits_pos, tok].item()))
    return logps
