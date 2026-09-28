"""Naive numpy reference implementation of the step+window visibility matrix.

This is for verification only -- the production path will use
``flex_attention`` with an equivalent ``mask_mod`` (Step C). Keeping a slow but
obviously-correct version here lets us unit-test the predicate and compare
against the kernel implementation.

Dual-branch extension: same base predicate, plus a cross-branch same-frame
block so A tokens cannot attend to B tokens (and vice versa) when they share
a frame. Cross-frame A/B visibility (through text_key or window) is preserved.
"""

from __future__ import annotations

import numpy as np

from .meta import StepMaskMeta


def build_visibility_matrix_numpy(meta: StepMaskMeta, window_frames: int) -> np.ndarray:
    """Return a [L, L] boolean matrix M where M[i, j] is True iff query i is
    allowed to attend to key j under the step+window (+ dual-branch) rule.

    Base visibility rule (single-branch and dual-branch alike):
        M[i, j] = (j <= i)                           # causal
                  AND ( NOT is_visual[j]              # text is always visible
                        OR ( same_segment(j, i)       # no step close in [j+1, i]
                             AND frame_id(i) - frame_id(j) < W ) )
        where frame_id(i) for a non-visual i is the most recent visual frame's id
        seen at or before i (running max).

    Dual-branch additional predicate (applied iff meta.is_dual_branch):
        AND NOT ( same_cur_frame(i, j)
                  AND ( (route_A[i] AND route_B[j])
                        OR (route_B[i] AND route_A[j]) ) )
    """
    L = meta.is_visual.shape[0]

    # segment id at each position: number of step closes strictly before that position.
    seg_id = np.cumsum(meta.step_close.astype(np.int32)) - meta.step_close.astype(np.int32)

    # cur_frame[i] = running max of frame_id (visual frames carry their id,
    # text positions carry the most recent frame's id). Positions before the
    # first frame get -1.
    cur_frame = np.empty(L, dtype=np.int32)
    last = -1
    for i in range(L):
        if meta.frame_id[i] >= 0:
            last = int(meta.frame_id[i])
        cur_frame[i] = last

    # Build the matrix. O(L^2) memory; only used for tiny debug samples.
    causal = np.tril(np.ones((L, L), dtype=bool))
    text_key = ~meta.is_visual[None, :].repeat(L, axis=0)  # M[i,j] depends only on j
    same_seg = seg_id[:, None] == seg_id[None, :]  # M[i,j]
    fid_j = meta.frame_id[None, :]  # [1, L]
    fid_i = cur_frame[:, None]  # [L, 1]
    in_window = (fid_i - fid_j) < window_frames  # broadcasts to [L, L]

    visual_ok = same_seg & in_window
    visible = causal & (text_key | visual_ok)

    if meta.is_dual_branch:
        # Cross-branch same-frame block.
        same_cur_frame = cur_frame[:, None] == cur_frame[None, :]
        q_A = meta.route_A[:, None]  # [L, 1]
        q_B = meta.route_B[:, None]
        k_A = meta.route_A[None, :]  # [1, L]
        k_B = meta.route_B[None, :]
        cross_conflict = same_cur_frame & ((q_A & k_B) | (q_B & k_A))
        visible = visible & ~cross_conflict

    return visible
