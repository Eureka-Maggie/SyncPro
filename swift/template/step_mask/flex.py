"""flex_attention BlockMask construction for the step+window visibility rule.

The rule is the same as in :func:`swift.template.step_mask.mask.build_visibility_matrix_numpy`,
but expressed as a ``mask_mod`` closure suitable for ``torch.nn.attention.flex_attention``.

Padding-free training packs multiple samples into a single sequence of length L
with bs=1 and ``position_ids`` that restart at 0 at each sample boundary; we
derive ``sample_id`` from that so the mask is block-diagonal across samples.

Dual-branch extension: when the meta detects dual-branch data (each assistant
turn contains a B (inner) span and an A (speak) span separated by a mid
``<|im_end|>``), an additional predicate is applied so that within the SAME
frame, A tokens cannot attend to B tokens and vice versa. Cross-frame
visibility of A/B tokens is unaffected -- future frames can still read
previous frames' inner and speak texts as persistent memory.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask

from .meta import build_step_mask_meta


def _meta_to_tensors(input_ids_1d, *, labels_1d=None, assistant_role_id=None, device):
    """Run the numpy metadata extractor and move derived arrays onto device.

    Returns a dict with all tensors needed by the mask_mod closure. In
    dual-branch mode also carries route_A, route_B and per-token frame ids
    (with a carry-forward variant ``turn_frame`` used for same-frame checks).
    """
    meta = build_step_mask_meta(input_ids_1d, labels=labels_1d, assistant_role_id=assistant_role_id)
    # seg_id[t] = number of step closes at positions < t.
    seg_id_np = np.cumsum(meta.step_close.astype(np.int32)) - meta.step_close.astype(np.int32)
    # cur_frame[t] = running max of frame_id over positions <= t (carry-forward
    # so text positions inherit the most recent frame's id).
    cur_frame_np = np.maximum.accumulate(np.where(meta.frame_id >= 0, meta.frame_id, -1))

    out = {
        'is_visual': torch.from_numpy(meta.is_visual).to(device=device),
        'frame_id': torch.from_numpy(meta.frame_id).to(device=device, dtype=torch.int32),
        'seg_id': torch.from_numpy(seg_id_np).to(device=device, dtype=torch.int32),
        'cur_frame': torch.from_numpy(cur_frame_np).to(device=device, dtype=torch.int32),
        'is_dual_branch': meta.is_dual_branch,
    }
    if meta.is_dual_branch:
        # ``turn_frame`` = carry-forward frame id used specifically for the
        # cross-branch predicate. For A/B route tokens (which are inside an
        # assistant turn responding to a specific frame), this equals the
        # id of the frame that turn is responding to. Same as ``cur_frame``
        # for our data layout (frame image_t appears immediately before its
        # dual-branch assistant turn), so we reuse cur_frame.
        out['route_A'] = torch.from_numpy(meta.route_A).to(device=device)
        out['route_B'] = torch.from_numpy(meta.route_B).to(device=device)
    return out


def _sample_ids_from_position_ids(position_ids):
    """Detect sample boundaries in a padding_free packed sequence.

    ``position_ids`` is the time-axis position id (axis 0 of the 3D mrope
    position_ids that Qwen3.5 uses). A new sample starts at every position == 0,
    so cumsum of (pos == 0) - 1 gives a contiguous 0-indexed sample id.
    """
    # position_ids: [L] int
    starts = (position_ids == 0).to(torch.int32)
    sample_id = torch.cumsum(starts, dim=0) - 1
    return sample_id.to(torch.int32)


def _sample_ids_from_cu_seqlens(cu_seqlens, seq_len: int, device):
    """Build one logical-sample id per packed token from authoritative bounds."""
    cu = torch.as_tensor(cu_seqlens, device=device, dtype=torch.long).flatten()
    if cu.numel() < 2 or int(cu[0]) != 0 or int(cu[-1]) != int(seq_len):
        raise ValueError(f'invalid cu_seqlens for sequence length {seq_len}: {cu.tolist()}')
    lengths = cu[1:] - cu[:-1]
    if bool((lengths <= 0).any()):
        raise ValueError(f'cu_seqlens must be strictly increasing: {cu.tolist()}')
    return torch.repeat_interleave(
        torch.arange(lengths.numel(), device=device, dtype=torch.int32), lengths, output_size=seq_len)


def build_step_window_block_mask(
    input_ids,
    position_ids,
    window_frames: int,
    *,
    labels=None,
    assistant_role_id: int = None,
    num_heads: int = 1,
    cu_seqlens=None,
):
    """Build a flex_attention BlockMask for the step+window rule (+ dual-branch
    cross-branch predicate when applicable).

    Inputs
    ------
    input_ids   : LongTensor [B, L] (B is expected to be 1 in padding_free mode).
    position_ids: LongTensor [L]  — text position ids used as a legacy
                  sample-boundary fallback when cu_seqlens is unavailable.
    cu_seqlens  : IntTensor [N+1] — authoritative logical-sample boundaries for
                  padding-free packed batches.
    window_frames: int            — W, sliding-window length in frames.
    labels / assistant_role_id    — exactly one must be provided; see
                                    :func:`build_step_mask_meta`.

    Returns
    -------
    BlockMask suitable for the flex_attention attention interface.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, (
        f'step+window mask currently assumes packed bs=1 input; got {tuple(input_ids.shape)}')
    device = input_ids.device
    L = int(input_ids.shape[1])

    ids_np = input_ids[0].detach().cpu().numpy()
    lbl_np = labels[0].detach().cpu().numpy() if labels is not None else None
    tensors = _meta_to_tensors(ids_np, labels_1d=lbl_np, assistant_role_id=assistant_role_id, device=device)

    is_visual = tensors['is_visual']
    frame_id = tensors['frame_id']
    seg_id = tensors['seg_id']
    cur_frame = tensors['cur_frame']
    is_dual = tensors['is_dual_branch']
    route_A = tensors.get('route_A')
    route_B = tensors.get('route_B')

    if cu_seqlens is not None:
        sample_id = _sample_ids_from_cu_seqlens(cu_seqlens, L, device)
    else:
        if position_ids is None:
            raise ValueError('position_ids or cu_seqlens is required')
        boundary_position_ids = position_ids
        if boundary_position_ids.dim() == 3 and boundary_position_ids.shape[0] in (3, 4):
            boundary_position_ids = boundary_position_ids[0]
        sample_id = _sample_ids_from_position_ids(boundary_position_ids.flatten())
        if sample_id.numel() != L:
            raise ValueError('multi-axis MRoPE position_ids cannot identify packed sample '
                             'boundaries; pass cu_seqlens')
    W = int(window_frames)

    if is_dual:
        # Dual-branch predicate: same rule as single-branch, plus block
        # A <-> B cross-branch attention within the same frame.
        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            same_sample = sample_id[q_idx] == sample_id[kv_idx]
            text_key = ~is_visual[kv_idx]
            same_seg = seg_id[q_idx] == seg_id[kv_idx]
            in_window = (cur_frame[q_idx] - frame_id[kv_idx]) < W
            visual_ok = same_seg & in_window
            base_visible = causal & same_sample & (text_key | visual_ok)

            # Cross-branch same-frame block: reject any (q, kv) pair where
            # both endpoints are in {A, B} route spans, they belong to
            # different branches, and they belong to the same frame (their
            # ``cur_frame`` values match).
            q_A, q_B = route_A[q_idx], route_B[q_idx]
            k_A, k_B = route_A[kv_idx], route_B[kv_idx]
            same_frame = cur_frame[q_idx] == cur_frame[kv_idx]
            cross_conflict = same_frame & ((q_A & k_B) | (q_B & k_A))
            return base_visible & ~cross_conflict
    else:

        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            same_sample = sample_id[q_idx] == sample_id[kv_idx]
            text_key = ~is_visual[kv_idx]
            same_seg = seg_id[q_idx] == seg_id[kv_idx]
            in_window = (cur_frame[q_idx] - frame_id[kv_idx]) < W
            visual_ok = same_seg & in_window
            return causal & same_sample & (text_key | visual_ok)

    return create_block_mask(
        mask_mod,
        B=1,
        H=num_heads,
        Q_LEN=L,
        KV_LEN=L,
        device=device,
        _compile=True,
    )
