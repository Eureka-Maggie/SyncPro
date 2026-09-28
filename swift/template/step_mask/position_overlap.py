"""Position ID overlap for dual-branch training/inference parity.

Purpose
-------
In dual-branch data the sequence contains:

    ... [image_t] [prefix] [B tokens (lb)] [A tokens (la)] [close_\\n] [image_{t+1}] ...

Under the standard sequential ``position_ids``:
    B tokens sit at positions P..P+lb-1
    A tokens sit at positions P+lb..P+lb+la-1
    image_{t+1}[0] sits at P+lb+la+K   (K = shared inter-frame tokens = 4)

At inference we do parallel decode: both B and A start decoding from the same
position (right after the shared prefix). So A tokens' inference positions
are P..P+la-1, not P+lb..P+lb+la-1. RoPE angles differ -> train/infer
mismatch.

Fix: rewrite training-time position_ids so A span overlaps B span:
    B tokens stay at P..P+lb-1
    A tokens move to P..P+la-1   (SAME start as B)
    close_\\n moves to P + max(lb, la)
    image_{t+1}[0] moves to P + max(lb, la) + K

The cross-branch same-frame mask already blocks B<->A attention within a
frame, so having identical position_ids for B and A tokens is safe (they
never attend to each other and never mix K/V into each other's states).

Activation: ``MSS_POSITION_ID_OVERLAP=1`` env var. Default off preserves
current sequential behavior.
"""

from __future__ import annotations

import numpy as np
import os
import torch
from typing import List, Optional, Sequence, Tuple


def env_enabled() -> bool:
    return os.environ.get('MSS_POSITION_ID_OVERLAP', '').strip() not in ('', '0', 'false', 'False')


Span = Tuple[int, int]


def _validate_sample_spans(sample_spans: Sequence[Span], length: int) -> List[Span]:
    spans = [tuple(map(int, span)) for span in sample_spans]
    if not spans:
        raise ValueError('sample_spans must not be empty')
    if spans[0][0] != 0 or spans[-1][1] != length:
        raise ValueError(f'sample_spans must cover [0, {length}), got {spans}')
    if any(start >= end for start, end in spans):
        raise ValueError(f'sample_spans must be non-empty, got {spans}')
    if any(left[1] != right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError(f'sample_spans must be contiguous, got {spans}')
    return spans


def _sample_spans_from_numpy_positions(position_ids: np.ndarray) -> List[Span]:
    """Best-effort fallback for legacy callers without packed boundaries."""
    length = int(position_ids.size)
    if length == 0:
        return []
    is_zero = position_ids == 0
    zero_run_start = is_zero.copy()
    zero_run_start[1:] &= ~is_zero[:-1]
    starts = np.flatnonzero(zero_run_start).tolist()
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    return list(zip(starts, starts[1:] + [length]))


def _sample_spans_from_torch_positions(position_ids: torch.Tensor) -> List[Span]:
    """Best-effort fallback for legacy callers without packed boundaries."""
    length = int(position_ids.numel())
    if length == 0:
        return []
    is_zero = position_ids == 0
    zero_run_start = is_zero.clone()
    zero_run_start[1:] &= ~is_zero[:-1]
    starts = torch.nonzero(zero_run_start, as_tuple=True)[0].tolist()
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    return list(zip(starts, starts[1:] + [length]))


def _branch_layout(
    length: int,
    b_spans: Sequence[Span],
    a_spans: Sequence[Span],
    sample_spans: Sequence[Span],
) -> List[Tuple[Span, Span, int]]:
    """Validate the physical fork layout and return each pair's sample end."""
    if len(b_spans) != len(a_spans):
        raise ValueError(f'len mismatch: b_spans={len(b_spans)}, a_spans={len(a_spans)}')
    samples = _validate_sample_spans(sample_spans, length)
    layout = []
    sample_idx = 0
    previous_a_end = 0
    for branch_idx, (b_span, a_span) in enumerate(zip(b_spans, a_spans)):
        b_start, b_end = map(int, b_span)
        a_start, a_end = map(int, a_span)
        if not (0 <= b_start < b_end == a_start < a_end <= length):
            raise ValueError(f'invalid branch pair #{branch_idx}: B={(b_start, b_end)} '
                             f'A={(a_start, a_end)} seq_len={length}')
        if b_start < previous_a_end:
            raise ValueError(f'branch spans overlap or are out of order at #{branch_idx}')
        previous_a_end = a_end

        while sample_idx + 1 < len(samples) and b_start >= samples[sample_idx][1]:
            sample_idx += 1
        sample_start, sample_end = samples[sample_idx]
        if b_start < sample_start or a_end > sample_end:
            raise ValueError(f'branch pair #{branch_idx} crosses packed-sample boundary '
                             f'[{sample_start}, {sample_end}): B={(b_start, b_end)} '
                             f'A={(a_start, a_end)}')
        layout.append(((b_start, b_end), (a_start, a_end), sample_end))
    return layout


def _overlap_shift_numpy(
    reference_positions: np.ndarray,
    b_spans: Sequence[Span],
    a_spans: Sequence[Span],
    sample_spans: Optional[Sequence[Span]],
) -> np.ndarray:
    length = int(reference_positions.size)
    samples = (_sample_spans_from_numpy_positions(reference_positions) if sample_spans is None else list(sample_spans))
    shift = np.zeros(length, dtype=reference_positions.dtype)
    for (b_start, b_end), (a_start, a_end), sample_end in _branch_layout(length, b_spans, a_spans, samples):
        len_b = b_end - b_start
        len_a = a_end - a_start
        shift[a_start:a_end] += len_b
        shift[a_end:sample_end] += min(len_b, len_a)
    return shift


def _overlap_shift_torch(
    reference_positions: torch.Tensor,
    b_spans: Sequence[Span],
    a_spans: Sequence[Span],
    sample_spans: Optional[Sequence[Span]],
) -> torch.Tensor:
    length = int(reference_positions.numel())
    samples = (_sample_spans_from_torch_positions(reference_positions) if sample_spans is None else list(sample_spans))
    shift = torch.zeros_like(reference_positions)
    for (b_start, b_end), (a_start, a_end), sample_end in _branch_layout(length, b_spans, a_spans, samples):
        len_b = b_end - b_start
        len_a = a_end - a_start
        shift[a_start:a_end] += len_b
        shift[a_end:sample_end] += min(len_b, len_a)
    return shift


def apply_overlap_1d(
    temporal_pos_ids: np.ndarray,
    b_spans: List[Tuple[int, int]],
    a_spans: List[Tuple[int, int]],
    sample_spans: Optional[Sequence[Span]] = None,
) -> None:
    """In-place: rewrite ``temporal_pos_ids`` (1D int array) to apply the
    dual-branch position overlap for every (b_span, a_span) pair.

    ``sample_spans`` should be supplied for padding-free batches. A legacy
    position-reset fallback is retained for inference verification utilities.
    """
    if temporal_pos_ids.ndim != 1:
        raise ValueError(f'expected 1D position ids, got shape {temporal_pos_ids.shape}')
    if not b_spans and not a_spans:
        return
    temporal_pos_ids -= _overlap_shift_numpy(temporal_pos_ids, b_spans, a_spans, sample_spans)


def apply_overlap_torch_1d(
    temporal_pos_ids: torch.Tensor,
    b_spans: List[Tuple[int, int]],
    a_spans: List[Tuple[int, int]],
    sample_spans: Optional[Sequence[Span]] = None,
) -> None:
    """Pure-torch in-place version of :func:`apply_overlap_1d`.

    Avoids a numpy round-trip. Production padding-free callers pass explicit
    sample spans, so no position tensor needs to be synchronized to the host.
    """
    if temporal_pos_ids.dim() != 1:
        raise ValueError(f'expected 1D position ids, got shape {temporal_pos_ids.shape}')
    if not b_spans and not a_spans:
        return
    temporal_pos_ids -= _overlap_shift_torch(temporal_pos_ids, b_spans, a_spans, sample_spans)


def apply_overlap_to_position_ids_tensor(
    position_ids: torch.Tensor,
    b_spans: List[Tuple[int, int]],
    a_spans: List[Tuple[int, int]],
    sample_spans: Optional[Sequence[Span]] = None,
) -> torch.Tensor:
    """Return cloned position IDs with the dual-branch timeline applied.

    The same scalar overlap shift is subtracted from all three Qwen3.5 MRoPE
    axes. This keeps text T/H/W equal and preserves native visual-grid offsets.
    The optional leading physical text-position row is never modified.

    Handles Qwen3.5's MRoPE:
        3D shape [3, B, L]: model-facing (time, height, width) MRoPE.
        3D shape [4, B, L]: Swift collator layout with physical text positions
                            in row 0 and model MRoPE in rows 1:4.
        2D shape [B, L]  : treat as temporal-only.
        1D shape [L]     : treat as temporal-only.

    Runs entirely on the tensor's device (no numpy round-trip).
    """
    if position_ids is None:
        return position_ids
    p = position_ids.clone()
    if not b_spans and not a_spans:
        return p
    if p.dim() == 3 and p.shape[0] in (3, 4):
        mrope_start = 0 if p.shape[0] == 3 else 1
        for b_idx in range(p.size(1)):
            reference = p[0, b_idx]
            shift = _overlap_shift_torch(reference, b_spans, a_spans, sample_spans)
            p[mrope_start:, b_idx] -= shift.unsqueeze(0)
    elif p.dim() == 2:
        for b_idx in range(p.size(0)):
            apply_overlap_torch_1d(p[b_idx], b_spans, a_spans, sample_spans=sample_spans)
    elif p.dim() == 1:
        apply_overlap_torch_1d(p, b_spans, a_spans, sample_spans=sample_spans)
    else:
        raise ValueError(f'unexpected position_ids shape: {tuple(p.shape)}')
    return p
