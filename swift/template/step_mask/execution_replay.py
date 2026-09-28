"""Shared token selection for inference-aligned training forwards."""

from __future__ import annotations

import torch
from dataclasses import dataclass

from .meta import build_step_mask_meta


@dataclass(frozen=True)
class CausalReplaySelection:
    """Selective-LM-head inputs whose rows retain causal-loss alignment."""

    logits_to_keep: torch.Tensor
    positions: torch.Tensor
    labels: torch.Tensor
    loss_scale: torch.Tensor | None


def physical_logits_positions(
    logits_to_keep: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Normalize a bool mask or index selector to unique physical positions."""
    if not isinstance(logits_to_keep, torch.Tensor) or logits_to_keep.ndim != 1:
        raise RuntimeError('execution replay requires 1D logits_to_keep')
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise RuntimeError('execution replay requires padding-free input_ids shaped [1, L], '
                           f'got {tuple(input_ids.shape)}')

    length = int(input_ids.shape[1])
    selector = logits_to_keep.detach()
    if selector.dtype == torch.bool:
        if int(selector.numel()) != length:
            raise RuntimeError('boolean logits_to_keep must cover the physical sequence: '
                               f'{selector.numel()} != {length}')
        positions = torch.nonzero(selector, as_tuple=False).flatten()
    elif selector.dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
    ):
        positions = selector.to(dtype=torch.long)
    else:
        raise RuntimeError('logits_to_keep must be a bool mask or integer positions, got '
                           f'{selector.dtype}')

    if positions.numel() == 0:
        raise RuntimeError('execution replay selected no LM-head positions')
    if bool(((positions < 0) | (positions >= length)).any().item()):
        raise RuntimeError(f'logits_to_keep contains a position outside [0, {length})')
    if int(torch.unique(positions).numel()) != int(positions.numel()):
        raise RuntimeError('logits_to_keep contains duplicate positions')
    return positions.to(device=input_ids.device)


def expand_dual_branch_companions(
    input_ids: torch.Tensor,
    required_positions: torch.Tensor,
    *,
    assistant_role_id: int,
) -> torch.Tensor:
    """Include every A/B row sharing a synchronous production LM-head call."""
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise RuntimeError('dual execution replay requires physical input_ids shaped [1, L]')
    length = int(input_ids.shape[1])
    required = {int(position) for position in required_positions.detach().cpu().flatten().tolist()}
    if any(position < 0 or position >= length for position in required):
        raise RuntimeError('required LM-head position lies outside input_ids')

    meta = build_step_mask_meta(
        input_ids[0].detach().cpu().numpy(),
        assistant_role_id=assistant_role_id,
        include_branch_source=True,
    )
    if not meta.is_dual_branch or len(meta.b_spans) != len(meta.a_spans):
        raise RuntimeError('execution replay requires complete dual turns')

    expanded: set[int] = set()
    branch_positions: set[int] = set()
    for (b_start, b_end), (a_start, a_end) in zip(meta.b_spans, meta.a_spans):
        max_length = max(b_end - b_start, a_end - a_start)
        for offset in range(max_length):
            group = []
            if b_start + offset < b_end:
                group.append(b_start + offset)
            if a_start + offset < a_end:
                group.append(a_start + offset)
            branch_positions.update(group)
            if required.intersection(group):
                expanded.update(group)

    outside = sorted(required - branch_positions)
    if outside:
        raise RuntimeError('supervised logits outside dual branches cannot reproduce the '
                           f'production decode grouping: {outside[:16]}')
    if not required.issubset(expanded):
        raise RuntimeError('dual companion expansion dropped required logits')
    return torch.tensor(sorted(expanded), device=input_ids.device, dtype=torch.long)


def build_inference_aligned_causal_selection(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_scale: torch.Tensor | None,
    *,
    assistant_role_id: int,
) -> CausalReplaySelection:
    """Prepare SFT loss rows while preserving production A/B GEMM grouping.

    Swift's causal loss rolls labels left by one row.  Companion rows that do
    not carry loss therefore have to be inserted into both the selective
    LM-head mask and the filtered labels.  A final ignored row keeps the first
    filtered label at ``-100``, matching the generic trainer's denominator and
    metric assumptions.
    """
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise RuntimeError('inference-aligned SFT requires padding-free input_ids [1, L]')
    if labels.shape != input_ids.shape:
        raise RuntimeError(f'labels shape {tuple(labels.shape)} != input_ids '
                           f'{tuple(input_ids.shape)}')
    if loss_scale is not None and loss_scale.shape != labels.shape:
        raise RuntimeError(f'loss_scale shape {tuple(loss_scale.shape)} != labels '
                           f'{tuple(labels.shape)}')

    length = int(input_ids.shape[1])
    target_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    if target_positions.numel() == 0:
        raise RuntimeError('inference-aligned SFT batch has no targets')
    if bool((target_positions == 0).any().item()):
        raise RuntimeError('a sequence-start token cannot be a causal target')
    required = target_positions - 1
    positions = expand_dual_branch_companions(
        input_ids,
        required,
        assistant_role_id=assistant_role_id,
    )

    # The ignored final row preserves Swift's conventional filtered layout:
    # labels[0] == -100 and roll(labels, -1)[i] supervises output row i.
    dummy = torch.tensor([length - 1], device=input_ids.device)
    positions = torch.unique(torch.cat([positions, dummy]), sorted=True)
    next_positions = positions + 1
    valid_next = next_positions < length
    aligned_targets = torch.full(
        (positions.numel(), ),
        -100,
        device=labels.device,
        dtype=labels.dtype,
    )
    aligned_targets[valid_next] = labels[0].index_select(0, next_positions[valid_next])
    if int((aligned_targets != -100).sum().item()) != int((labels != -100).sum().item()):
        raise RuntimeError('execution-aligned selection changed the supervised target count')
    filtered_labels = torch.roll(aligned_targets, shifts=1).unsqueeze(0)
    if int(filtered_labels[0, 0].item()) != -100:
        raise RuntimeError('execution-aligned causal labels lack leading ignore')

    filtered_scale = None
    if loss_scale is not None:
        aligned_scale = torch.zeros(
            positions.numel(),
            device=loss_scale.device,
            dtype=loss_scale.dtype,
        )
        aligned_scale[valid_next] = loss_scale[0].index_select(0, next_positions[valid_next])
        aligned_scale[aligned_targets == -100] = 0
        filtered_scale = torch.roll(aligned_scale, shifts=1).unsqueeze(0)

    mask = torch.zeros(length, device=input_ids.device, dtype=torch.bool)
    mask[positions] = True
    return CausalReplaySelection(
        logits_to_keep=mask,
        positions=positions,
        labels=filtered_labels,
        loss_scale=filtered_scale,
    )
