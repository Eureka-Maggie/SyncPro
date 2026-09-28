"""Pure-CPU contracts for post-rollout dual-GRPO actor rebalancing.

This module deliberately has no dependency on the trainer, torch, PIL, or
distributed collectives.  The trainer may use it after rollout/reward/
advantage computation to exchange a compact description of complete query
groups, derive one deterministic actor assignment, and flatten the groups in
the same group-major order used by rollout.

Images and frozen visual features are intentionally absent from the compact
metadata payload.  A receiving rank reconstructs PIL inputs from authoritative
case metadata and recomputes frozen visual features.  The payload retains
finite scalar reward components for self-consistency, but rollout-owned
reporting/fingerprints must continue to use the original local trajectories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from examples.rl.dual_grpo_schedule import rank_actor_prompt_slot_plan


ACTOR_RANK_REBALANCE_NONE = 'none'
ACTOR_RANK_REBALANCE_POST_ROLLOUT = 'post_rollout_groups_v1'
ACTOR_RANK_REBALANCE_CHOICES = (
    ACTOR_RANK_REBALANCE_NONE,
    ACTOR_RANK_REBALANCE_POST_ROLLOUT,
)

_ACTOR_FRAME_FIELDS = tuple(
    f'{route}_{suffix}'
    for route in ('B', 'A')
    for suffix in ('tokens', 'token_positions', 'token_logps')
)
_ACTOR_TRAJECTORY_FIELDS = frozenset({
    'prompt_slot',
    'group_index',
    'case_id',
    'messages',
    'frames',
    'physical_ids',
    'seed',
    'reward',
    'advantage',
})
_ACTOR_GROUP_FIELDS = frozenset({
    'prompt_slot',
    'case_id',
    'sampled_decision_cost',
    'trajectories',
})
_ACTOR_GROUP_COST_FIELDS = frozenset({
    'prompt_slot',
    'case_id',
    'sampled_decision_cost',
})


def actor_rebalance_enabled(mode: str, world_size: int) -> bool:
    """Return whether the requested mode should exchange groups.

    A valid non-default mode is still an exact no-op for one process.
    """
    if mode not in ACTOR_RANK_REBALANCE_CHOICES:
        raise ValueError(f'unsupported actor rank rebalance mode: {mode!r}')
    world_size = _strict_int(world_size, name='world_size', minimum=1)
    return mode == ACTOR_RANK_REBALANCE_POST_ROLLOUT and world_size > 1


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def _cpu_copy(value: Any, *, path: str) -> Any:
    """Copy a compact payload value while rejecting opaque/GPU objects."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f'{path} contains an object-dtype array')
        return value.copy()
    if isinstance(value, Mapping):
        return {
            str(key): _cpu_copy(item, path=f'{path}.{key}')
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_cpu_copy(item, path=f'{path}[{index}]') for index, item in enumerate(value)]
    raise TypeError(f'{path} contains unsupported {type(value).__name__}; '
                    'compact actor payloads may contain CPU primitives/NumPy only')


def _compact_frame(frame: Mapping[str, Any], *, frame_index: int) -> dict[str, list]:
    if not isinstance(frame, Mapping):
        raise TypeError(f'frames[{frame_index}] must be a mapping')
    compact = {}
    for field in _ACTOR_FRAME_FIELDS:
        if field not in frame:
            raise ValueError(f'frames[{frame_index}] lacks {field}')
        value = _cpu_copy(frame[field], path=f'frames[{frame_index}].{field}')
        if not isinstance(value, list):
            if isinstance(value, np.ndarray) and value.ndim == 1:
                value = value.tolist()
            else:
                raise TypeError(f'frames[{frame_index}].{field} must be one-dimensional')
        compact[field] = value

    positions = []
    for route in ('B', 'A'):
        tokens = compact[f'{route}_tokens']
        route_positions = compact[f'{route}_token_positions']
        logps = compact[f'{route}_token_logps']
        if not (len(tokens) == len(route_positions) == len(logps)):
            raise ValueError(f'frames[{frame_index}] {route} token metadata length mismatch')
        compact[f'{route}_tokens'] = [
            _strict_int(token, name=f'frames[{frame_index}].{route}_tokens', minimum=0)
            for token in tokens
        ]
        route_positions = [
            _strict_int(position,
                        name=f'frames[{frame_index}].{route}_token_positions',
                        minimum=1) for position in route_positions
        ]
        route_logps = [float(logp) for logp in logps]
        if not np.isfinite(route_logps).all():
            raise ValueError(f'frames[{frame_index}] {route} contains non-finite logp')
        compact[f'{route}_token_positions'] = route_positions
        compact[f'{route}_token_logps'] = route_logps
        positions.extend(route_positions)
    if len(set(positions)) != len(positions):
        raise ValueError(f'frames[{frame_index}] contains duplicate sampled positions')
    return compact


def _compact_reward(reward: Mapping[str, Any]) -> dict[str, int | float]:
    compact = {}
    for key, value in reward.items():
        if not isinstance(key, str):
            raise TypeError('trajectory.reward keys must be strings')
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise TypeError(f'trajectory.reward.{key} must be a numeric scalar')
        value = value.item() if isinstance(value, np.generic) else value
        if not np.isfinite(float(value)):
            raise ValueError(f'trajectory.reward.{key} must be finite')
        compact[key] = value
    return compact


def compact_actor_trajectory(trajectory: Any, advantage: float) -> dict:
    """Strip one rollout trajectory to the exact metadata actor replay needs."""
    case = getattr(trajectory, 'case', None)
    case_id = getattr(case, 'row_index', None)
    if case_id is None:
        raise TypeError('trajectory.case.row_index is required')
    reward = getattr(trajectory, 'reward', None)
    if not isinstance(reward, Mapping) or 'total' not in reward:
        raise ValueError('trajectory.reward.total is required before actor rebalance')
    reward = _compact_reward(reward)
    reward_total = float(reward['total'])
    advantage = float(advantage)
    if not np.isfinite((reward_total, advantage)).all():
        raise ValueError('reward and advantage must be finite')

    physical_ids = np.asarray(getattr(trajectory, 'physical_ids', None))
    if physical_ids.ndim != 1 or not np.issubdtype(physical_ids.dtype, np.integer):
        raise TypeError('trajectory.physical_ids must be a one-dimensional integer array')
    frames = getattr(trajectory, 'frames', None)
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or not frames:
        raise ValueError('trajectory.frames must be a non-empty sequence')
    compact_frames = [
        _compact_frame(frame, frame_index=index) for index, frame in enumerate(frames)
    ]
    sampled_positions = []
    for frame in compact_frames:
        for route in ('B', 'A'):
            for token, position in zip(
                    frame[f'{route}_tokens'], frame[f'{route}_token_positions']):
                if position >= len(physical_ids) or int(physical_ids[position]) != token:
                    raise ValueError(f'{route} sampled token does not match physical_ids '
                                     f'at position {position}')
                sampled_positions.append(position)
    if len(set(sampled_positions)) != len(sampled_positions):
        raise ValueError('trajectory contains duplicate sampled positions')

    messages = _cpu_copy(getattr(trajectory, 'messages', None), path='messages')
    if not isinstance(messages, list) or not messages:
        raise TypeError('trajectory.messages must be a non-empty list')
    return {
        'prompt_slot': _strict_int(getattr(trajectory, 'prompt_slot', None), name='prompt_slot'),
        'group_index': _strict_int(getattr(trajectory, 'group_index', None), name='group_index'),
        'case_id': _strict_int(case_id, name='case_id'),
        'messages': messages,
        'frames': compact_frames,
        'physical_ids': physical_ids.astype(np.int64, copy=True),
        'seed': _strict_int(getattr(trajectory, 'seed', None), name='seed'),
        # Actor replay currently reads only total.  Keeping the complete
        # finite scalar reward is cheap and prevents future reporting code
        # from silently losing component/window metrics.
        'reward': reward,
        'advantage': advantage,
    }


def sampled_decision_count(compact_trajectory: Mapping[str, Any]) -> int:
    """Return the actor cost proxy from validated compact frame metadata."""
    frames = compact_trajectory.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('compact trajectory has no frames')
    count = sum(
        len(frame[f'{route}_tokens'])
        for frame in frames
        for route in ('B', 'A')
    )
    if count < 1:
        raise ValueError('compact trajectory contains no sampled decisions')
    return count


def _assign_actor_shards_with_locality(
    shards: Sequence[Sequence[int]],
    source_rank_by_prompt_slot: Mapping[int, int],
) -> list[list[int]]:
    """Map balanced shards to ranks while greedily retaining local queries.

    Reordering whole shards cannot change their loads.  Stable shard-order
    tie breaks keep the assignment deterministic and avoid exchanging visual
    inputs when an equally balanced local placement exists.
    """
    remaining = [(index, list(shard)) for index, shard in enumerate(shards)]
    localized = []
    for target_rank in range(len(shards)):
        chosen_index, (_, chosen_shard) = min(
            enumerate(remaining),
            key=lambda item: (
                -sum(source_rank_by_prompt_slot[slot] == target_rank
                     for slot in item[1][1]),
                item[1][0],
            ),
        )
        localized.append(chosen_shard)
        del remaining[chosen_index]
    return localized


def compact_actor_groups(
    trajectories: Sequence[Any],
    advantages: Sequence[float],
    *,
    prompt_slots: Sequence[int],
    group_size: int,
) -> list[dict]:
    """Build complete-G, query-owned payloads for one rollout rank."""
    group_size = _strict_int(group_size, name='group_size', minimum=1)
    prompt_slots = [_strict_int(slot, name='prompt_slot') for slot in prompt_slots]
    if len(set(prompt_slots)) != len(prompt_slots):
        raise ValueError('prompt_slots must be distinct')
    if len(trajectories) != len(advantages):
        raise ValueError('trajectory and advantage counts differ')

    rows = [
        compact_actor_trajectory(trajectory, advantage)
        for trajectory, advantage in zip(trajectories, advantages)
    ]
    expected = {
        (prompt_slot, group_index)
        for prompt_slot in prompt_slots
        for group_index in range(group_size)
    }
    actual = [(row['prompt_slot'], row['group_index']) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('local actor payload must cover every (prompt_slot, group_index) once')

    groups = []
    for prompt_slot in sorted(prompt_slots):
        group_rows = sorted(
            (row for row in rows if row['prompt_slot'] == prompt_slot),
            key=lambda row: row['group_index'],
        )
        case_ids = {row['case_id'] for row in group_rows}
        if len(case_ids) != 1:
            raise ValueError(f'prompt slot {prompt_slot} contains multiple case ids')
        groups.append({
            'prompt_slot': prompt_slot,
            'case_id': next(iter(case_ids)),
            'sampled_decision_cost': sum(sampled_decision_count(row) for row in group_rows),
            'trajectories': group_rows,
        })
    return groups


def plan_gathered_actor_groups(
    gathered_rank_groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    world_size: int,
    local_query_capacity: int,
    group_size: int,
) -> dict:
    """Validate all gathered groups and derive a deterministic rank plan.

    This is intentionally a pure helper: a two-rank test can feed two local
    Python payload lists without initializing torch.distributed.
    """
    world_size = _strict_int(world_size, name='world_size', minimum=1)
    local_query_capacity = _strict_int(
        local_query_capacity, name='local_query_capacity', minimum=1)
    group_size = _strict_int(group_size, name='group_size', minimum=1)
    if len(gathered_rank_groups) != world_size:
        raise ValueError('gathered rank payload count differs from world_size')

    groups_by_prompt_slot = {}
    source_rank_by_prompt_slot = {}
    for source_rank, rank_groups in enumerate(gathered_rank_groups):
        if not isinstance(rank_groups, Sequence) or isinstance(rank_groups, (str, bytes)):
            raise TypeError(f'rank {source_rank} payload must be a sequence')
        if len(rank_groups) != local_query_capacity:
            raise ValueError(f'rank {source_rank} group count differs from local capacity')
        for group in rank_groups:
            if not isinstance(group, Mapping):
                raise TypeError(f'rank {source_rank} contains a non-mapping group')
            if set(group) != _ACTOR_GROUP_FIELDS:
                raise ValueError(f'rank {source_rank} actor group fields differ from compact contract')
            prompt_slot = _strict_int(group.get('prompt_slot'), name='prompt_slot')
            case_id = _strict_int(group.get('case_id'), name='case_id')
            if prompt_slot in groups_by_prompt_slot:
                raise ValueError(f'duplicate global prompt slot {prompt_slot}')
            rows = group.get('trajectories')
            if not isinstance(rows, list) or len(rows) != group_size:
                raise ValueError(f'prompt slot {prompt_slot} is not a complete G group')
            indices = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError(f'prompt slot {prompt_slot} contains an invalid trajectory')
                if set(row) != _ACTOR_TRAJECTORY_FIELDS:
                    raise ValueError(f'prompt slot {prompt_slot} trajectory fields differ '
                                     'from compact contract')
                if _strict_int(row.get('prompt_slot'), name='trajectory.prompt_slot') != prompt_slot:
                    raise ValueError(f'prompt slot {prompt_slot} contains a foreign trajectory')
                if _strict_int(row.get('case_id'), name='trajectory.case_id') != case_id:
                    raise ValueError(f'prompt slot {prompt_slot} contains a foreign case')
                indices.append(_strict_int(row.get('group_index'), name='group_index'))
            if sorted(indices) != list(range(group_size)):
                raise ValueError(f'prompt slot {prompt_slot} does not cover group indices 0..G-1')
            cost = _strict_int(
                group.get('sampled_decision_cost'),
                name='sampled_decision_cost',
                minimum=1,
            )
            recomputed_cost = sum(sampled_decision_count(row) for row in rows)
            if cost != recomputed_cost:
                raise ValueError(f'prompt slot {prompt_slot} decision cost mismatch: '
                                 f'{cost} != {recomputed_cost}')
            groups_by_prompt_slot[prompt_slot] = dict(group)
            source_rank_by_prompt_slot[prompt_slot] = source_rank

    return _plan_validated_actor_groups(
        groups_by_prompt_slot,
        source_rank_by_prompt_slot,
        world_size=world_size,
        local_query_capacity=local_query_capacity,
    )


def plan_gathered_actor_group_costs(
    gathered_rank_groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    world_size: int,
    local_query_capacity: int,
) -> dict:
    """Plan from small per-query costs before exchanging actor payloads."""
    world_size = _strict_int(world_size, name='world_size', minimum=1)
    local_query_capacity = _strict_int(
        local_query_capacity, name='local_query_capacity', minimum=1)
    if len(gathered_rank_groups) != world_size:
        raise ValueError('gathered rank cost count differs from world_size')

    groups_by_prompt_slot = {}
    source_rank_by_prompt_slot = {}
    for source_rank, rank_groups in enumerate(gathered_rank_groups):
        if not isinstance(rank_groups, Sequence) or isinstance(rank_groups, (str, bytes)):
            raise TypeError(f'rank {source_rank} cost payload must be a sequence')
        if len(rank_groups) != local_query_capacity:
            raise ValueError(f'rank {source_rank} cost count differs from local capacity')
        for group in rank_groups:
            if not isinstance(group, Mapping):
                raise TypeError(f'rank {source_rank} contains a non-mapping cost')
            if set(group) != _ACTOR_GROUP_COST_FIELDS:
                raise ValueError(f'rank {source_rank} actor cost fields differ from compact contract')
            prompt_slot = _strict_int(group.get('prompt_slot'), name='prompt_slot')
            if prompt_slot in groups_by_prompt_slot:
                raise ValueError(f'duplicate global prompt slot {prompt_slot}')
            groups_by_prompt_slot[prompt_slot] = {
                'prompt_slot': prompt_slot,
                'case_id': _strict_int(group.get('case_id'), name='case_id'),
                'sampled_decision_cost': _strict_int(
                    group.get('sampled_decision_cost'),
                    name='sampled_decision_cost',
                    minimum=1,
                ),
            }
            source_rank_by_prompt_slot[prompt_slot] = source_rank

    return _plan_validated_actor_groups(
        groups_by_prompt_slot,
        source_rank_by_prompt_slot,
        world_size=world_size,
        local_query_capacity=local_query_capacity,
    )


def _plan_validated_actor_groups(
    groups_by_prompt_slot: Mapping[int, Mapping[str, Any]],
    source_rank_by_prompt_slot: Mapping[int, int],
    *,
    world_size: int,
    local_query_capacity: int,
) -> dict:
    expected_slots = list(range(world_size * local_query_capacity))
    if sorted(groups_by_prompt_slot) != expected_slots:
        raise ValueError('gathered actor groups must cover contiguous global prompt slots')
    case_ids = [groups_by_prompt_slot[slot]['case_id'] for slot in expected_slots]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError('gathered actor groups contain duplicate global case ids')
    costs = [groups_by_prompt_slot[slot]['sampled_decision_cost'] for slot in expected_slots]
    current_plan = [
        sorted(
            slot for slot in expected_slots
            if source_rank_by_prompt_slot[slot] == rank
        )
        for rank in range(world_size)
    ]
    current_loads = [
        sum(groups_by_prompt_slot[slot]['sampled_decision_cost'] for slot in slots)
        for slots in current_plan
    ]
    plan, loads = rank_actor_prompt_slot_plan(costs, world_size, local_query_capacity)
    plan = _assign_actor_shards_with_locality(plan, source_rank_by_prompt_slot)
    loads = [
        sum(groups_by_prompt_slot[slot]['sampled_decision_cost'] for slot in slots)
        for slots in plan
    ]
    current_max = max(current_loads)
    planned_max = max(loads)
    relative_gain = (current_max - planned_max) / current_max
    if relative_gain < 0:
        raise RuntimeError('balanced actor plan increased the maximum decision load')
    return {
        'prompt_slots_by_rank': plan,
        'sampled_decision_loads': loads,
        'current_prompt_slots_by_rank': current_plan,
        'current_sampled_decision_loads': current_loads,
        'max_load_relative_gain': relative_gain,
        'groups_by_prompt_slot': groups_by_prompt_slot,
        'source_rank_by_prompt_slot': source_rank_by_prompt_slot,
    }


def actor_rows_for_rank(
    plan: Mapping[str, Any],
    *,
    rank: int,
    group_size: int,
) -> tuple[list[Mapping[str, Any]], list[float]]:
    """Flatten one planned rank in rollout-compatible group-major order."""
    rank = _strict_int(rank, name='rank')
    group_size = _strict_int(group_size, name='group_size', minimum=1)
    slots_by_rank = plan.get('prompt_slots_by_rank')
    groups_by_slot = plan.get('groups_by_prompt_slot')
    if not isinstance(slots_by_rank, list) or rank >= len(slots_by_rank):
        raise ValueError('rank is outside actor plan')
    if not isinstance(groups_by_slot, Mapping):
        raise TypeError('actor plan has no groups_by_prompt_slot mapping')
    assigned_slots = [int(slot) for slot in slots_by_rank[rank]]

    rows = []
    for group_index in range(group_size):
        for prompt_slot in assigned_slots:
            group = groups_by_slot[prompt_slot]
            matches = [
                row for row in group['trajectories']
                if int(row['group_index']) == group_index
            ]
            if len(matches) != 1:
                raise ValueError(f'prompt slot {prompt_slot} lacks group index {group_index}')
            rows.append(matches[0])
    advantages = [float(row['advantage']) for row in rows]
    if not np.isfinite(advantages).all():
        raise ValueError('planned actor advantages are non-finite')
    return rows, advantages
