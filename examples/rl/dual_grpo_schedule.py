"""Deterministic batching and rank assignment for dual-stream GRPO."""

from __future__ import annotations

import hashlib

import numpy as np

from examples.rl.proactive_timing_data import BEHAVIOR_CATEGORIES


BALANCED_FRAMES = 'balanced_frames_v1'
CONTIGUOUS = 'contiguous_v0'
RANK_SHARDING_CHOICES = (BALANCED_FRAMES, CONTIGUOUS)
LENGTH_BUCKETED = 'length_bucketed_v1'
RANDOM_BATCHES = 'random_v0'
FIXED_EPOCH = 'fixed_epoch_v1'
BATCH_SCHEDULE_CHOICES = (LENGTH_BUCKETED, RANDOM_BATCHES, FIXED_EPOCH)

FULL_POSITIVE_DATA = 'full_positive_v1'
SCHEDULED_POSITIVE_DATA = 'scheduled_positive_v1'
SOURCE_EVENT_BALANCED_DATA = 'source_event_balanced_v1'
BEHAVIOR_SOURCE_MIXED_DATA = 'behavior_source_mixed_v1'
MAX_BALANCED_BATCH_WORK_RATIO = 1.5
TIMING_DATA_VARIANTS = (
    FULL_POSITIVE_DATA,
    SCHEDULED_POSITIVE_DATA,
    SOURCE_EVENT_BALANCED_DATA,
    BEHAVIOR_SOURCE_MIXED_DATA,
)


def _stable_rank(*values) -> bytes:
    return hashlib.sha256('\0'.join(str(value) for value in values).encode('utf-8')).digest()


def _behavior_quotas(step: int, query_batch_size: int) -> dict[str, int]:
    if query_batch_size < len(BEHAVIOR_CATEGORIES):
        raise ValueError('behavior-mixed query_batch_size must cover every behavior category')
    base, remainder = divmod(query_batch_size, len(BEHAVIOR_CATEGORIES))
    quotas = {category: base for category in BEHAVIOR_CATEGORIES}
    for offset in range(remainder):
        quotas[BEHAVIOR_CATEGORIES[(step + offset) % len(BEHAVIOR_CATEGORIES)]] += 1
    return quotas


def _bounded_allocations(total: int, capacities: tuple[int, ...]):
    """Yield lexicographically ordered count vectors with a fixed total."""
    allocation = [0] * len(capacities)

    def visit(position: int, remaining: int):
        if position == len(capacities) - 1:
            if remaining <= capacities[position]:
                allocation[position] = remaining
                yield tuple(allocation)
            return
        for count in range(min(capacities[position], remaining) + 1):
            allocation[position] = count
            yield from visit(position + 1, remaining - count)

    yield from visit(0, total)


def behavior_source_epoch_quotas(
    case_metadata: list[dict],
    *,
    epoch_steps: int,
    query_batch_size: int,
) -> dict[str, dict[str, int]]:
    """Allocate each behavior's epoch slots to sources by pool capacity."""
    step_quotas = [_behavior_quotas(step, query_batch_size) for step in range(epoch_steps)]
    capacities: dict[str, dict[str, int]] = {category: {} for category in BEHAVIOR_CATEGORIES}
    for metadata in case_metadata:
        source = metadata.get('source')
        category = metadata.get('behavior_category')
        if not isinstance(source, str) or not source or category not in capacities:
            raise ValueError('behavior/source epoch quotas require canonical source and behavior metadata')
        capacities[category][source] = capacities[category].get(source, 0) + 1
    quotas = {}
    for category in BEHAVIOR_CATEGORIES:
        if not capacities[category]:
            raise ValueError(f'behavior category {category!r} has no clips')
        total_slots = sum(row[category] for row in step_quotas)
        total_capacity = sum(capacities[category].values())
        source_quotas = {
            source: total_slots * capacity // total_capacity
            for source, capacity in capacities[category].items()
        }
        remainder_slots = total_slots - sum(source_quotas.values())
        remainder_order = sorted(
            capacities[category],
            key=lambda source: (
                -(total_slots * capacities[category][source] % total_capacity),
                source,
            ),
        )
        for source in remainder_order[:remainder_slots]:
            source_quotas[source] += 1
        quotas[category] = source_quotas
    return quotas


def _future_source_feasible(
    remaining: tuple[int, ...],
    capacities: tuple[int, ...],
    future_behavior_quotas: tuple[int, ...],
) -> bool:
    """Check all source-subset capacity constraints for future steps."""
    if sum(remaining) != sum(future_behavior_quotas):
        return False
    source_count = len(remaining)
    for mask in range(1, 1 << source_count):
        selected = [index for index in range(source_count) if mask & (1 << index)]
        required = sum(remaining[index] for index in selected)
        available = sum(
            min(quota, sum(min(capacities[index], quota) for index in selected))
            for quota in future_behavior_quotas)
        if required > available:
            return False
    return True


def _step_source_allocation(
    pools: dict[str, dict[str, list[int]]],
    quotas: dict[str, int],
    remaining: dict[str, dict[str, int]],
    future_quotas: list[dict[str, int]],
) -> tuple[tuple[int, ...], ...]:
    """Maximize step source diversity without violating fixed epoch quotas."""
    sources = tuple(sorted({source for values in pools.values() for source in values}))
    if not sources:
        raise ValueError('behavior-mixed data has no sources')
    zero = (0, ) * len(sources)
    options_by_category = []
    urgency_by_category = []
    for category in BEHAVIOR_CATEGORIES:
        capacities = tuple(len(pools[category].get(source, ())) for source in sources)
        remaining_counts = tuple(remaining[category].get(source, 0) for source in sources)
        options = []
        urgency = {}
        future = tuple(row[category] for row in future_quotas)
        for allocation in _bounded_allocations(
                quotas[category],
                tuple(min(capacity, count) for capacity, count in zip(capacities, remaining_counts))):
            after = tuple(count - selected for count, selected in zip(remaining_counts, allocation))
            if not _future_source_feasible(after, capacities, future):
                continue
            options.append(allocation)
            available = tuple(
                sum(min(capacity, quota) for quota in (quotas[category], *future))
                for capacity in capacities)
            urgency[allocation] = sum(
                selected * count / max(1, capacity)
                for selected, count, capacity in zip(allocation, remaining_counts, available))
        if not options:
            raise ValueError(f'behavior category {category!r} has no future-feasible source allocation')
        options_by_category.append(tuple(options))
        urgency_by_category.append(urgency)

    # Global source counts -> (urgency, category allocation path).
    states: dict[tuple[int, ...], tuple[float, tuple[tuple[int, ...], ...]]] = {
        zero: (0.0, ()),
    }
    for options, urgency in zip(options_by_category, urgency_by_category):
        next_states = {}
        for counts, (score, path) in states.items():
            for allocation in options:
                combined = tuple(left + right for left, right in zip(counts, allocation))
                candidate = (score + urgency[allocation], path + (allocation, ))
                current = next_states.get(combined)
                if current is None or (-candidate[0], candidate[1]) < (-current[0], current[1]):
                    next_states[combined] = candidate
        states = next_states
    if not states:
        raise ValueError('behavior/source allocation has no feasible solution')

    _, (_, path) = min(
        states.items(),
        key=lambda item: (
            -sum(count > 0 for count in item[0]),
            max(item[0]),
            sum(count * count for count in item[0]),
            -item[1][0],
            item[1][1],
        ),
    )
    return path


def _behavior_source_mixed_indices(
    case_metadata: list[dict],
    *,
    epoch_steps: int,
    query_batch_size: int,
    seed: int,
    work_units: list[int],
) -> list[int]:
    pools: dict[str, dict[str, list[int]]] = {category: {} for category in BEHAVIOR_CATEGORIES}
    for index, metadata in enumerate(case_metadata):
        source = metadata.get('source')
        category = metadata.get('behavior_category')
        if not isinstance(source, str) or not source:
            raise ValueError('behavior-mixed timing data requires source metadata')
        if category not in pools:
            raise ValueError(f'unsupported behavior category: {category!r}')
        pools[category].setdefault(source, []).append(index)
    for category in BEHAVIOR_CATEGORIES:
        if not any(pools[category].values()):
            raise ValueError(f'behavior category {category!r} has no clips')

    sources = tuple(sorted({source for values in pools.values() for source in values}))
    usage = [0] * len(case_metadata)
    step_quotas = [_behavior_quotas(step, query_batch_size) for step in range(epoch_steps)]
    remaining = behavior_source_epoch_quotas(
        case_metadata,
        epoch_steps=epoch_steps,
        query_batch_size=query_batch_size,
    )
    step_cells: list[dict[tuple[str, str], int]] = []
    for step, quotas in enumerate(step_quotas):
        for category, quota in quotas.items():
            available = sum(len(values) for values in pools[category].values())
            if available < quota:
                raise ValueError(
                    f'behavior category {category!r} has {available} unique clips for quota={quota}')
        allocation = _step_source_allocation(pools, quotas, remaining, step_quotas[step + 1:])
        cells = {}
        for category, source_counts in zip(BEHAVIOR_CATEGORIES, allocation):
            for source, count in zip(sources, source_counts):
                if not count:
                    continue
                cells[(category, source)] = count
                remaining[category][source] -= count
        if sum(cells.values()) != query_batch_size:
            raise RuntimeError('behavior/source allocation did not fill one query batch')
        step_cells.append(cells)
    if any(count for values in remaining.values() for count in values.values()):
        raise RuntimeError(f'behavior/source epoch quotas were not exhausted: {remaining}')

    selected_by_cell: dict[tuple[str, str], list[tuple[int, int]]] = {}
    all_cells = sorted({cell for cells in step_cells for cell in cells})
    for category, source in all_cells:
        total = sum(cells.get((category, source), 0) for cells in step_cells)
        occurrences = []
        for occurrence in range(total):
            index = min(
                pools[category][source],
                key=lambda candidate: (
                    usage[candidate],
                    _stable_rank(seed, 'epoch', category, source, candidate),
                    candidate,
                ),
            )
            usage[index] += 1
            occurrences.append((index, occurrence))
        selected_by_cell[(category, source)] = occurrences

    step_totals = [0] * epoch_steps
    step_indices = [[] for _ in range(epoch_steps)]
    step_index_sets = [set() for _ in range(epoch_steps)]
    cell_order = sorted(
        all_cells,
        key=lambda cell: (
            -max(int(work_units[index]) for index, _ in selected_by_cell[cell]),
            -len(selected_by_cell[cell]),
            cell,
        ),
    )
    for cell in cell_order:
        occurrences = sorted(
            selected_by_cell[cell],
            key=lambda value: (
                -int(work_units[value[0]]),
                _stable_rank(seed, 'occurrence', cell[0], cell[1], value[0], value[1]),
            ),
        )
        capacity = [cells.get(cell, 0) for cells in step_cells]
        for index, occurrence in occurrences:
            eligible = [
                step for step in range(epoch_steps)
                if capacity[step] and index not in step_index_sets[step]
            ]
            if not eligible:
                raise ValueError(f'behavior/source length placement is infeasible for {cell!r}')
            step = min(
                eligible,
                key=lambda candidate: (
                    step_totals[candidate],
                    _stable_rank(seed, 'step', cell[0], cell[1], index, occurrence, candidate),
                    candidate,
                ),
            )
            capacity[step] -= 1
            step_indices[step].append(index)
            step_index_sets[step].add(index)
            step_totals[step] += int(work_units[index])
        if any(capacity):
            raise RuntimeError(f'behavior/source length placement left unfilled slots for {cell!r}')

    result = []
    for step, selected in enumerate(step_indices):
        if len(selected) != query_batch_size or len(set(selected)) != query_batch_size:
            raise RuntimeError('behavior-mixed schedule produced an invalid global query batch')
        selected.sort(key=lambda index: (_stable_rank(seed, step, 'batch', index), index))
        result.extend(selected)
    return result


def deterministic_batch_indices(
    case_count: int,
    step: int,
    query_batch_size: int,
    seed: int,
    *,
    work_units: list[int] | None = None,
    strategy: str = RANDOM_BATCHES,
) -> list[int]:
    """Return one deterministic, duplicate-free global-query batch."""
    if case_count < query_batch_size:
        raise ValueError(f'training data has {case_count} cases for query_bs={query_batch_size}')
    if step < 0 or query_batch_size < 1:
        raise ValueError('step must be non-negative and query_batch_size must be positive')
    if strategy not in BATCH_SCHEDULE_CHOICES:
        raise ValueError(f'unsupported batch schedule: {strategy!r}')
    steps_per_epoch = (case_count + query_batch_size - 1) // query_batch_size
    epoch, batch_index = divmod(step, steps_per_epoch)
    if strategy == FIXED_EPOCH:
        start = batch_index * query_batch_size
        return [(start + offset) % case_count for offset in range(query_batch_size)]
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(case_count).tolist()
    if strategy == RANDOM_BATCHES:
        start = batch_index * query_batch_size
        indices = order[start:start + query_batch_size]
        if len(indices) < query_batch_size:
            indices.extend(order[:query_batch_size - len(indices)])
        return indices

    if work_units is None or len(work_units) != case_count or any(int(value) < 1 for value in work_units):
        raise ValueError('length_bucketed_v1 requires one positive work estimate per case')
    # Random permutation supplies an epoch-specific tie order. Adjacent work
    # buckets keep all ranks on a step comparable; batch and within-batch
    # shuffles avoid a monotonic short-to-long curriculum.
    order.sort(key=lambda index: -int(work_units[index]))
    batches = [order[start:start + query_batch_size] for start in range(0, case_count, query_batch_size)]
    if len(batches[-1]) < query_batch_size:
        missing = query_batch_size - len(batches[-1])
        tail_start = case_count - len(batches[-1])
        batches[-1].extend(order[tail_start - missing:tail_start])
    for batch in batches:
        rng.shuffle(batch)
        if len(batch) != query_batch_size or len(set(batch)) != query_batch_size:
            raise RuntimeError('batch schedule produced an invalid global query batch')
    rng.shuffle(batches)
    return batches[batch_index]


def timing_epoch_indices(
    case_metadata: list[dict],
    *,
    epoch_steps: int,
    query_batch_size: int,
    seed: int,
    variant: str,
    work_units: list[int],
) -> list[int]:
    """Materialize one deterministic timing-data epoch.

    ``scheduled_positive_v1`` preserves the existing length-bucketed control.
    ``source_event_balanced_v1`` preserves the source-homogeneous ablation.
    ``behavior_source_mixed_v1`` fixes per-step behavior quotas, assigns
    epoch-level source quotas in proportion to labeled pool capacity, and
    performs deterministic length-aware step placement. No variant repeats a
    query inside one global query batch.
    """
    if variant not in TIMING_DATA_VARIANTS:
        raise ValueError(f'unsupported timing data variant: {variant!r}')
    if variant == FULL_POSITIVE_DATA:
        raise ValueError('full_positive_v1 is not a materialized epoch')
    if epoch_steps < 1 or query_batch_size < 1:
        raise ValueError('epoch_steps and query_batch_size must be positive')
    if len(case_metadata) != len(work_units) or len(case_metadata) < query_batch_size:
        raise ValueError('case metadata/work estimates do not cover one query batch')

    if variant == SCHEDULED_POSITIVE_DATA:
        return [
            index
            for step in range(epoch_steps)
            for index in deterministic_batch_indices(
                len(case_metadata),
                step,
                query_batch_size,
                seed,
                work_units=work_units,
                strategy=LENGTH_BUCKETED,
            )
        ]

    if variant == BEHAVIOR_SOURCE_MIXED_DATA:
        return _behavior_source_mixed_indices(
            case_metadata,
            epoch_steps=epoch_steps,
            query_batch_size=query_batch_size,
            seed=seed,
            work_units=work_units,
        )

    by_source: dict[str, dict[str, list[int]]] = {}
    for index, metadata in enumerate(case_metadata):
        source = metadata.get('source')
        profile = metadata.get('timing_profile')
        if not isinstance(source, str) or not source or not isinstance(profile, str) or not profile:
            raise ValueError('balanced timing data requires source and timing_profile metadata')
        by_source.setdefault(source, {}).setdefault(profile, []).append(index)
    sources = sorted(by_source)
    if not sources:
        raise ValueError('balanced timing data has no sources')

    rng = np.random.default_rng(seed)
    if epoch_steps % len(sources):
        raise ValueError('source-balanced epoch_steps must be divisible by the number of sources')
    batches_per_source = epoch_steps // len(sources)
    batches = []
    for source in sources:
        source_cases = [index for values in by_source[source].values() for index in values]
        if len(source_cases) < query_batch_size:
            raise ValueError(f'source {source!r} has fewer cases than query_batch_size')
        eligible = set()
        for anchor in source_cases:
            nearest = sorted(
                source_cases,
                key=lambda index: (abs(int(work_units[index]) - int(work_units[anchor])), index),
            )[:query_batch_size]
            ratio = max(work_units[index] for index in nearest) / min(work_units[index] for index in nearest)
            if ratio <= MAX_BALANCED_BATCH_WORK_RATIO:
                eligible.add(anchor)
        eligible_by_profile = {
            profile: [index for index in values if index in eligible]
            for profile, values in by_source[source].items()
        }
        eligible_by_profile = {profile: values for profile, values in eligible_by_profile.items() if values}
        if not eligible_by_profile:
            raise ValueError(f'source {source!r} has no length-packable anchor cases')
        source_profiles = sorted(eligible_by_profile)
        anchors = []
        while len(anchors) < batches_per_source:
            cycle = {profile: list(eligible_by_profile[profile]) for profile in source_profiles}
            for values in cycle.values():
                rng.shuffle(values)
            while any(cycle.values()) and len(anchors) < batches_per_source:
                for profile in source_profiles:
                    if cycle[profile] and len(anchors) < batches_per_source:
                        anchors.append(cycle[profile].pop())

        usage = {index: 0 for index in source_cases}
        tie_order = list(source_cases)
        rng.shuffle(tie_order)
        tie_rank = {index: rank for rank, index in enumerate(tie_order)}
        for anchor in anchors:
            anchor_work = int(work_units[anchor])
            anchor_profile = case_metadata[anchor]['timing_profile']
            batch = sorted(
                source_cases,
                key=lambda index: (
                    abs(int(work_units[index]) - anchor_work),
                    usage[index],
                    case_metadata[index]['timing_profile'] != anchor_profile,
                    tie_rank[index],
                ),
            )[:query_batch_size]
            if len(set(batch)) != query_batch_size:
                raise RuntimeError('balanced timing batch contains a duplicate query')
            work_ratio = max(work_units[index] for index in batch) / min(work_units[index] for index in batch)
            if work_ratio > MAX_BALANCED_BATCH_WORK_RATIO:
                raise RuntimeError(f'balanced timing batch exceeds work-ratio limit: {work_ratio:.3f}')
            for index in batch:
                usage[index] += 1
            rng.shuffle(batch)
            batches.append(batch)
    rng.shuffle(batches)
    return [index for batch in batches for index in batch]


def estimated_stream_work(frame_count: int) -> int:
    """Estimate streaming/full-attention work from a clip's frame count.

    Every frame adds image and branch tokens to the physical history.  Both
    cached rollout attention and full-sequence actor replay therefore grow
    super-linearly with the number of frames.  The square is deliberately a
    simple, metadata-only proxy; generated token lengths are not known until
    after rollout.
    """
    if frame_count < 1:
        raise ValueError('frame_count must be positive')
    return frame_count * frame_count


def rank_prompt_slot_shards(
    work_units: list[int],
    world_size: int,
    strategy: str = BALANCED_FRAMES,
) -> list[list[int]]:
    """Assign global prompt slots to equal-size rank shards deterministically."""
    if strategy not in RANK_SHARDING_CHOICES:
        raise ValueError(f'unsupported rank sharding strategy: {strategy!r}')
    if world_size < 1 or len(work_units) < world_size or len(work_units) % world_size:
        raise ValueError('prompt count must be >= and divisible by world_size')
    if any(int(value) < 1 for value in work_units):
        raise ValueError('work units must be positive integers')

    local_size = len(work_units) // world_size
    if strategy == CONTIGUOUS:
        return [list(range(rank * local_size, (rank + 1) * local_size)) for rank in range(world_size)]

    shards = [[] for _ in range(world_size)]
    loads = [0] * world_size
    # Capacity-constrained longest-processing-time assignment. Stable slot and
    # rank tie-breaks make every process derive the same plan independently.
    for prompt_slot in sorted(range(len(work_units)), key=lambda slot: (-work_units[slot], slot)):
        available = [rank for rank, shard in enumerate(shards) if len(shard) < local_size]
        rank = min(available, key=lambda value: (loads[value], len(shards[value]), value))
        shards[rank].append(prompt_slot)
        loads[rank] += int(work_units[prompt_slot])
    for shard in shards:
        shard.sort()
    if sorted(slot for shard in shards for slot in shard) != list(range(len(work_units))):
        raise RuntimeError('rank sharding did not cover every prompt exactly once')
    return shards


def shard_loads(shards: list[list[int]], work_units: list[int]) -> list[int]:
    """Return the estimated load of every rank in a validated shard plan."""
    slots = [slot for shard in shards for slot in shard]
    if sorted(slots) != list(range(len(work_units))):
        raise ValueError('shards must cover every prompt exactly once')
    return [sum(int(work_units[slot]) for slot in shard) for shard in shards]


def rank_actor_prompt_slot_plan(
    sampled_decision_costs: list[int],
    world_size: int,
    local_query_capacity: int,
) -> tuple[list[list[int]], list[int]]:
    """Balance complete query groups by their post-rollout decision cost.

    One input item is the total sampled-decision cost of all ``G``
    trajectories for one global prompt slot. The group is therefore never
    split, and every rank receives exactly ``local_query_capacity`` groups.
    """
    if world_size < 1 or local_query_capacity < 1:
        raise ValueError('world_size and local_query_capacity must be positive')
    expected = world_size * local_query_capacity
    if len(sampled_decision_costs) != expected:
        raise ValueError('actor cost count must equal world_size * local_query_capacity: '
                         f'{len(sampled_decision_costs)} != {expected}')
    invalid = [
        index for index, value in enumerate(sampled_decision_costs)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 1
    ]
    if invalid:
        raise ValueError(f'actor costs must be positive integers; invalid slots={invalid}')

    costs = [int(value) for value in sampled_decision_costs]
    plan = rank_prompt_slot_shards(costs, world_size, BALANCED_FRAMES)
    if any(len(slots) != local_query_capacity for slots in plan):
        raise RuntimeError('actor plan differs from the requested per-rank capacity')
    return plan, shard_loads(plan, costs)
