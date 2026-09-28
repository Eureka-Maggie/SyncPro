#!/usr/bin/env python3
"""Build an auditable distributed shard plan for canonical timing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.rl.dual_grpo_schedule import (  # noqa: E402
    BALANCED_FRAMES, BEHAVIOR_SOURCE_MIXED_DATA, CONTIGUOUS, FIXED_EPOCH,
    FULL_POSITIVE_DATA,
    deterministic_batch_indices, rank_prompt_slot_shards, shard_loads, timing_epoch_indices)
from examples.rl.proactive_timing_data import (  # noqa: E402
    file_sha256, load_verified_clip_catalog, load_verified_split, training_clip_catalog)

SHARD_PLAN_SCHEMA = 'dual_grpo_shard_plan_v1'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_jsonl', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--clip_catalog', default=None)
    parser.add_argument('--clip_catalog_manifest', default=None)
    parser.add_argument('--output', default=None, help='Optional JSON plan path; stdout always gets a summary.')
    parser.add_argument('--max_frames', type=int, default=60)
    parser.add_argument('--world_size', type=int, required=True)
    parser.add_argument('--per_device_query_batch_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_schedule', choices=(FIXED_EPOCH, ), default=FIXED_EPOCH)
    parser.add_argument('--timing_data_variant', choices=(FULL_POSITIVE_DATA, ), default=FULL_POSITIVE_DATA)
    parser.add_argument('--timing_epoch_steps', type=int, default=0)
    parser.add_argument('--rank_sharding', choices=(BALANCED_FRAMES, ), default=BALANCED_FRAMES)
    parser.add_argument('--steps', type=int, default=0, help='0 plans one complete epoch.')
    args = parser.parse_args(argv)
    if min(args.max_frames, args.world_size, args.per_device_query_batch_size) < 1 \
            or min(args.steps, args.timing_epoch_steps) < 0:
        parser.error('frame, world-size, batch, and step values must be non-negative/positive')
    if args.timing_data_variant == FULL_POSITIVE_DATA:
        if args.timing_epoch_steps:
            parser.error('full_positive_v1 requires timing_epoch_steps=0')
    elif args.timing_epoch_steps < 1 or args.batch_schedule != FIXED_EPOCH:
        parser.error('materialized timing data requires timing_epoch_steps>=1 and fixed_epoch_v1')
    if bool(args.clip_catalog) != bool(args.clip_catalog_manifest):
        parser.error('clip catalog and clip catalog manifest must be provided together')
    return args


def _expanded_clips(samples: list[dict], max_frames: int) -> list[dict]:
    return [{**row, 'case_id': row['pool_case_id']} for row in training_clip_catalog(samples, max_frames)]


def build_plan(args) -> dict:
    samples, manifest = load_verified_split(args.train_jsonl, args.manifest)
    clips = _expanded_clips(samples, args.max_frames)
    clip_catalog_info = None
    if args.clip_catalog:
        catalog, catalog_manifest = load_verified_clip_catalog(
            args.clip_catalog, args.clip_catalog_manifest)
        recomputed = [{key: value for key, value in clip.items() if key != 'case_id'} for clip in clips]
        if catalog != recomputed:
            raise ValueError('clip catalog labels differ from canonical segmentation')
        clip_catalog_info = {
            'path': str(Path(args.clip_catalog).expanduser().resolve()),
            'sha256': catalog_manifest['output']['sha256'],
            'manifest_path': str(Path(args.clip_catalog_manifest).expanduser().resolve()),
            'manifest_sha256': file_sha256(args.clip_catalog_manifest),
        }
    query_batch_size = args.world_size * args.per_device_query_batch_size
    pool_case_count = len(clips)
    pool_behavior_counts = {}
    pool_source_behavior_counts = {}
    for clip in clips:
        behavior = clip['behavior_category']
        pool_behavior_counts[behavior] = pool_behavior_counts.get(behavior, 0) + 1
        key = f'{clip["source"]}/{behavior}'
        pool_source_behavior_counts[key] = pool_source_behavior_counts.get(key, 0) + 1
    if args.timing_data_variant != FULL_POSITIVE_DATA:
        indices = timing_epoch_indices(
            clips,
            epoch_steps=args.timing_epoch_steps,
            query_batch_size=query_batch_size,
            seed=args.seed,
            variant=args.timing_data_variant,
            work_units=[clip['work_units'] for clip in clips],
        )
        clips = [{
            **clips[index],
            'case_id': slot,
            'pool_case_id': clips[index]['case_id'],
            'epoch_slot': slot,
        } for slot, index in enumerate(indices)]
    schedule_rows = []
    if args.timing_data_variant != FULL_POSITIVE_DATA:
        for clip in clips:
            row = {
                'epoch_slot': clip['epoch_slot'],
                'pool_case_id': clip['pool_case_id'],
                'sample_id': clip['sample_id'],
                'source': clip['source'],
                'timing_profile': clip['timing_profile'],
            }
            if args.timing_data_variant == BEHAVIOR_SOURCE_MIXED_DATA:
                row['behavior_category'] = clip['behavior_category']
            schedule_rows.append(row)
    schedule_sha256 = (hashlib.sha256(
        json.dumps(schedule_rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                       if schedule_rows else None)
    source_counts = {}
    source_profile_counts = {}
    behavior_counts = {}
    source_behavior_counts = {}
    for clip in clips:
        source_counts[clip['source']] = source_counts.get(clip['source'], 0) + 1
        key = f'{clip["source"]}/{clip["timing_profile"]}'
        source_profile_counts[key] = source_profile_counts.get(key, 0) + 1
        behavior = clip['behavior_category']
        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        key = f'{clip["source"]}/{behavior}'
        source_behavior_counts[key] = source_behavior_counts.get(key, 0) + 1
    steps_per_epoch = (len(clips) + query_batch_size - 1) // query_batch_size
    step_count = args.steps or steps_per_epoch
    all_work_units = [clip['work_units'] for clip in clips]
    steps = []
    for step in range(step_count):
        indices = deterministic_batch_indices(
            len(clips),
            step,
            query_batch_size,
            args.seed,
            work_units=all_work_units,
            strategy=args.batch_schedule,
        )
        batch = [clips[index] for index in indices]
        work_units = [clip['work_units'] for clip in batch]
        balanced = rank_prompt_slot_shards(work_units, args.world_size, BALANCED_FRAMES)
        contiguous = rank_prompt_slot_shards(work_units, args.world_size, CONTIGUOUS)
        balanced_loads = shard_loads(balanced, work_units)
        contiguous_loads = shard_loads(contiguous, work_units)
        selected_shards = rank_prompt_slot_shards(work_units, args.world_size, args.rank_sharding)
        selected_loads = shard_loads(selected_shards, work_units)
        steps.append({
            'step': step,
            'contiguous_rank_loads': contiguous_loads,
            'balanced_rank_loads': balanced_loads,
            'contiguous_max_min_ratio': max(contiguous_loads) / min(contiguous_loads),
            'balanced_max_min_ratio': max(balanced_loads) / min(balanced_loads),
            'selected_rank_loads': selected_loads,
            'ranks': [{
                'rank': rank,
                'prompt_slots': slots,
                'estimated_work_units': selected_loads[rank],
                'cases': [{
                    'prompt_slot': slot,
                    **batch[slot],
                } for slot in slots],
            } for rank, slots in enumerate(selected_shards)],
        })
    return {
        'schema_version': SHARD_PLAN_SCHEMA,
        'train_jsonl': str(Path(args.train_jsonl).expanduser().resolve()),
        'train_sha256': file_sha256(args.train_jsonl),
        'manifest': str(Path(args.manifest).expanduser().resolve()),
        'manifest_sha256': file_sha256(args.manifest),
        'source_schema_version': manifest['schema_version'],
        'source_rows': len(samples),
        'expanded_cases': len(clips),
        'pool_cases': pool_case_count,
        'clip_catalog': clip_catalog_info,
        'pool_behavior_counts': pool_behavior_counts,
        'pool_source_behavior_counts': pool_source_behavior_counts,
        'max_frames': args.max_frames,
        'seed': args.seed,
        'batch_schedule': args.batch_schedule,
        'timing_data_variant': args.timing_data_variant,
        'timing_epoch_steps': args.timing_epoch_steps,
        'schedule_sha256': schedule_sha256,
        'source_counts': source_counts,
        'source_profile_counts': source_profile_counts,
        'behavior_counts': behavior_counts,
        'source_behavior_counts': source_behavior_counts,
        'rank_sharding': args.rank_sharding,
        'world_size': args.world_size,
        'per_device_query_batch_size': args.per_device_query_batch_size,
        'query_batch_size': query_batch_size,
        'steps_per_epoch': steps_per_epoch,
        'planned_steps': step_count,
        'steps': steps,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    ratios = [step['balanced_max_min_ratio'] for step in plan['steps']]
    print(
        f"[shard-plan] rows={plan['source_rows']} clips={plan['expanded_cases']} "
        f"query_bs={plan['query_batch_size']} world_size={plan['world_size']} "
        f"steps={plan['planned_steps']} worst_balanced_max/min={max(ratios):.3f}",
        flush=True,
    )
    first = plan['steps'][0]
    print(
        f"[shard-plan] step=0 contiguous={first['contiguous_rank_loads']} "
        f"balanced={first['balanced_rank_loads']}",
        flush=True,
    )
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f'.{output.name}.tmp.{os.getpid()}')
        temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(output)
        print(f'[shard-plan] wrote {output.resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
