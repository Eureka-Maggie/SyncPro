#!/usr/bin/env python3
"""Fail-closed validation for fixed-order screened timing RL schedules."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rl.dual_grpo_schedule import FIXED_EPOCH, deterministic_batch_indices
from examples.rl.proactive_timing_data import (
    BEHAVIOR_CATEGORIES,
    file_sha256,
    load_verified_clip_catalog,
    load_verified_split,
    training_clip_catalog,
)


SCREENED_SCHEDULE_QUOTAS = {
    'balanced': (4, 4, 4, 4),
    'precision_guard': (6, 5, 2, 3),
    'multi_event': (3, 3, 4, 6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--clip_catalog', required=True)
    parser.add_argument('--clip_catalog_manifest', required=True)
    parser.add_argument('--prefix_train')
    parser.add_argument('--query_batch_size', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=60)
    parser.add_argument('--max_frames', type=int, default=60)
    parser.add_argument(
        '--video-root', '--video_root',
        dest='video_root',
        action='append',
        default=[],
        metavar='SOURCE=PATH',
        help='Override a video root from the portable manifest; repeat as needed.',
    )
    return parser.parse_args()


def verify(args: argparse.Namespace) -> dict:
    if args.query_batch_size < len(BEHAVIOR_CATEGORIES) or args.max_steps < 1:
        raise ValueError('schedule must cover all behavior categories for at least one step')

    train, manifest = load_verified_split(args.train, args.manifest)
    validation, _ = load_verified_split(args.validation, args.manifest)
    catalog, _ = load_verified_clip_catalog(args.clip_catalog, args.clip_catalog_manifest)
    recomputed = training_clip_catalog(train, args.max_frames)
    if catalog != recomputed:
        raise ValueError('clip catalog differs from canonical training segmentation')
    if not manifest.get('fixed_epoch_order'):
        raise ValueError('screened timing mix must declare fixed_epoch_order=true')
    if int(manifest.get('query_batch_size', -1)) != args.query_batch_size:
        raise ValueError('manifest query_batch_size differs from the requested schedule')
    if len(catalog) % args.query_batch_size:
        raise ValueError('fixed epoch does not contain complete global query batches')
    if int(manifest.get('optimizer_steps', -1)) != args.max_steps:
        raise ValueError('manifest optimizer_steps differs from the requested run')

    prefix_info = None
    prefix = manifest.get('append_only_prefix')
    if prefix is not None:
        if not isinstance(prefix, dict) or not args.prefix_train:
            raise ValueError('append-only data requires --prefix_train and a prefix contract')
        prefix_path = Path(args.prefix_train).expanduser().resolve()
        prefix_bytes = prefix_path.read_bytes()
        train_bytes = Path(args.train).expanduser().resolve().read_bytes()
        prefix_rows = int(prefix.get('rows', -1))
        prefix_steps = int(prefix.get('optimizer_steps', -1))
        if file_sha256(prefix_path) != prefix.get('train_sha256'):
            raise ValueError('prefix train SHA256 differs from the manifest contract')
        if prefix_bytes.count(b'\n') != prefix_rows:
            raise ValueError('prefix train row count differs from the manifest contract')
        if prefix_rows != prefix_steps * args.query_batch_size:
            raise ValueError('prefix rows do not match prefix steps * query batch size')
        if not train_bytes.startswith(prefix_bytes):
            raise ValueError('steps before the extension are not a byte-exact train prefix')
        prefix_info = {
            'rows': prefix_rows,
            'steps': prefix_steps,
            'sha256': file_sha256(prefix_path),
            'exact_byte_prefix': True,
        }
    elif args.prefix_train:
        raise ValueError('--prefix_train was provided but the manifest has no prefix contract')

    manifest_dir = Path(args.manifest).expanduser().resolve().parent
    video_roots = {
        source: [
            (Path(value).expanduser() if Path(value).is_absolute()
             else (manifest_dir / value).resolve())
            for value in values
        ]
        for source, values in manifest.get('video_roots', {}).items()
    }
    overrides = {}
    for value in args.video_root:
        source, separator, path = value.partition('=')
        if not separator or not source or not path:
            raise ValueError(f'--video-root must be SOURCE=PATH, got {value!r}')
        overrides.setdefault(source, []).append(Path(path).expanduser().resolve())
    video_roots.update(overrides)
    unique_videos = {}
    for row in [*train, *validation]:
        video = row['video']
        key = (video['root_key'], video['path'])
        unique_videos.setdefault(key, row['sample_id'])
    for (root_key, relative_path), sample_id in unique_videos.items():
        if not any((root / relative_path).is_file() for root in video_roots.get(root_key, [])):
            raise FileNotFoundError(
                f'{sample_id} video is unavailable for root_key={root_key!r}: {relative_path}'
            )

    if len(catalog) != len(train):
        raise ValueError('screened fixed-order data must yield exactly one clip per source row')
    annotated_categories = []
    for index, row in enumerate(train):
        category = row.get('screening', {}).get('behavior_category')
        if category not in BEHAVIOR_CATEGORIES:
            raise ValueError(f'train row {index} lacks a canonical screening behavior category')
        annotated_categories.append(category)

    epoch_steps = len(catalog) // args.query_batch_size
    schedule_info = None
    schedule_arm = manifest.get('schedule_arm')
    if schedule_arm is not None:
        if schedule_arm not in SCREENED_SCHEDULE_QUOTAS:
            raise ValueError(f'unsupported screened schedule arm: {schedule_arm!r}')
        base_steps = defaultdict(list)
        base_occurrences = defaultdict(list)
        base_categories = {}
        repeated_flags = 0
        for index, row in enumerate(train):
            category = annotated_categories[index]
            schedule = row.get('screening', {}).get('schedule')
            if not isinstance(schedule, dict):
                raise ValueError(f'train row {index} lacks materialized schedule metadata')
            step = index // args.query_batch_size
            base_id = schedule.get('base_sample_id')
            occurrence = schedule.get('exposure_occurrence')
            repeated = schedule.get('is_repeated_exposure')
            if (schedule.get('arm') != schedule_arm or schedule.get('step') != step
                    or schedule.get('behavior_slot') != category):
                raise ValueError(f'train row {index} differs from its materialized schedule slot')
            if not isinstance(base_id, str) or not base_id:
                raise ValueError(f'train row {index} has an invalid base_sample_id')
            if (not isinstance(occurrence, int) or occurrence < 0
                    or not isinstance(repeated, bool) or repeated != (occurrence > 0)):
                raise ValueError(f'train row {index} has invalid exposure metadata')
            previous_category = base_categories.setdefault(base_id, category)
            if previous_category != category:
                raise ValueError(f'base sample {base_id!r} changes behavior category')
            base_steps[base_id].append(step)
            base_occurrences[base_id].append(occurrence)
            repeated_flags += int(repeated)

        for step in range(epoch_steps):
            start = step * args.query_batch_size
            base_ids = [
                train[index]['screening']['schedule']['base_sample_id']
                for index in range(start, start + args.query_batch_size)
            ]
            if len(base_ids) != len(set(base_ids)):
                raise ValueError(f'step {step + 1} repeats a base sample inside one query batch')
        for base_id, occurrences in base_occurrences.items():
            if sorted(occurrences) != list(range(len(occurrences))):
                raise ValueError(f'base sample {base_id!r} has non-contiguous exposure occurrences')
        repeat_gaps = [
            current - previous
            for steps in base_steps.values()
            for previous, current in zip(steps, steps[1:])
        ]
        if repeat_gaps and min(repeat_gaps) < 40:
            raise ValueError('repeated base samples must be separated by at least 40 optimizer steps')
        repeated_exposures = len(train) - len(base_steps)
        if repeated_flags != repeated_exposures:
            raise ValueError('schedule repeat flags differ from repeated base-sample exposures')
        schedule_info = {
            'arm': schedule_arm,
            'exposure_counts': dict(Counter(annotated_categories)),
            'unique_base_counts': dict(Counter(base_categories.values())),
            'unique_base_rows': len(base_steps),
            'repeated_exposures': repeated_exposures,
            'minimum_repeat_gap_steps': min(repeat_gaps) if repeat_gaps else None,
        }

    declared_phases = manifest.get('schedule_phases')
    phases = []
    if declared_phases is None:
        totals = Counter(annotated_categories)
        if any(totals[category] % epoch_steps for category in BEHAVIOR_CATEGORIES):
            raise ValueError('epoch behavior totals do not define an exact per-step ratio')
        expected = {category: totals[category] // epoch_steps for category in BEHAVIOR_CATEGORIES}
        if schedule_arm is not None:
            declared = dict(zip(BEHAVIOR_CATEGORIES, SCREENED_SCHEDULE_QUOTAS[schedule_arm]))
            if expected != declared:
                raise ValueError(
                    f'{schedule_arm} epoch totals differ from its declared per-step quota: '
                    f'expected={declared} actual={expected}')
        phases = [{'start_step': 1, 'end_step': epoch_steps, 'per_step_behavior_counts': expected}]
    else:
        if not isinstance(declared_phases, list) or not declared_phases:
            raise ValueError('schedule_phases must be a non-empty list')
        next_step = 1
        for phase in declared_phases:
            start_step = int(phase.get('start_step', -1))
            end_step = int(phase.get('end_step', -1))
            counts = phase.get('per_step_behavior_counts')
            if start_step != next_step or end_step < start_step or not isinstance(counts, dict):
                raise ValueError('schedule_phases must cover consecutive, non-empty step ranges')
            expected = {category: int(counts.get(category, 0)) for category in BEHAVIOR_CATEGORIES}
            if sum(expected.values()) != args.query_batch_size or any(value < 1 for value in expected.values()):
                raise ValueError(f'invalid per-step behavior counts in phase {start_step}-{end_step}')
            phases.append({
                'start_step': start_step,
                'end_step': end_step,
                'per_step_behavior_counts': expected,
            })
            next_step = end_step + 1
        if next_step != epoch_steps + 1:
            raise ValueError('schedule_phases do not cover the complete fixed epoch')

    for step in range(args.max_steps):
        indices = deterministic_batch_indices(
            len(catalog),
            step,
            args.query_batch_size,
            seed=0,
            strategy=FIXED_EPOCH,
        )
        epoch_step = step % epoch_steps + 1
        phase = next(
            phase for phase in phases
            if phase['start_step'] <= epoch_step <= phase['end_step']
        )
        expected = Counter(phase['per_step_behavior_counts'])
        actual = Counter(annotated_categories[index] for index in indices)
        if actual != expected:
            raise ValueError(
                f'step {step + 1} behavior ratio differs: '
                f'expected={dict(expected)} actual={dict(actual)}'
            )

    return {
        'train': str(Path(args.train)),
        'train_rows': len(train),
        'train_sha256': file_sha256(args.train),
        'validation': str(Path(args.validation)),
        'validation_rows': len(validation),
        'validation_sha256': file_sha256(args.validation),
        'unique_videos': len(unique_videos),
        'epoch_steps': epoch_steps,
        'run_steps': args.max_steps,
        'query_batch_size': args.query_batch_size,
        'schedule_phases': phases,
        'materialized_schedule': schedule_info,
        'append_only_prefix': prefix_info,
        'fixed_epoch_cycles': args.max_steps / epoch_steps,
    }


def main() -> None:
    print(json.dumps(verify(parse_args()), ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
