#!/usr/bin/env python3
"""Create the portable balanced hard-F1 annotation bundle.

The released bundle intentionally stores no machine-specific video roots and
no screening-service metadata. Video locations are supplied at launch time as
``SOURCE=/path`` mappings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.rl.proactive_timing_data import (
    SCHEMA_VERSION,
    file_sha256,
    validate_samples,
    write_training_clip_catalog,
)


def json_line(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sanitize(row: dict, *, keep_schedule: bool) -> dict:
    result = {
        key: row[key]
        for key in (
            'schema_version',
            'sample_id',
            'source',
            'task',
            'query',
            'video',
            'stream',
            'gt_windows',
        )
    }
    if keep_schedule:
        screening = row['screening']
        schedule = screening['schedule']
        result['screening'] = {
            'behavior_category': screening['behavior_category'],
            'provenance': {
                'dataset': screening['original_dataset'],
                'sample_id': screening['original_sample_id'],
                'video_id': screening['original_video_id'],
                'crop_start_frame': screening['crop_start_frame'],
                'crop_end_frame': screening['crop_end_frame'],
            },
            'schedule': {
                key: schedule[key]
                for key in (
                    'arm',
                    'step',
                    'behavior_slot',
                    'base_sample_id',
                    'exposure_occurrence',
                    'is_repeated_exposure',
                )
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--validation', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--query-batch-size', type=int, default=16)
    parser.add_argument(
        '--optimizer-steps',
        type=int,
        default=None,
        help='Keep the fixed-order prefix used by the reported checkpoint.',
    )
    parser.add_argument('--max-frames', type=int, default=60)
    args = parser.parse_args()

    train = validate_samples(sanitize(row, keep_schedule=True) for row in load_jsonl(args.train))
    if args.optimizer_steps is not None:
        if args.optimizer_steps < 1:
            raise ValueError('--optimizer-steps must be positive')
        required_rows = args.optimizer_steps * args.query_batch_size
        if required_rows > len(train):
            raise ValueError(
                f'requested {required_rows} rows for {args.optimizer_steps} steps, '
                f'but only {len(train)} are available'
            )
        train = train[:required_rows]
    validation = validate_samples(sanitize(row, keep_schedule=False) for row in load_jsonl(args.validation))
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / 'train.jsonl'
    validation_path = output / 'val.jsonl'
    train_path.write_bytes(b''.join(json_line(row) for row in train))
    validation_path.write_bytes(b''.join(json_line(row) for row in validation))

    if len(train) % args.query_batch_size:
        raise ValueError('training rows must form complete query batches')
    source_counts = Counter(row['source'] for row in train)
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'purpose': 'balanced_pool_weighted_exposure_schedule',
        'schedule_arm': 'balanced',
        'fixed_epoch_order': True,
        'query_batch_size': args.query_batch_size,
        'optimizer_steps': len(train) // args.query_batch_size,
        'seed': 20260815,
        'sources': {source: {'train_rows': count} for source, count in sorted(source_counts.items())},
        'video_roots': {},
        'outputs': {
            'train.jsonl': {'rows': len(train), 'sha256': file_sha256(train_path)},
            'val.jsonl': {'rows': len(validation), 'sha256': file_sha256(validation_path)},
        },
    }
    manifest_path = output / 'manifest.json'
    manifest_path.write_bytes(json_line(manifest))
    write_training_clip_catalog(
        train,
        output / f'train_clip_catalog_f{args.max_frames}.jsonl',
        output / f'train_clip_catalog_f{args.max_frames}.manifest.json',
        source_path=train_path,
        source_manifest_path=manifest_path,
        max_frames=args.max_frames,
    )
    print(f'wrote portable balanced schedule to {output.resolve()}')


if __name__ == '__main__':
    main()
