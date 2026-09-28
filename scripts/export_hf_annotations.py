#!/usr/bin/env python3
"""Build the annotation-only SyncPro-189k Hugging Face release."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import re
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_KEYS = {'messages', 'images', 'data_source', 'video_id'}
SOURCE_DIRECTORIES = {
    'HACS': 'HACS',
    'MAGQA': 'magqa',
    'ActivityNet_Captions': 'ActivityNet_Captions',
    'COIN': 'COIN',
    'qvh': 'qvh',
    'didemo': 'didemo',
    'StreamingCoT': 'StreamingCoT',
    'MovieNet': 'movienet',
    'Ego4D-GoalStep': 'ego4d',
    'PerceptionTest': 'perceptiontest',
    'youcook2': 'youcook2',
    'UCF-Crime': 'ucf',
}
INTERNAL_MARKERS = (
    '/pri' + 'mus',
    'pri' + 'mus_',
    'tian' + 'xueyun',
    'gitlab.' + 'ali' + 'baba',
)
WINDOWS_ABSOLUTE = re.compile(r'^[A-Za-z]:[\\/]')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')


def deterministic_gzip_writer(path: Path, compresslevel: int) -> TextIO:
    raw = path.open('wb')
    compressed = gzip.GzipFile(filename='', mode='wb', compresslevel=compresslevel, fileobj=raw, mtime=0)
    return io.TextIOWrapper(compressed, encoding='utf-8', newline='\n')


def validate_scalar_strings(value: Any, *, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            validate_scalar_strings(child, location=f'{location}.{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_scalar_strings(child, location=f'{location}[{index}]')
    elif isinstance(value, str):
        lowered = value.lower()
        marker = next((item for item in INTERNAL_MARKERS if item in lowered), None)
        if marker:
            raise ValueError(f'internal marker {marker!r} at {location}')
        if Path(value).is_absolute() or WINDOWS_ABSOLUTE.match(value):
            raise ValueError(f'absolute path at {location}: {value!r}')


def validate_target(target: str, *, row_number: int) -> None:
    try:
        branch_b, branch_a = target.split('<|im_end|>\n', 1)
    except ValueError as error:
        raise ValueError(f'row {row_number}: assistant target has no branch separator') from error
    valid_b = branch_b == '<silent>' or (branch_b.startswith('<inner>') and branch_b.endswith('</inner>'))
    valid_a = branch_a == '<silent>' or (branch_a.startswith('<speak>') and branch_a.endswith('</speak>'))
    if not valid_b or not valid_a:
        raise ValueError(f'row {row_number}: assistant target violates the dual-stream grammar')


def portable_image_path(source: str, video_id: str, raw_path: str, *, row_number: int) -> str:
    if source not in SOURCE_DIRECTORIES:
        raise ValueError(f'row {row_number}: unknown data source {source!r}')
    if not video_id or '/' in video_id or '\\' in video_id or video_id in {'.', '..'}:
        raise ValueError(f'row {row_number}: unsafe video_id {video_id!r}')
    source_path = PurePosixPath(raw_path.replace('\\', '/'))
    if not source_path.name or source_path.name in {'.', '..'}:
        raise ValueError(f'row {row_number}: unsafe frame path {raw_path!r}')
    if len(source_path.parts) < 3 or source_path.parts[-3] != 'frames' or source_path.parts[-2] != video_id:
        raise ValueError(f'row {row_number}: frame path does not match video_id: {raw_path!r}')
    return PurePosixPath(SOURCE_DIRECTORIES[source], 'frames', video_id, source_path.name).as_posix()


def sanitize_sft_row(row: dict[str, Any], *, row_number: int) -> tuple[dict[str, Any], int]:
    if set(row) != EXPECTED_KEYS:
        raise ValueError(f'row {row_number}: expected keys {sorted(EXPECTED_KEYS)}, got {sorted(row)}')
    messages = row['messages']
    images = row['images']
    source = row['data_source']
    video_id = row['video_id']
    if not isinstance(messages, list) or not isinstance(images, list):
        raise ValueError(f'row {row_number}: messages and images must be lists')
    assistants = [message for message in messages if message.get('role') == 'assistant']
    if len(assistants) != len(images):
        raise ValueError(
            f'row {row_number}: {len(assistants)} assistant turns but {len(images)} image references'
        )
    for message in messages:
        if set(message) != {'role', 'content'} or message['role'] not in {'user', 'assistant'}:
            raise ValueError(f'row {row_number}: invalid message object')
        if not isinstance(message['content'], str):
            raise ValueError(f'row {row_number}: message content must be text')
        if message['role'] == 'assistant':
            validate_target(message['content'], row_number=row_number)
    portable = [portable_image_path(source, video_id, path, row_number=row_number) for path in images]
    result = {
        'messages': messages,
        'images': portable,
        'data_source': source,
        'video_id': video_id,
    }
    validate_scalar_strings(result, location=f'sft[{row_number}]')
    return result, len(portable)


def copy_and_validate_rl(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    files = {
        'train.jsonl': output_dir / 'train.jsonl',
        'val.jsonl': output_dir / 'validation.jsonl',
        'manifest.json': output_dir / 'metadata' / 'schedule_manifest.json',
        'train_clip_catalog_f60.jsonl': output_dir / 'metadata' / 'train_clip_catalog_f60.jsonl',
        'train_clip_catalog_f60.manifest.json': output_dir / 'metadata' / 'train_clip_catalog_f60.manifest.json',
    }
    files['manifest.json'].parent.mkdir(parents=True)
    summary: dict[str, Any] = {'files': {}}
    for source_name, destination in files.items():
        source = source_dir / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix == '.jsonl':
            rows = 0
            with source.open(encoding='utf-8') as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    validate_scalar_strings(json.loads(line), location=f'rl/{source_name}:{line_number}')
                    rows += 1
            summary['files'][destination.relative_to(output_dir.parent).as_posix()] = {'rows': rows}
        else:
            validate_scalar_strings(json.loads(source.read_text(encoding='utf-8')), location=f'rl/{source_name}')
        shutil.copyfile(source, destination)
        summary['files'].setdefault(destination.relative_to(output_dir.parent).as_posix(), {})
        summary['files'][destination.relative_to(output_dir.parent).as_posix()].update({
            'bytes': destination.stat().st_size,
            'sha256': sha256(destination),
        })
    if summary['files']['rl/train.jsonl'].get('rows') != 800:
        raise ValueError('RL train split must contain 800 rows')
    if summary['files']['rl/validation.jsonl'].get('rows') != 58:
        raise ValueError('RL validation split must contain 58 rows')
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft-input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--rl-dir', type=Path, default=ROOT / 'data' / 'rl' / 'balanced')
    parser.add_argument('--dataset-card', type=Path, default=ROOT / 'docs' / 'huggingface_dataset_card.md')
    parser.add_argument('--expected-sft-rows', type=int, default=189403)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--compresslevel', type=int, default=6, choices=range(1, 10))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shards < 1 or args.expected_sft_rows < 1:
        raise SystemExit('--shards and --expected-sft-rows must be positive')
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'refusing non-empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    sft_dir = output / 'sft'
    sft_dir.mkdir()
    shard_paths = [
        sft_dir / f'train-{index:05d}-of-{args.shards:05d}.jsonl.gz'
        for index in range(args.shards)
    ]
    rows_per_shard = (args.expected_sft_rows + args.shards - 1) // args.shards
    input_digest = hashlib.sha256()
    source_counts: Counter[str] = Counter()
    shard_rows: Counter[int] = Counter()
    total_rows = 0
    total_images = 0

    with contextlib.ExitStack() as stack:
        writers = [stack.enter_context(deterministic_gzip_writer(path, args.compresslevel)) for path in shard_paths]
        with args.sft_input.expanduser().open('rb') as handle:
            for row_number, raw_line in enumerate(handle, start=1):
                input_digest.update(raw_line)
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                clean, image_count = sanitize_sft_row(row, row_number=row_number)
                shard = min(total_rows // rows_per_shard, args.shards - 1)
                writers[shard].write(json_bytes(clean).decode('utf-8'))
                source_counts[clean['data_source']] += 1
                shard_rows[shard] += 1
                total_images += image_count
                total_rows += 1
                if total_rows % 10000 == 0:
                    print(f'[sft] validated and wrote {total_rows:,} rows', flush=True)

    if total_rows != args.expected_sft_rows:
        raise ValueError(f'expected {args.expected_sft_rows} SFT rows, found {total_rows}')
    if set(source_counts) != set(SOURCE_DIRECTORIES):
        raise ValueError(f'unexpected source set: {sorted(source_counts)}')

    sft_files = {}
    for index, path in enumerate(shard_paths):
        sft_files[path.relative_to(output).as_posix()] = {
            'rows': shard_rows[index],
            'bytes': path.stat().st_size,
            'sha256': sha256(path),
        }

    rl_summary = copy_and_validate_rl(args.rl_dir.expanduser(), output / 'rl')
    shutil.copyfile(args.dataset_card.expanduser(), output / 'README.md')
    manifest = {
        'release': 'SyncPro-189k',
        'annotations_only': True,
        'license': 'CC-BY-NC-SA-4.0',
        'sft': {
            'rows': total_rows,
            'image_references': total_images,
            'source_counts': dict(sorted(source_counts.items())),
            'input_sha256': input_digest.hexdigest(),
            'files': sft_files,
        },
        'rl': rl_summary,
    }
    manifest_path = output / 'release_manifest.json'
    manifest_path.write_bytes(json_bytes(manifest))
    validate_scalar_strings(manifest, location='release_manifest')
    print(f'wrote validated Hugging Face release to {output}')
    print(f'SFT rows={total_rows:,} image references={total_images:,}')


if __name__ == '__main__':
    main()
