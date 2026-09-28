"""Canonical proactive benchmark data for hard-window timing RL."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = 'proactive_timing_v1'
CLIP_CATALOG_SCHEMA_VERSION = 'proactive_timing_clip_catalog_v1'
DEFAULT_SEED = 20260724
DEFAULT_VAL_PER_SOURCE = 8
SOURCES = frozenset({
    'sb_po',
    'ovo_crr',
    'ovo_rec',
    'ovo_ssr',
    'pvqa_ego',
    'pvqa_web',
    'pvqa_vad',
    'screened_activitynet',
    'screened_coin',
    'screened_didemo',
    'screened_ego4d',
    'screened_hacs',
    'screened_magqa',
    'screened_movienet',
    'screened_perceptiontest',
    'screened_qvh',
    'screened_streamingcot',
    'screened_ucfcrime',
    'screened_youcook2',
})
SOURCE_TASKS = {
    'sb_po': 'PO',
    'ovo_crr': 'CRR',
    'ovo_rec': 'REC',
    'ovo_ssr': 'SSR',
    'pvqa_ego': 'PVQA',
    'pvqa_web': 'PVQA',
    'pvqa_vad': 'PVQA',
    'screened_activitynet': 'PO',
    'screened_coin': 'PO',
    'screened_didemo': 'PO',
    'screened_ego4d': 'PO',
    'screened_hacs': 'PO',
    'screened_magqa': 'PO',
    'screened_movienet': 'PO',
    'screened_perceptiontest': 'PO',
    'screened_qvh': 'PO',
    'screened_streamingcot': 'PO',
    'screened_ucfcrime': 'PO',
    'screened_youcook2': 'PO',
}
BEHAVIOR_CATEGORIES = (
    'single_long_tail',
    'single_short_tail',
    'multi_sparse',
    'multi_dense',
)


def _finite_float(value, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{field} must be a finite number')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be a finite number') from exc
    if not math.isfinite(result):
        raise ValueError(f'{field} must be a finite number')
    return result


def _relative_video_path(path: str | Path, roots: Sequence[str | Path]) -> str:
    resolved = Path(path).expanduser().resolve()
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        try:
            return resolved.relative_to(root_path).as_posix()
        except ValueError:
            continue
    raise ValueError(f'video {resolved} is outside its configured roots')


def _canonical_window(start, end, *, field: str) -> dict:
    start_s = _finite_float(start, field=f'{field}.start_s')
    end_s = _finite_float(end, field=f'{field}.end_s')
    if start_s > end_s:
        raise ValueError(f'{field} is reversed: {start_s} > {end_s}')
    return {'start_s': start_s, 'end_s': end_s}


def _make_sample(
    *,
    sample_id: str,
    source: str,
    task: str,
    query: str,
    video_path: str,
    video_group: str,
    stream_start_s: float,
    stream_end_s: float,
    gt_windows: list[dict],
) -> dict:
    windows = sorted(gt_windows, key=lambda window: (window['start_s'], window['end_s']))
    sample = {
        'schema_version': SCHEMA_VERSION,
        'sample_id': sample_id,
        'source': source,
        'task': task,
        'query': query,
        'video': {
            'root_key': source,
            'path': video_path,
            'group_id': video_group,
        },
        'stream': {
            'start_s': float(stream_start_s),
            'end_s': float(stream_end_s),
            'fps': 1.0,
        },
        'gt_windows': windows,
    }
    validate_sample(sample)
    return sample


def validate_sample(sample: dict) -> None:
    if not isinstance(sample, dict):
        raise ValueError('sample must be an object')
    required = {
        'schema_version',
        'sample_id',
        'source',
        'task',
        'query',
        'video',
        'stream',
        'gt_windows',
    }
    missing = sorted(required - sample.keys())
    if missing:
        raise ValueError(f'sample is missing fields: {missing}')
    if sample['schema_version'] != SCHEMA_VERSION:
        raise ValueError(f'unsupported schema_version: {sample["schema_version"]!r}')
    for field in ('sample_id', 'task', 'query'):
        if not isinstance(sample[field], str) or not sample[field].strip():
            raise ValueError(f'{field} must be a non-empty string')
    source = sample['source']
    if source not in SOURCES:
        raise ValueError(f'unsupported source: {source!r}')
    if sample['task'] != SOURCE_TASKS[source]:
        raise ValueError(f'{source} requires task={SOURCE_TASKS[source]!r}, ' f'got {sample["task"]!r}')

    video = sample['video']
    if not isinstance(video, dict):
        raise ValueError('video must be an object')
    for field in ('root_key', 'path', 'group_id'):
        if not isinstance(video.get(field), str) or not video[field].strip():
            raise ValueError(f'video.{field} must be a non-empty string')
    if video['root_key'] != source:
        raise ValueError('video.root_key must equal source')

    stream = sample['stream']
    if not isinstance(stream, dict):
        raise ValueError('stream must be an object')
    stream_start = _finite_float(stream.get('start_s'), field='stream.start_s')
    stream_end = _finite_float(stream.get('end_s'), field='stream.end_s')
    fps = _finite_float(stream.get('fps'), field='stream.fps')
    if stream_start > stream_end:
        raise ValueError('stream interval is reversed')
    if fps <= 0:
        raise ValueError('stream.fps must be positive')

    windows = sample['gt_windows']
    if not isinstance(windows, list) or not windows:
        raise ValueError('gt_windows must be a non-empty list')
    normalized = []
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f'gt_windows[{index}] must be an object')
        canonical = _canonical_window(
            window.get('start_s'),
            window.get('end_s'),
            field=f'gt_windows[{index}]',
        )
        if canonical['end_s'] < stream_start \
                or canonical['start_s'] > stream_end:
            raise ValueError(f'gt_windows[{index}] does not overlap the stream interval')
        normalized.append((canonical['start_s'], canonical['end_s']))
    if len(set(normalized)) != len(normalized) and sample['task'] != 'REC':
        raise ValueError('only REC may contain simultaneous duplicate event windows')
    if normalized != sorted(normalized):
        raise ValueError('gt_windows must be sorted chronologically')


def validate_samples(samples: Iterable[dict]) -> list[dict]:
    rows = list(samples)
    seen = set()
    for sample in rows:
        validate_sample(sample)
        sample_id = sample['sample_id']
        if sample_id in seen:
            raise ValueError(f'duplicate sample_id: {sample_id}')
        seen.add(sample_id)
    return rows


def frame_times(sample: dict) -> list[float]:
    """Return the canonical inclusive sampling timeline for one stream."""
    validate_sample(sample)
    stream = sample['stream']
    start = float(stream['start_s'])
    end = float(stream['end_s'])
    step = 1.0 / float(stream['fps'])
    count = int(math.floor((end - start) / step + 1e-9)) + 1
    return [start + index * step for index in range(count)]


def timing_profile(sample: dict) -> str:
    """Return the source-agnostic event profile used by data ablations."""
    validate_sample(sample)
    stream_start = float(sample['stream']['start_s'])
    stream_end = float(sample['stream']['end_s'])
    first = sample['gt_windows'][0]
    first_midpoint = (float(first['start_s']) + float(first['end_s'])) / 2.0
    span = max(stream_end - stream_start, 1e-12)
    position = min(1.0, max(0.0, (first_midpoint - stream_start) / span))
    position_bin = 'early' if position < 1 / 3 else 'middle' if position < 2 / 3 else 'late'
    count_bin = 'single' if len(sample['gt_windows']) == 1 else 'multi'
    return f'{position_bin}_{count_bin}'


def behavior_metadata(sample: dict) -> dict:
    """Return canonical sampling labels for one segmented training clip.

    The definition is source-independent and uses the actual inclusive frame
    timeline.  This keeps a 60-frame 1-fps clip at 60 seconds rather than the
    59-second distance between its first and last timestamps.
    """
    validate_sample(sample)
    times = frame_times(sample)
    frame_period = 1.0 / float(sample['stream']['fps'])
    duration_s = times[-1] - times[0] + frame_period
    windows = sample['gt_windows']
    post_event_tail_s = max(0.0, times[-1] - float(windows[-1]['end_s']))
    event_rate_per_minute = 60.0 * len(windows) / duration_s
    if len(windows) == 1:
        category = 'single_long_tail' if post_event_tail_s >= 10.0 else 'single_short_tail'
    else:
        category = 'multi_sparse' if event_rate_per_minute <= 6.0 else 'multi_dense'
    return {
        'frame_count': len(times),
        'duration_s': duration_s,
        'event_count': len(windows),
        'post_event_tail_s': post_event_tail_s,
        'event_rate_per_minute': event_rate_per_minute,
        'behavior_category': category,
    }


def behavior_category(sample: dict) -> str:
    """Return the canonical behavior class for one sampled clip."""
    return behavior_metadata(sample)['behavior_category']


def training_clip_catalog(samples: Iterable[dict], max_frames: int) -> list[dict]:
    """Label every deterministic training clip for sampling and audit."""
    rows = []
    for source_row_index, sample in enumerate(validate_samples(samples)):
        for clip in segment_for_training(sample, max_frames):
            labels = behavior_metadata(clip)
            rows.append({
                'pool_case_id': len(rows),
                'source_row_index': source_row_index,
                'sample_id': clip['sample_id'],
                'parent_sample_id': clip.get('parent_sample_id', clip['sample_id']),
                'segment_index': int(clip.get('segment_index', 0)),
                'source': clip['source'],
                'dataset': clip['source'],
                'task': clip['task'],
                'timing_profile': timing_profile(clip),
                **labels,
                'work_units': labels['frame_count']**2,
            })
    return rows


def segment_for_training(sample: dict, max_frames: int) -> list[dict]:
    """Split a sample into bounded clips without dropping any GT window.

    Actor replay retains every image until backward, so arbitrarily long
    benchmark streams are not a safe training unit. Windows that fit together
    are grouped into one clip; every window appears in exactly one clip and
    absolute timestamps are preserved for reward computation.
    """
    validate_sample(sample)
    if max_frames < 1:
        raise ValueError('training max_frames must be >= 1')
    fps = float(sample['stream']['fps'])
    max_span = (max_frames - 1) / fps
    groups: list[list[int]] = []
    for index, window in enumerate(sample['gt_windows']):
        group_start = None if not groups else sample['gt_windows'][groups[-1][0]]['start_s']
        if group_start is None or window['end_s'] - group_start > max_span + 1e-9:
            groups.append([index])
        else:
            groups[-1].append(index)

    stream_start = float(sample['stream']['start_s'])
    stream_end = float(sample['stream']['end_s'])
    clips = []
    for segment_index, target_indices in enumerate(groups):
        group = [sample['gt_windows'][index] for index in target_indices]
        if group[0]['end_s'] - group[0]['start_s'] > max_span + 1e-9:
            # Some PVQA reply spans are wider than the validated actor
            # horizon. Keep one bounded portion of that same positive window;
            # the original absolute window remains the reward target.
            start = max(stream_start, group[0]['start_s'])
            end = min(stream_end, start + max_span)
            group = group[:1]
        else:
            # Keep as much causal history as possible before the last rewarded
            # boundary while retaining the whole group inside the clip.
            end = min(stream_end, max(group[-1]['end_s'], stream_start + max_span))
            start = max(stream_start, end - max_span)
            if group[0]['start_s'] < start - 1e-9:
                start = group[0]['start_s']
                end = min(stream_end, start + max_span)
            if group[-1]['end_s'] > end + 1e-9:
                raise RuntimeError('training segmentation failed to retain a GT window')
        clip = {
            **sample,
            'sample_id': f'{sample["sample_id"]}#segment-{segment_index}',
            'stream': {
                **sample['stream'],
                'start_s': start,
                'end_s': end,
            },
            'gt_windows': [dict(window) for window in group],
            'parent_sample_id': sample['sample_id'],
            'segment_index': segment_index,
        }
        validate_sample(clip)
        if len(frame_times(clip)) > max_frames:
            raise RuntimeError('training segment exceeds max_frames')
        clips.append(clip)
    return clips


def adapt_streamingbench_po(
    csv_path: str | Path,
    video_roots: Sequence[str | Path],
) -> list[dict]:
    from examples.infer.proactive_eval.data_streamingbench import load_streamingbench_po

    roots = [Path(root).expanduser().resolve() for root in video_roots]
    if not roots:
        raise ValueError('StreamingBench requires at least one video root')
    samples = []
    for question in load_streamingbench_po(
            str(csv_path),
        [str(root) for root in roots],
            require_video=True,
    ):
        match = re.search(r'sample_(\d+)_\d+$', question.question_id)
        if match is None:
            raise ValueError(f'illegal StreamingBench question_id: {question.question_id!r}')
        sample_number = match.group(1)
        gt = float(question.gt_time_s)
        samples.append(
            _make_sample(
                sample_id=f'sb_po:{question.question_id}',
                source='sb_po',
                task='PO',
                query=question.question,
                video_path=_relative_video_path(question.video_path, roots),
                video_group=f'sb_po:sample_{sample_number}',
                stream_start_s=float(question.ask_time_s),
                stream_end_s=gt + 4.0,
                gt_windows=[_canonical_window(gt - 2.0, gt + 2.0, field='sb_po.gt_window')],
            ))
    return validate_samples(samples)


def adapt_ovo_far(
    json_path: str | Path,
    video_root: str | Path,
    *,
    tasks: Sequence[str] = ('CRR', 'REC', 'SSR'),
) -> list[dict]:
    from examples.infer.proactive_eval.data_ovo import load_ovo_far

    normalized_tasks = tuple(str(task).upper() for task in tasks)
    invalid_tasks = sorted(set(normalized_tasks) - {'CRR', 'REC', 'SSR'})
    if not normalized_tasks or invalid_tasks:
        raise ValueError(f'OVO tasks must be a non-empty subset of CRR/REC/SSR, got {normalized_tasks}')
    root = Path(video_root).expanduser().resolve()
    samples = []
    for question in load_ovo_far(
            str(json_path),
            str(root),
            tasks=normalized_tasks,
            require_video=True,
    ):
        source = f'ovo_{question.task.lower()}'
        windows = [
            _canonical_window(
                float(center) - 2.0,
                float(center) + 2.0,
                field=f'{source}.{question.qid}.gt_window',
            ) for center in question.gt_timestamps
        ]
        relative_path = _relative_video_path(question.video_path, [root])
        samples.append(
            _make_sample(
                sample_id=f'{source}:{question.qid}',
                source=source,
                task=question.task,
                query=question.question,
                video_path=relative_path,
                video_group=f'{source}:{relative_path}',
                stream_start_s=float(question.ask_time_s),
                stream_end_s=float(question.end_time_s) + 2.0,
                gt_windows=windows,
            ))
    return validate_samples(samples)


def adapt_ovo_ssr(
    json_path: str | Path,
    video_root: str | Path,
) -> list[dict]:
    """Backward-compatible SSR-only adapter."""
    return adapt_ovo_far(json_path, video_root, tasks=('SSR', ))


def adapt_pvqa(
    json_path: str | Path,
    video_root: str | Path,
    *,
    source: str,
) -> list[dict]:
    from examples.infer.proactive_eval.data_pvqa import load_pvqa

    if source not in {'pvqa_ego', 'pvqa_web', 'pvqa_vad'}:
        raise ValueError(f'unsupported PVQA source: {source!r}')
    root = Path(video_root).expanduser().resolve()
    samples = []
    for question in load_pvqa(
            str(json_path),
            str(root),
            require_video=True,
    ):
        relative_path = _relative_video_path(question.video_path, [root])
        windows = [
            _canonical_window(
                answer.get('reply_timespan', [None, None])[0],
                answer.get('reply_timespan', [None, None])[1],
                field=f'{source}.{question.question_id}.reply_timespan',
            ) for answer in question.gt_answers
        ]
        samples.append(
            _make_sample(
                sample_id=f'{source}:{question.question_id}',
                source=source,
                task='PVQA',
                query=question.query,
                video_path=relative_path,
                video_group=f'{source}:{relative_path}',
                stream_start_s=max(0.0, float(question.start_time_s)),
                stream_end_s=min(
                    math.ceil(float(question.last_reply_end_s)) + 1.0,
                    float(question.duration_s),
                ),
                gt_windows=windows,
            ))
    return validate_samples(samples)


def _group_rank(seed: int, source: str, group_id: str) -> bytes:
    payload = f'{seed}\0{source}\0{group_id}'.encode('utf-8')
    return hashlib.sha256(payload).digest()


def split_by_video(
    samples: Iterable[dict],
    *,
    val_per_source: int = DEFAULT_VAL_PER_SOURCE,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], list[dict]]:
    rows = validate_samples(samples)
    if val_per_source < 1:
        raise ValueError('val_per_source must be >= 1')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['source']][row['video']['group_id']].append(row)

    val_groups = set()
    for source, groups in grouped.items():
        if len(groups) < 2:
            raise ValueError(f'source {source} needs at least two video groups to split')
        selected_rows = 0
        selected_groups = 0
        ranked = sorted(
            groups,
            key=lambda group: (_group_rank(seed, source, group), group),
        )
        for group in ranked:
            if selected_rows >= val_per_source:
                break
            if selected_groups == len(groups) - 1:
                break
            val_groups.add((source, group))
            selected_rows += len(groups[group])
            selected_groups += 1
        if selected_rows < val_per_source:
            raise ValueError(f'source {source} cannot provide {val_per_source} validation '
                             'rows while retaining a training video group')

    train, val = [], []
    for row in rows:
        key = (row['source'], row['video']['group_id'])
        (val if key in val_groups else train).append(row)
    order = lambda row: (row['source'], row['sample_id'])
    train.sort(key=order)
    val.sort(key=order)

    train_ids = {row['sample_id'] for row in train}
    val_ids = {row['sample_id'] for row in val}
    if train_ids & val_ids or len(train) + len(val) != len(rows):
        raise RuntimeError('split lost or duplicated samples')
    train_groups = {(row['source'], row['video']['group_id']) for row in train}
    held_out_groups = {(row['source'], row['video']['group_id']) for row in val}
    if train_groups & held_out_groups:
        raise RuntimeError('train/validation video-group leakage detected')
    return train, val


def _json_line(value: dict) -> bytes:
    return (json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ) + '\n').encode('utf-8')


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_split(path: str | Path, manifest_path: str | Path) -> tuple[list[dict], dict]:
    """Load one canonical split and verify bytes/counts against its manifest."""
    path = Path(path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'timing split does not exist: {path}')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'timing manifest does not exist: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'unsupported timing manifest: {manifest_path}')
    expected = manifest.get('outputs', {}).get(path.name)
    if not isinstance(expected, dict):
        raise ValueError(f'manifest does not describe split {path.name!r}')
    if file_sha256(path) != expected.get('sha256'):
        raise ValueError(f'timing split SHA256 differs from manifest: {path}')
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid JSON at {path}:{line_number}') from exc
    validate_samples(rows)
    if len(rows) != int(expected.get('rows', -1)):
        raise ValueError(f'timing split row count differs from manifest: {path}')
    return rows, manifest


def write_training_clip_catalog(
    samples: Iterable[dict],
    output_path: str | Path,
    catalog_manifest_path: str | Path,
    *,
    source_path: str | Path,
    source_manifest_path: str | Path,
    max_frames: int,
) -> dict:
    """Write the immutable labeled clip index and its provenance manifest."""
    output_path = Path(output_path).expanduser()
    catalog_manifest_path = Path(catalog_manifest_path).expanduser()
    source_path = Path(source_path).expanduser().resolve()
    source_manifest_path = Path(source_manifest_path).expanduser().resolve()
    rows = training_clip_catalog(samples, max_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b''.join(_json_line(row) for row in rows))
    manifest = {
        'schema_version': CLIP_CATALOG_SCHEMA_VERSION,
        'max_frames': max_frames,
        'source': {
            'name': source_path.name,
            'sha256': file_sha256(source_path),
            'manifest_name': source_manifest_path.name,
            'manifest_sha256': file_sha256(source_manifest_path),
        },
        'output': {
            'name': output_path.name,
            'rows': len(rows),
            'sha256': file_sha256(output_path),
        },
    }
    catalog_manifest_path.write_bytes(_json_line(manifest))
    return manifest


def load_verified_clip_catalog(
    path: str | Path,
    manifest_path: str | Path,
) -> tuple[list[dict], dict]:
    """Load a labeled clip catalog and fail closed on bytes or schema drift."""
    path = Path(path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != CLIP_CATALOG_SCHEMA_VERSION:
        raise ValueError(f'unsupported clip catalog manifest: {manifest_path}')
    expected = manifest.get('output', {})
    if expected.get('name') != path.name or expected.get('sha256') != file_sha256(path):
        raise ValueError(f'clip catalog SHA256 differs from manifest: {path}')
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid JSON at {path}:{line_number}') from exc
            required = {
                'pool_case_id', 'source_row_index', 'sample_id', 'parent_sample_id', 'segment_index',
                'source', 'dataset', 'task', 'timing_profile', 'frame_count', 'duration_s', 'event_count',
                'post_event_tail_s', 'event_rate_per_minute', 'behavior_category', 'work_units',
            }
            if required - row.keys() or row.get('pool_case_id') != len(rows):
                raise ValueError(f'invalid clip catalog row at {path}:{line_number}')
            if row['behavior_category'] not in BEHAVIOR_CATEGORIES or row['source'] != row['dataset']:
                raise ValueError(f'invalid clip catalog labels at {path}:{line_number}')
            rows.append(row)
    if len(rows) != int(expected.get('rows', -1)):
        raise ValueError(f'clip catalog row count differs from manifest: {path}')
    return rows, manifest


def write_splits(
    samples: Iterable[dict],
    output_dir: str | Path,
    *,
    input_paths: Sequence[str | Path],
    video_roots: dict[str, Sequence[str | Path]] | None = None,
    val_per_source: int = DEFAULT_VAL_PER_SOURCE,
    seed: int = DEFAULT_SEED,
) -> dict:
    rows = validate_samples(samples)
    train, val = split_by_video(rows, val_per_source=val_per_source, seed=seed)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / 'train.jsonl'
    val_path = output / 'val.jsonl'
    train_path.write_bytes(b''.join(_json_line(row) for row in train))
    val_path.write_bytes(b''.join(_json_line(row) for row in val))

    input_rows = []
    for value in input_paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'input metadata does not exist: {path}')
        input_rows.append({'name': path.name, 'sha256': file_sha256(path)})
    input_rows.sort(key=lambda row: (row['name'], row['sha256']))

    by_source = {}
    for source in sorted({row['source'] for row in rows}):
        source_rows = [row for row in rows if row['source'] == source]
        source_train = [row for row in train if row['source'] == source]
        source_val = [row for row in val if row['source'] == source]
        by_source[source] = {
            'rows': len(source_rows),
            'video_groups': len({row['video']['group_id']
                                 for row in source_rows}),
            'train_rows': len(source_train),
            'train_video_groups': len({row['video']['group_id']
                                       for row in source_train}),
            'val_rows': len(source_val),
            'val_video_groups': len({row['video']['group_id']
                                     for row in source_val}),
        }
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'seed': seed,
        'val_per_source_target': val_per_source,
        'inputs': input_rows,
        'video_roots': {
            source: [str(Path(root).expanduser().resolve()) for root in roots]
            for source, roots in sorted((video_roots or {}).items())
        },
        'sources': by_source,
        'outputs': {
            'train.jsonl': {
                'rows': len(train),
                'sha256': file_sha256(train_path)
            },
            'val.jsonl': {
                'rows': len(val),
                'sha256': file_sha256(val_path)
            },
        },
    }
    manifest_path = output / 'manifest.json'
    manifest_path.write_bytes(_json_line(manifest))
    return manifest
