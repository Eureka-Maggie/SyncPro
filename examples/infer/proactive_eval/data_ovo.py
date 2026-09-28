"""Load OVO-Bench FAR subset (CRR / REC / SSR) questions + videos.

Annotation: ``OVO-Bench/data/ovo_bench_new.json`` (1640 entries; 172 FAR).
Videos: ``ovo_bench/src_videos/<dataset>/<basename>.mp4`` after untarring
``src_videos.tar.part*``.

Per-task fields and question phrasing follow Em-Garde
``eval/ovo_bench/ovo_bench_eval_online.py`` so the comparison is apples-to-
apples — only the trigger mechanism (our SPEAK token vs Em-Garde's embedding
detector) differs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


@dataclass
class OVOQuestion:
    qid: str  # int id from json, kept as str for jsonl
    task: str  # CRR | REC | SSR
    video_path: str  # absolute path to *.mp4
    question: str  # natural-language query fed to the model
    ask_time_s: int  # first frame we start streaming from
    end_time_s: int  # stop streaming at (from last testpoint)
    gt_timestamps: List[float]  # ground-truth response times (0..N)
    gt_contents: List[str]  # reference answer aligned with each timestamp


def _per_task(item: dict, video_root: str) -> Optional[OVOQuestion]:
    task = item['task']
    if task not in ('CRR', 'REC', 'SSR'):
        return None
    video_rel = item['video']
    video_path = os.path.join(video_root, video_rel)

    test_info = item.get('test_info', [])
    end_time = int(test_info[-1]['realtime']) if test_info else 0

    if task == 'CRR':
        question = item['question']
        ask = int(item['ask_time'])
        gts = [float(item['clue_time'])]
        contents = [str(item['answer']).strip()]
    elif task == 'REC':
        question = f"How many times does the event: {item['activity']} happen?"
        ask = 0
        gts = [float(t) + 2.0 for t in item['start_times']]
        contents = [str(index) for index in range(1, len(gts) + 1)]
    elif task == 'SSR':
        # Em-Garde inserts spaces before each capital and lowercases the
        # tutorial slug.
        tutorial = re.sub(r'(?<!^)(?=[A-Z])', ' ', item['tutorial']).lower()
        question = f'Describe the steps of {tutorial}'
        ask = 0
        gts = [float(t) + 2.0 for t in item['start_time']]
        contents = [str(step).strip() for step in item['all_steps']]
    else:
        return None

    if len(contents) != len(gts) or any(not content for content in contents):
        raise ValueError(f'{task} id={item["id"]} has unaligned/empty content targets')
    return OVOQuestion(
        qid=str(item['id']),
        task=task,
        video_path=video_path,
        question=question,
        ask_time_s=ask,
        end_time_s=end_time,
        gt_timestamps=gts,
        gt_contents=contents,
    )


def load_ovo_far(
        json_path: str,
        video_root: str,
        *,
        tasks: tuple = ('CRR', 'REC', 'SSR'),
        require_video: bool = True,
) -> Iterator[OVOQuestion]:
    import sys
    with open(json_path) as f:
        data = json.load(f)
    for item in data:
        if item['task'] not in tasks:
            continue
        q = _per_task(item, video_root)
        if q is None:
            continue
        if require_video and not os.path.exists(q.video_path):
            print(f'  [skip] no video for {q.qid} ({q.task}): {q.video_path}', file=sys.stderr)
            continue
        yield q
