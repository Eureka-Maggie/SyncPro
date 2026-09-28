"""Load ProactiveVideoQA questions + their videos.

Annotation: ``proactive_video_qa/{EGO,WEB,VAD}/anno.json``. Schema per
record (only the fields we use):

  question_id  : str (matches video basename)
  video        : str (basename of mp4 inside the dataset's ``videos/``)
  conversation : [{'role': 'user', 'content': query, 'time': start_s}]
  duration     : float (seconds)
  answer       : [{'content': gt_text, 'reply_timespan': [[a, b], ...]} ...]

We feed ``conversation[0]['content']`` as the query, stream from
``conversation[0]['time']`` to ``duration``, and dump per-frame SPEAK actions
so downstream PAUC@0.5 (computed with the official PVQA scorer) can match
each SPEAK time against ``reply_timespan`` envelopes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Tuple


@dataclass
class PVQAQuestion:
    question_id: str
    video_path: str
    query: str
    start_time_s: float
    duration_s: float
    last_reply_end_s: float  # max(reply_timespan[1] for all answers); past
    # this point the PAUC scorer ignores any response,
    # so we can early-stop streaming here (lossless).
    gt_answers: List[dict]  # raw answer list, kept for downstream PAUC


def load_pvqa(
    anno_path: str,
    video_root: str,
    *,
    require_video: bool = True,
) -> Iterator[PVQAQuestion]:
    """Yield PVQAQuestion records; skip missing videos with a warning."""
    import sys
    with open(anno_path) as f:
        data = json.load(f)
    for item in data:
        conv = item['conversation']
        if not conv or conv[0]['role'] != 'user':
            continue
        vp = os.path.join(video_root, item['video'])
        if require_video and not os.path.exists(vp):
            print(f'  [skip] no video for {item["question_id"]}: {vp}', file=sys.stderr)
            continue
        answers = item.get('answer', []) or []
        if answers:
            last_end = max(ans['reply_timespan'][1] for ans in answers)
        else:
            last_end = float(item['duration'])
        yield PVQAQuestion(
            question_id=str(item['question_id']),
            video_path=vp,
            query=conv[0]['content'],
            start_time_s=float(conv[0].get('time', 0)),
            duration_s=float(item['duration']),
            last_reply_end_s=float(last_end),
            gt_answers=answers,
        )


def load_pvqa_ego(
    anno_path: str,
    video_root: str,
    *,
    require_video: bool = True,
) -> Iterator[PVQAQuestion]:
    """Backward-compatible alias for older EGO-only call sites."""
    yield from load_pvqa(anno_path, video_root, require_video=require_video)
