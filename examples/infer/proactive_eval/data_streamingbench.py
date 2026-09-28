"""Load StreamingBench PO questions + their videos.

Each row in ``Proactive_Output.csv`` is one question (video, ask time, gt
time, gt output string). Videos live as zips at the dataset root and unzip
into a single ``Proactive Output_*/*_proactive.mp4`` per row.

We don't reformulate the question — the model sees the raw ``question`` text.
Reformulation (e.g. stripping "output X") can be revisited if smoke tests
show the train-vs-eval phrasing gap is hurting recall.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass
class SBQuestion:
    question_id: str
    task_type: str  # "Proactive Output"
    question: str  # raw natural-language query, fed to the model as-is
    ask_time_s: int  # time the question is asked (start streaming here)
    gt_time_s: int  # ground-truth time the response should fire
    gt_output: str  # expected response string (not scored by Recall@2s)
    temporal_clue_type: str  # Subsequent | Concurrent | Antecedent
    frames_required: str  # single | multi
    video_path: str  # absolute path to *_proactive.mp4


def _hms_to_sec(s: str) -> int:
    """Convert 'HH:MM:SS' or 'MM:SS' to integer seconds."""
    return sum(int(x) * 60**i for i, x in enumerate(reversed(s.split(':'))))


def _find_video_for_qid(qid: str, video_search_roots: List[str]) -> Optional[str]:
    """Locate ``sample_<N>/video.mp4`` for ``qid``.

    Question IDs look like ``Proactive Output_sample_3_2`` (sample 3, sub-q 2).
    The Proactive_Output zips unpack to ``sample_N/video.mp4``. We extract
    the sample number from the qid and probe each candidate root.
    """
    m = re.search(r'sample_(\d+)_\d+$', qid)
    if not m:
        return None
    n = m.group(1)
    # Mostly ``sample_N/video.mp4``; a handful of samples (45/48/50 at least)
    # keep the legacy ``sample_N/Active Output_<X>.mp4`` naming.
    for root in video_search_roots:
        d = os.path.join(root, f'sample_{n}')
        if not os.path.isdir(d):
            continue
        cand = os.path.join(d, 'video.mp4')
        if os.path.exists(cand):
            return cand
        for fn in os.listdir(d):
            if fn.lower().endswith('.mp4'):
                return os.path.join(d, fn)
    return None


def load_streamingbench_po(
    csv_path: str,
    video_search_roots: List[str],
    *,
    require_video: bool = True,
) -> Iterator[SBQuestion]:
    """Yield SBQuestion rows; skip questions whose video isn't found if
    ``require_video=True`` (after warning to stderr)."""
    import sys
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row['question_id']
            video = _find_video_for_qid(qid, video_search_roots)
            if video is None:
                if require_video:
                    print(f'  [skip] no video for {qid}', file=sys.stderr)
                    continue
                video = ''
            yield SBQuestion(
                question_id=qid,
                task_type=row['task_type'],
                question=row['question'],
                ask_time_s=_hms_to_sec(row['time_stamp']),
                gt_time_s=_hms_to_sec(row['ground_truth_time_stamp']),
                gt_output=row['ground_truth_output'],
                temporal_clue_type=row['temporal_clue_type'],
                frames_required=row['frames_required'],
                video_path=video,
            )
