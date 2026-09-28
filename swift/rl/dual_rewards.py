"""Hard-window F1 reward and target parsing for dual-stream trajectories."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

DUAL_TARGET_SEPARATOR = '<|im_end|>\n'
HARD_WINDOW_F1 = 'hard_window_f1_v1'
TIMING_REWARD_MODES = (HARD_WINDOW_F1, )

_RE_B_SILENT = re.compile(r'^\s*<silent>\s*$')
_RE_B_INNER = re.compile(r'^\s*<inner>(.+?)</inner>\s*$', re.DOTALL)
_RE_A_SILENT = re.compile(r'^\s*<silent>\s*$')
_RE_A_SPEAK = re.compile(r'^\s*<speak>(.+?)</speak>\s*$', re.DOTALL)


def _strip_terminal_im_end(text: str) -> str:
    text = text or ''
    if text.endswith('<|im_end|>'):
        text = text[:-len('<|im_end|>')]
    return text.strip()


def parse_dual_target(content: str) -> tuple[str, str]:
    """Parse one SFT target into private branch B and visible branch A."""
    if not isinstance(content, str):
        raise TypeError('dual assistant target must be a string')
    if content.count(DUAL_TARGET_SEPARATOR) != 1:
        raise ValueError('dual assistant target must contain exactly one '
                         f'{DUAL_TARGET_SEPARATOR!r}')
    branch_b, branch_a = (part.strip() for part in content.split(DUAL_TARGET_SEPARATOR))
    if not is_valid_branch_b(branch_b):
        raise ValueError(f'illegal branch-B target: {branch_b!r}')
    if not is_valid_branch_a(branch_a):
        raise ValueError(f'illegal branch-A target: {branch_a!r}')
    return branch_b, branch_a


def is_valid_branch_b(content: str) -> bool:
    return bool(_RE_B_SILENT.match(content or '') or _RE_B_INNER.match(content or ''))


def is_valid_branch_a(content: str) -> bool:
    return bool(_RE_A_SILENT.match(content or '') or _RE_A_SPEAK.match(content or ''))


def extract_dual_frame(frame: dict) -> tuple[str, str]:
    branch_b = _strip_terminal_im_end(frame.get('internal_response', ''))
    branch_a = _strip_terminal_im_end(frame.get('external_response', ''))
    return branch_b, branch_a


def dual_format_reward(frames: list[dict]) -> float:
    if not frames:
        return 0.0
    valid = 0
    for frame in frames:
        branch_b, branch_a = extract_dual_frame(frame)
        valid += int(is_valid_branch_b(branch_b) and is_valid_branch_a(branch_a))
    return valid / len(frames)


def _finite_float(value, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{field} must be finite')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be finite') from exc
    if not math.isfinite(result):
        raise ValueError(f'{field} must be finite')
    return result


def _chronological_window_matches(
    sorted_predictions: list[float],
    sorted_windows: list[tuple[float, float]],
) -> list[tuple[int, int]]:
    """Match each window to its first unused in-window prediction."""
    matches = []
    used_predictions = set()
    for window_index, (start, end) in enumerate(sorted_windows):
        for prediction_index, prediction in enumerate(sorted_predictions):
            if prediction_index in used_predictions:
                continue
            if start <= prediction <= end:
                used_predictions.add(prediction_index)
                matches.append((prediction_index, window_index))
                break
    return matches


def temporal_window_metrics(speak_times: list[float], gt_windows: list[dict]) -> dict:
    """Compute the selected chronological one-to-one hard-window F1 reward."""
    if not isinstance(gt_windows, list) or not gt_windows:
        raise ValueError('temporal F1 requires at least one gt window')
    predictions = sorted(
        _finite_float(value, field=f'speak_times[{index}]')
        for index, value in enumerate(speak_times)
    )
    windows = []
    for index, window in enumerate(gt_windows):
        if not isinstance(window, dict):
            raise ValueError(f'gt_windows[{index}] must be an object')
        start = _finite_float(window.get('start_s'), field=f'gt_windows[{index}].start_s')
        end = _finite_float(window.get('end_s'), field=f'gt_windows[{index}].end_s')
        if start > end:
            raise ValueError(f'gt_windows[{index}] is reversed')
        windows.append((start, end))
    windows.sort()

    matches = _chronological_window_matches(predictions, windows)
    matched_predictions = {prediction_index for prediction_index, _ in matches}
    true_positive = len(matched_predictions)
    prediction_count = len(predictions)
    target_count = len(windows)
    false_positive = prediction_count - true_positive
    false_negative = target_count - true_positive
    precision = true_positive / prediction_count if prediction_count else 0.0
    recall = true_positive / target_count
    f1 = 2.0 * true_positive / (target_count + prediction_count)

    duplicate = 0
    out_of_window = 0
    for prediction_index, prediction in enumerate(predictions):
        if prediction_index in matched_predictions:
            continue
        if any(start <= prediction <= end for start, end in windows):
            duplicate += 1
        else:
            out_of_window += 1
    first_accuracy = float(bool(predictions) and any(start <= predictions[0] <= end for start, end in windows))
    return {
        'tp': true_positive,
        'fp': false_positive,
        'fn': false_negative,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'first_speak_accuracy': first_accuracy,
        'speak_count': prediction_count,
        'silent': float(not predictions),
        'out_of_window_speak_count': out_of_window,
        'duplicate_speak_count': duplicate,
    }


def derive_dual_gt(messages: list[dict], *, max_frames: Optional[int] = None) -> dict:
    assistant_messages = [message for message in messages if message.get('role') == 'assistant']
    if max_frames is not None:
        assistant_messages = assistant_messages[:max_frames]
    event_times = []
    frame_targets = []
    for frame_index, message in enumerate(assistant_messages):
        branch_b, branch_a = parse_dual_target(message.get('content'))
        mode = 'SPEAK' if _RE_A_SPEAK.match(branch_a) else 'SILENT'
        if mode == 'SPEAK':
            event_times.append(float(frame_index))
        frame_targets.append({'frame_idx': frame_index, 'branch_b': branch_b, 'branch_a': branch_a, 'mode': mode})
    return {'gt_event_times': event_times, 'frame_targets': frame_targets}


@dataclass
class DualRewardWeights:
    format: float = 0.0
    timing: float = 1.0
    sigma: float = 2.0
    timing_mode: str = HARD_WINDOW_F1


def aggregate_dual_reward(
    frames: list[dict],
    gt: dict,
    weights: Optional[DualRewardWeights] = None,
) -> dict:
    """Score a rollout with the selected timing-only hard-window F1 objective."""
    weights = weights or DualRewardWeights()
    if weights.timing_mode != HARD_WINDOW_F1:
        raise ValueError(f'unsupported timing reward mode: {weights.timing_mode!r}')
    external_frames = []
    for frame_index, frame in enumerate(frames):
        _, branch_a = extract_dual_frame(frame)
        external_frames.append({
            'frame_idx': float(frame.get('t', frame.get('frame_idx', frame_index))),
            'mode': 'SPEAK' if _RE_A_SPEAK.match(branch_a) else 'SILENT',
        })
    fmt = dual_format_reward(frames)
    window_metrics = None
    if 'gt_windows' not in gt:
        raise ValueError('hard-window F1 requires gt_windows')
    speak_times = [frame['frame_idx'] for frame in external_frames if frame['mode'] == 'SPEAK']
    window_metrics = temporal_window_metrics(speak_times, gt['gt_windows'])
    timing = window_metrics['f1']
    total = weights.format * fmt + weights.timing * timing
    result = {
        'total': total,
        'format': fmt,
        'timing': timing,
        'speak_count_mul': 1.0,
        'shaped_timing': timing,
        'speak_count': sum(frame['mode'] == 'SPEAK' for frame in external_frames),
        'silent': float(not any(frame['mode'] == 'SPEAK' for frame in external_frames)),
    }
    result.update(window_metrics)
    return result
