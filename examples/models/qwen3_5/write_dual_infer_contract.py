#!/usr/bin/env python3
"""Write or validate the final split-loss+ATR SFT runtime contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def _bool01(raw: str) -> bool:
    if raw in {'1', 'true', 'True'}:
        return True
    if raw in {'0', 'false', 'False'}:
        return False
    raise argparse.ArgumentTypeError(f'expected 0/1 or true/false, got {raw!r}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--rank', required=True, type=int)
    parser.add_argument('--alpha', required=True, type=float)
    parser.add_argument('--dropout', required=True, type=float)
    parser.add_argument('--merge-mode', required=True, choices=('additive', ))
    parser.add_argument('--merge-gamma', required=True, type=float)
    parser.add_argument('--merge-gate-mode', required=True, choices=('shared', ))
    parser.add_argument('--position-overlap', required=True, type=_bool01)
    parser.add_argument('--source-fork', required=True, type=_bool01)
    parser.add_argument('--step-mask-window', required=True, type=int)
    parser.add_argument('--trainable-regex', required=True)
    parser.add_argument('--system-prompt', required=True, type=Path)
    parser.add_argument('--first-silent-weight', required=True, type=float)
    parser.add_argument('--special-inner-weight', required=True, type=float)
    parser.add_argument('--special-silent-weight', required=True, type=float)
    parser.add_argument('--special-speak-weight', required=True, type=float)
    parser.add_argument('--action-loss-mode', required=True, choices=('restricted_binary', ))
    parser.add_argument('--a-temporal-rank-lambda', required=True, type=float)
    parser.add_argument('--a-temporal-rank-margin', required=True, type=float)
    parser.add_argument('--a-temporal-rank-radius', required=True, type=int)
    parser.add_argument('--a-temporal-rank-hard-negatives', required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finite = {
        'alpha': args.alpha,
        'dropout': args.dropout,
        'merge_gamma': args.merge_gamma,
        'first_silent_weight': args.first_silent_weight,
        'special_inner_weight': args.special_inner_weight,
        'special_silent_weight': args.special_silent_weight,
        'special_speak_weight': args.special_speak_weight,
        'a_temporal_rank_lambda': args.a_temporal_rank_lambda,
        'a_temporal_rank_margin': args.a_temporal_rank_margin,
    }
    invalid = [name for name, value in finite.items() if not math.isfinite(value)]
    if invalid:
        raise SystemExit(f'non-finite contract values: {invalid}')
    if args.rank < 1 or args.alpha <= 0 or not 0 <= args.dropout < 1:
        raise SystemExit('rank/alpha must be positive and dropout must be in [0, 1)')
    if not 0 < args.merge_gamma < 1:
        raise SystemExit('merge gamma must be in (0, 1)')
    if args.step_mask_window < 1:
        raise SystemExit('step mask window must be positive')
    if args.a_temporal_rank_lambda <= 0 or args.a_temporal_rank_margin < 0:
        raise SystemExit('ATR lambda must be positive and margin non-negative')
    if args.a_temporal_rank_radius < 1 or args.a_temporal_rank_hard_negatives < 1:
        raise SystemExit('ATR radius and hard-negative count must be positive')
    if 'gamma_logit' in args.trainable_regex or 'dual_merge' in args.trainable_regex:
        raise SystemExit('final fixed-gamma recipe requires the merge gate to stay frozen')
    if args.merge_gamma != 0.5:
        raise SystemExit('final fixed-gamma recipe requires merge gamma=0.5')

    prompt_bytes = args.system_prompt.read_bytes()
    contract = {
        'contract_version': 3,
        'architecture': 'qwen3_5_dual_source_fork',
        'rank': args.rank,
        'alpha': args.alpha,
        'dropout': args.dropout,
        'merge_mode': 'additive',
        'merge_policy': 'fixed0p5',
        'position_overlap': args.position_overlap,
        'source_fork': args.source_fork,
        'step_mask_window': args.step_mask_window,
        'trainable_parameters_regex': args.trainable_regex,
        'system_prompt_sha256': hashlib.sha256(prompt_bytes).hexdigest(),
        'system_prompt_bytes': len(prompt_bytes),
        'system_prompt_terminal_newline': prompt_bytes.endswith(b'\n'),
        'first_silent_weight': args.first_silent_weight,
        'special_token_weights': {
            '<inner>': args.special_inner_weight,
            '<silent>': args.special_silent_weight,
            '<speak>': args.special_speak_weight,
        },
        'action_loss_mode': 'restricted_binary',
        'action_space': 'dual_grammar_v1',
        'requires_constrained_decode': True,
        'a_temporal_rank': {
            'lambda': args.a_temporal_rank_lambda,
            'margin': args.a_temporal_rank_margin,
            'radius': args.a_temporal_rank_radius,
            'hard_negatives': args.a_temporal_rank_hard_negatives,
            'distance': 'a_action_decisions',
            'negative_selection': 'highest_speak_margin',
            'normalization': 'mean_per_positive',
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / 'dual_infer_contract.json'
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding='utf-8'))
        if existing != contract:
            raise SystemExit('refusing incompatible dual checkpoint resume:\n'
                             f'  existing={json.dumps(existing, sort_keys=True)}\n'
                             f'  requested={json.dumps(contract, sort_keys=True)}')
        print(f'[dual-contract] unchanged: {destination}')
        return

    payload = json.dumps(contract, indent=2, sort_keys=True) + '\n'
    fd, temporary = tempfile.mkstemp(prefix='.dual_infer_contract.', suffix='.tmp', dir=args.output_dir)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'[dual-contract] wrote: {destination}')


if __name__ == '__main__':
    main()
