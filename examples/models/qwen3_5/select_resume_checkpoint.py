#!/usr/bin/env python3
"""Select the highest complete, architecture-compatible SFT checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CHECKPOINT_RE = re.compile(r'checkpoint-(\d+)$')
VERSION_RE = re.compile(r'v(\d+)-')


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _load_json(path: Path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f'invalid checkpoint directory name: {path}')
    return int(match.group(1))


def _version(path: Path) -> int:
    match = VERSION_RE.match(path.parent.name)
    return int(match.group(1)) if match else 1_000_000_000


def _model_weight_errors(checkpoint: Path) -> list[str]:
    index_path = checkpoint / 'model.safetensors.index.json'
    if _nonempty(index_path):
        try:
            index = _load_json(index_path)
            shard_names = set(index.get('weight_map', {}).values())
        except Exception as exc:  # noqa: BLE001
            return [f'invalid model index: {exc}']
        if not shard_names:
            return ['model index contains no weight shards']
        missing = sorted(name for name in shard_names if not _nonempty(checkpoint / name))
        return [f'missing/empty model shard: {name}' for name in missing]

    for name in ('model.safetensors', 'pytorch_model.bin'):
        if _nonempty(checkpoint / name):
            return []
    return ['no complete model weights found']


def _merge_gate_layout_errors(checkpoint: Path, expected_mode: str | None) -> list[str]:
    """Reject shared-gamma/split-alpha-beta checkpoint topology changes.

    DeepSpeed restores module state strictly, so changing the merge gate layout
    cannot be treated as an ordinary optimizer resume.  A sharded full-model
    checkpoint exposes all tensor names in its safetensors index, which lets us
    catch the mismatch before distributed workers and the model are started.
    ``args.json`` is a fallback for unusual checkpoint serialization layouts.
    """
    if expected_mode is None:
        return []

    weight_names: set[str] | None = None
    index_path = checkpoint / 'model.safetensors.index.json'
    if _nonempty(index_path):
        try:
            index = _load_json(index_path)
            weight_names = set(index.get('weight_map', {}))
        except Exception as exc:  # noqa: BLE001
            return [f'cannot inspect merge gate layout: invalid model index: {exc}']

    actual_mode: str | None = None
    if weight_names is not None:
        gate_names = {
            name.rsplit('.', 1)[-1]
            for name in weight_names
            if '.dual_merge.' in name
        }
        if gate_names == {'gamma_logit'}:
            actual_mode = 'shared'
        elif gate_names == {'alpha_logit', 'beta_logit'}:
            actual_mode = 'split'
        elif gate_names:
            return [f'invalid merge gate tensor layout: {sorted(gate_names)}']

    if actual_mode is None:
        args_path = checkpoint / 'args.json'
        if _nonempty(args_path):
            try:
                trainable = str(_load_json(args_path).get('trainable_parameters_regex') or '')
            except Exception as exc:  # noqa: BLE001
                return [f'cannot inspect merge gate layout: invalid args.json: {exc}']
            has_alpha = 'alpha_logit' in trainable
            has_beta = 'beta_logit' in trainable
            has_gamma = 'gamma_logit' in trainable
            if has_alpha or has_beta:
                if not (has_alpha and has_beta) or has_gamma:
                    return ['invalid merge gate names in trainable_parameters_regex']
                actual_mode = 'split'
            elif has_gamma:
                actual_mode = 'shared'

    if actual_mode is not None and actual_mode != expected_mode:
        return [
            f'merge gate topology mismatch: checkpoint={actual_mode}, current={expected_mode}; '
            'start a new OUTPUT_DIR (or set AUTO_RESUME=0 with an empty OUTPUT_DIR)'
        ]
    return []


def _inspect_checkpoint(
    checkpoint: Path,
    world_size: int,
    expected_merge_gate_mode: str | None = None,
) -> tuple[int, list[str]]:
    step = _checkpoint_step(checkpoint)
    errors: list[str] = []

    trainer_state_path = checkpoint / 'trainer_state.json'
    if not _nonempty(trainer_state_path):
        errors.append('missing/empty trainer_state.json')
    else:
        try:
            trainer_step = int(_load_json(trainer_state_path).get('global_step', -1))
            if trainer_step != step:
                errors.append(f'trainer_state global_step={trainer_step}, expected {step}')
        except Exception as exc:  # noqa: BLE001
            errors.append(f'invalid trainer_state.json: {exc}')

    for name in ('scheduler.pt', 'training_args.bin', 'args.json'):
        if not _nonempty(checkpoint / name):
            errors.append(f'missing/empty {name}')

    latest_path = checkpoint / 'latest'
    expected_tag = f'global_step{step}'
    if not _nonempty(latest_path):
        errors.append('missing/empty DeepSpeed latest marker')
    elif latest_path.read_text(encoding='utf-8').strip() != expected_tag:
        errors.append(f'DeepSpeed latest marker does not point to {expected_tag}')

    state_dir = checkpoint / expected_tag
    if not state_dir.is_dir():
        errors.append(f'missing DeepSpeed state directory {expected_tag}')
    else:
        if not _nonempty(state_dir / 'mp_rank_00_model_states.pt'):
            errors.append('missing/empty DeepSpeed model state')
        optimizer_shards = list(state_dir.glob('*_zero_pp_rank_*_optim_states.pt'))
        if len(optimizer_shards) != world_size:
            errors.append(
                f'ZeRO optimizer world size mismatch: checkpoint={len(optimizer_shards)}, current={world_size}')
        elif any(not _nonempty(path) for path in optimizer_shards):
            errors.append('one or more ZeRO optimizer shards are empty')

    rng_pattern = 'rng_state.pth' if world_size == 1 else 'rng_state_*.pth'
    rng_states = list(checkpoint.glob(rng_pattern))
    expected_rng = 1 if world_size == 1 else world_size
    if len(rng_states) != expected_rng or any(not _nonempty(path) for path in rng_states):
        errors.append(f'RNG state count mismatch: checkpoint={len(rng_states)}, current={expected_rng}')

    errors.extend(_model_weight_errors(checkpoint))
    errors.extend(_merge_gate_layout_errors(checkpoint, expected_merge_gate_mode))
    return step, errors


def select_checkpoint(
    output_dir: Path,
    world_size: int,
    expected_merge_gate_mode: str | None = None,
) -> Path | None:
    candidates = sorted(
        set(output_dir.glob('checkpoint-*')) | set(output_dir.glob('v*-*/checkpoint-*')),
        key=str,
    )
    if not candidates:
        return None

    valid: list[tuple[int, int, Path]] = []
    invalid: list[tuple[int, Path, list[str]]] = []
    for checkpoint in candidates:
        if not checkpoint.is_dir() or CHECKPOINT_RE.fullmatch(checkpoint.name) is None:
            continue
        step, errors = _inspect_checkpoint(checkpoint, world_size, expected_merge_gate_mode)
        if errors:
            invalid.append((step, checkpoint, errors))
        else:
            valid.append((step, _version(checkpoint), checkpoint))

    if not valid:
        print(f'[resume-check] no complete checkpoint under {output_dir}', file=sys.stderr)
        for _, checkpoint, errors in sorted(invalid, reverse=True)[:3]:
            print(f"[resume-check] rejected {checkpoint}: {'; '.join(errors)}", file=sys.stderr)
        raise SystemExit(2)

    selected_step, _, selected = max(valid)
    newer_world_mismatches = [(checkpoint, errors) for step, checkpoint, errors in invalid
                              if step >= selected_step and any('world size mismatch' in error for error in errors)]
    if newer_world_mismatches:
        checkpoint, errors = newer_world_mismatches[0]
        print(f"[resume-check] refusing mixed-topology resume from {checkpoint}: {'; '.join(errors)}", file=sys.stderr)
        raise SystemExit(2)

    skipped_newer = sum(step > selected_step for step, _, _ in invalid)
    print(
        f'[resume-check] selected step={selected_step} world_size={world_size}; '
        f'skipped_newer_incomplete={skipped_newer}',
        file=sys.stderr,
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--expected-merge-gate-mode', choices=('shared', ))
    args = parser.parse_args()
    if args.world_size < 1:
        parser.error('--world-size must be positive')
    checkpoint = select_checkpoint(args.output_dir, args.world_size, args.expected_merge_gate_mode)
    if checkpoint is not None:
        print(checkpoint.resolve())


if __name__ == '__main__':
    main()
