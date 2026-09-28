"""Fail-closed safetensors metadata helpers for proactive inference."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict


def is_dual_weight_key(key: str) -> bool:
    return ('.lora_A_' in key or '.lora_B_' in key or '.dual_merge.' in key
            or re.search(r'\.mlp\.(gate_proj|up_proj|down_proj)\.base\.weight$', key) is not None)


def checkpoint_weight_locations(model_path: str | Path) -> Dict[str, Path]:
    """Return ``weight key -> safetensors file`` without loading tensors.

    A sharded index is authoritative when present.  Invalid metadata raises
    instead of being interpreted as a normal single-branch checkpoint.
    """
    model_dir = Path(model_path)
    index_path = model_dir / 'model.safetensors.index.json'
    if index_path.exists():
        if not index_path.is_file():
            raise RuntimeError(f'safetensors index is not a regular file: {index_path}')
        try:
            index = json.loads(index_path.read_text(encoding='utf-8'))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f'cannot parse safetensors index: {index_path}') from e
        weight_map = index.get('weight_map') if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f'safetensors index has no non-empty weight_map: {index_path}')
        invalid = [(key, filename) for key, filename in weight_map.items()
                   if not isinstance(key, str) or not key or not isinstance(filename, str) or not filename]
        if invalid:
            raise RuntimeError(f'safetensors index contains invalid weight_map entries: '
                               f'{invalid[:3]!r}')
        return {key: model_dir / filename for key, filename in weight_map.items()}

    single_path = model_dir / 'model.safetensors'
    if not single_path.exists():
        return {}
    if not single_path.is_file():
        raise RuntimeError(f'safetensors checkpoint is not a regular file: {single_path}')
    try:
        from safetensors import safe_open

        with safe_open(single_path, framework='pt', device='cpu') as handle:
            keys = list(handle.keys())
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f'cannot read safetensors metadata: {single_path}') from e
    if not keys:
        raise RuntimeError(f'safetensors checkpoint contains no tensors: {single_path}')
    return {key: single_path for key in keys}


def checkpoint_has_dual_weights(model_path: str | Path) -> bool:
    return any(is_dual_weight_key(key) for key in checkpoint_weight_locations(model_path))


def find_checkpoint_weight(
    model_path: str | Path,
    predicate: Callable[[str], bool],
) -> tuple[str, Path] | None:
    for key, path in checkpoint_weight_locations(model_path).items():
        if predicate(key):
            return key, path
    return None
