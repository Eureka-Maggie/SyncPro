#!/usr/bin/env bash
set -euo pipefail

# Rebuild the eight cached train/validation shards used by the reference SFT
# run. Each released shard is split independently with the original seed and
# 1% validation ratio.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODEL_PATH:?set MODEL_PATH to the Qwen3.5-4B model directory}"
: "${SFT_SHARD_DIR:?set SFT_SHARD_DIR to the downloaded SyncPro-189k/sft directory}"
: "${SFT_MEDIA_ROOT:?set SFT_MEDIA_ROOT to the directory containing the source frame trees}"
: "${SFT_CACHE_DIR:?set SFT_CACHE_DIR to the output cache directory}"

PYTHON="${PYTHON:-python3}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-$ROOT/data/system_prompt_dual.txt}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-32}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
DATA_SEED="${DATA_SEED:-42}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0.01}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-128}"

for path in "$MODEL_PATH" "$SFT_SHARD_DIR" "$SFT_MEDIA_ROOT" "$SYSTEM_PROMPT"; do
    test -e "$path" || { echo "missing required input: $path" >&2; exit 2; }
done
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python not found: $PYTHON" >&2; exit 2; }

MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
SFT_SHARD_DIR="$(cd "$SFT_SHARD_DIR" && pwd)"
SFT_MEDIA_ROOT="$(cd "$SFT_MEDIA_ROOT" && pwd)"
SYSTEM_PROMPT="$(cd "$(dirname "$SYSTEM_PROMPT")" && pwd)/$(basename "$SYSTEM_PROMPT")"
mkdir -p "$SFT_CACHE_DIR"
SFT_CACHE_DIR="$(cd "$SFT_CACHE_DIR" && pwd)"

mapfile -t SHARDS < <(find "$SFT_SHARD_DIR" -maxdepth 1 -type f -name 'train-*.jsonl.gz' | sort)
if (( ${#SHARDS[@]} != 8 )); then
    echo "expected 8 SFT shards in $SFT_SHARD_DIR, found ${#SHARDS[@]}" >&2
    exit 2
fi

for shard in "${SHARDS[@]}"; do
    gzip -t "$shard"
done
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    "$PYTHON" -c 'import swift.cli.main'
    echo "[SFT cache] preflight passed for 8 release shards"
    exit 0
fi

mkdir -p "$SFT_CACHE_DIR/raw"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export IMAGE_MAX_TOKEN_NUM
export MSS_DUAL_BRANCH=1
export MSS_MERGE_MODE=additive
export MSS_DUAL_BRANCH_FORK_SOURCE=1
export MSS_POSITION_ID_OVERLAP=1
export STEP_MASK_WINDOW=1

for index in "${!SHARDS[@]}"; do
    shard="${SHARDS[$index]}"
    printf -v chunk_name 'chunk_%02d' "$index"
    cache_path="$SFT_CACHE_DIR/$chunk_name"
    raw_path="$SFT_CACHE_DIR/raw/$chunk_name.jsonl"

    if [[ -f "$cache_path/train/dataset_info.json" && -f "$cache_path/val/dataset_info.json" ]]; then
        echo "[SFT cache] already complete: $cache_path"
        continue
    fi

    gzip -cd "$shard" > "$raw_path"
    echo "[SFT cache] preparing $chunk_name from $(basename "$shard")"
    (
        cd "$SFT_MEDIA_ROOT"
        "$PYTHON" -m swift.cli.main export \
            --to_cached_dataset true \
            --model "$MODEL_PATH" \
            --new_special_tokens '<inner>' '</inner>' '<silent>' '<speak>' '</speak>' \
            --system "$SYSTEM_PROMPT" \
            --dataset "$raw_path" \
            --max_length "$MAX_LENGTH" \
            --truncation_strategy delete \
            --add_non_thinking_prefix true \
            --loss_scale ignore_empty_think \
            --split_dataset_ratio "$SPLIT_DATASET_RATIO" \
            --data_seed "$DATA_SEED" \
            --dataset_shuffle true \
            --dataset_num_proc "$DATASET_NUM_PROC" \
            --load_from_cache_file false \
            --output_dir "$cache_path"
    )
    test -f "$cache_path/train/dataset_info.json"
    test -f "$cache_path/val/dataset_info.json"
done

echo "[SFT cache] prepared 8 deterministic train/validation shards in $SFT_CACHE_DIR"
