#!/usr/bin/env bash
set -euo pipefail

# Final SFT recipe: dual split action loss + local A-branch temporal ranking
# (ATR), with the additive merge coefficient fixed at gamma=0.5.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${MODEL_PATH:?set MODEL_PATH to the Qwen3.5-4B model directory}"
: "${SFT_CACHE_DIR:?set SFT_CACHE_DIR to the output of scripts/prepare_sft_cache.sh}"

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/sft_split_atr}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-$ROOT/data/system_prompt_dual.txt}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

for path in "$MODEL_PATH" "$SFT_CACHE_DIR" "$SYSTEM_PROMPT"; do
    test -e "$path" || { echo "missing required input: $path" >&2; exit 2; }
done
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python not found: $PYTHON" >&2; exit 2; }

CACHED_TRAIN_DATA=()
CACHED_VAL_DATA=()
for chunk in "$SFT_CACHE_DIR"/chunk_*; do
    [[ -d "$chunk" ]] || continue
    [[ -f "$chunk/train/dataset_info.json" && -f "$chunk/val/dataset_info.json" ]] || {
        echo "incomplete cached shard: $chunk" >&2
        exit 2
    }
    CACHED_TRAIN_DATA+=("$chunk/train")
    CACHED_VAL_DATA+=("$chunk/val")
done
if (( ${#CACHED_TRAIN_DATA[@]} != 8 )); then
    echo "expected 8 complete cached shards in $SFT_CACHE_DIR, found ${#CACHED_TRAIN_DATA[@]}" >&2
    exit 2
fi

for value in "$TOTAL_BATCH_SIZE" "$PER_DEVICE_BATCH_SIZE" "$NPROC_PER_NODE" "$NNODES"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "batch/topology values must be positive integers" >&2; exit 2; }
done
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
MICRO_BATCH=$((PER_DEVICE_BATCH_SIZE * WORLD_SIZE))
if (( TOTAL_BATCH_SIZE % MICRO_BATCH != 0 )); then
    echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE must be divisible by PER_DEVICE_BATCH_SIZE*WORLD_SIZE=$MICRO_BATCH" >&2
    exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / MICRO_BATCH))

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MSS_DUAL_BRANCH=1
export MSS_MERGE_MODE=additive
export MSS_MERGE_GATE_MODE=shared
export MSS_MERGE_GAMMA=0.5
export MSS_LORA_RANK=64
export MSS_LORA_ALPHA=128
export MSS_LORA_DROPOUT=0.05
export MSS_POSITION_ID_OVERLAP=1
export MSS_DUAL_BRANCH_FORK_SOURCE=1
export MSS_DUAL_ACTION_LOSS_MODE=restricted_binary
export FIRST_SILENT_WEIGHT=1.0
export A_TEMPORAL_RANK_LAMBDA=0.05
export A_TEMPORAL_RANK_MARGIN=0.5
export A_TEMPORAL_RANK_RADIUS=4
export A_TEMPORAL_RANK_HARD_NEGATIVES=1
export REPEAT_SILENT_FOCAL_LAMBDA=0
export STEP_MASK_WINDOW=1
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-128}"
export MSS_FLEX_KERNEL_OPTS="${MSS_FLEX_KERNEL_OPTS:-BLOCK_M=32,BLOCK_N=32,num_stages=1,num_warps=4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SPECIAL_INNER_WEIGHT=2.5
SPECIAL_SILENT_WEIGHT=0.2
SPECIAL_SPEAK_WEIGHT=3.0
# gamma_logit is intentionally absent. Its zero initialization remains frozen,
# so sigmoid(gamma_logit)=0.5 for the entire run.
TRAINABLE_PARAMETERS_REGEX='(lora_A_up|lora_A_down|lora_B_up|lora_B_down|embed_tokens|lm_head)'

mkdir -p "$OUTPUT_DIR"
"$PYTHON" examples/models/qwen3_5/write_dual_infer_contract.py \
    --output-dir "$OUTPUT_DIR" \
    --rank "$MSS_LORA_RANK" \
    --alpha "$MSS_LORA_ALPHA" \
    --dropout "$MSS_LORA_DROPOUT" \
    --merge-mode "$MSS_MERGE_MODE" \
    --merge-gamma "$MSS_MERGE_GAMMA" \
    --merge-gate-mode "$MSS_MERGE_GATE_MODE" \
    --position-overlap "$MSS_POSITION_ID_OVERLAP" \
    --source-fork "$MSS_DUAL_BRANCH_FORK_SOURCE" \
    --step-mask-window "$STEP_MASK_WINDOW" \
    --trainable-regex "$TRAINABLE_PARAMETERS_REGEX" \
    --system-prompt "$SYSTEM_PROMPT" \
    --first-silent-weight "$FIRST_SILENT_WEIGHT" \
    --special-inner-weight "$SPECIAL_INNER_WEIGHT" \
    --special-silent-weight "$SPECIAL_SILENT_WEIGHT" \
    --special-speak-weight "$SPECIAL_SPEAK_WEIGHT" \
    --action-loss-mode restricted_binary \
    --a-temporal-rank-lambda "$A_TEMPORAL_RANK_LAMBDA" \
    --a-temporal-rank-margin "$A_TEMPORAL_RANK_MARGIN" \
    --a-temporal-rank-radius "$A_TEMPORAL_RANK_RADIUS" \
    --a-temporal-rank-hard-negatives "$A_TEMPORAL_RANK_HARD_NEGATIVES"

RESUME_ARGS=()
if [[ "${AUTO_RESUME:-1}" == "1" ]]; then
    RESUME_FROM_CHECKPOINT="$("$PYTHON" examples/models/qwen3_5/select_resume_checkpoint.py \
        --output-dir "$OUTPUT_DIR" \
        --world-size "$WORLD_SIZE" \
        --expected-merge-gate-mode shared)"
    if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
        RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
    fi
fi

echo "[SFT] method=split-loss+ATR merge=fixed-gamma-0.5 world_size=$WORLD_SIZE global_batch=$TOTAL_BATCH_SIZE"
echo "[SFT] model=$MODEL_PATH cache=$SFT_CACHE_DIR output=$OUTPUT_DIR"
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[SFT] preflight passed"
    exit 0
fi

REPORT_TO="${REPORT_TO:-none}"
REPORT_ARGS=(--report_to "$REPORT_TO")
if [[ "$REPORT_TO" == "swanlab" ]]; then
    : "${SWANLAB_API_KEY:?set SWANLAB_API_KEY when REPORT_TO=swanlab}"
    REPORT_ARGS+=(
        --swanlab_mode "${SWANLAB_MODE:-cloud}"
        --swanlab_project "${SWANLAB_PROJECT:-mss-sft}"
        --swanlab_exp_name "${SWANLAB_EXP_NAME:-split-loss-atr}"
    )
fi

exec env \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    NNODES="$NNODES" \
    NODE_RANK="$NODE_RANK" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    "$PYTHON" -m swift.cli.main sft \
    --model "$MODEL_PATH" \
    --attn_impl flash_attention_2 \
    --train_type full \
    --gradient_checkpointing true \
    --freeze_parameters_regex '.*' \
    --trainable_parameters_regex "$TRAINABLE_PARAMETERS_REGEX" \
    --padding_free true \
    --loss_scale ignore_empty_think \
    --add_non_thinking_prefix true \
    --new_special_tokens '<inner>' '</inner>' '<silent>' '<speak>' '</speak>' \
    --freeze_vit true \
    --system "$SYSTEM_PROMPT" \
    --cached_dataset "${CACHED_TRAIN_DATA[@]}" \
    --cached_val_dataset "${CACHED_VAL_DATA[@]}" \
    --use_logits_to_keep true \
    --special_token_names '<inner>' '<silent>' '<speak>' \
    --special_token_weights "$SPECIAL_INNER_WEIGHT" "$SPECIAL_SILENT_WEIGHT" "$SPECIAL_SPEAK_WEIGHT" \
    --torch_dtype bfloat16 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
    --optim "${OPTIM:-adamw_torch_fused}" \
    --weight_decay "${WEIGHT_DECAY:-0.1}" \
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
    --warmup_ratio "${WARMUP_RATIO:-0.05}" \
    --seed "${SEED:-42}" \
    --data_seed "${DATA_SEED:-42}" \
    --output_dir "$OUTPUT_DIR" \
    --add_version false \
    --resume_only_model false \
    --ignore_data_skip false \
    --save_only_model false \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps "${EVAL_STEPS:-50}" \
    --save_steps "${SAVE_STEPS:-10}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-20}" \
    --logging_steps "${LOGGING_STEPS:-5}" \
    --max_length "${MAX_LENGTH:-32768}" \
    --dataset_num_proc "${DATASET_NUM_PROC:-32}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
    --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR:-1}" \
    --deepspeed zero2 \
    "${REPORT_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "$@"
