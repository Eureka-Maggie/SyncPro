#!/usr/bin/env bash
set -euo pipefail

# Final RL recipe: the fixed 50-step balanced schedule with hard-window F1 reward.
# No content reward or external LLM judge is used.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${SFT_CHECKPOINT:?set SFT_CHECKPOINT to the split-loss+ATR SFT checkpoint}"
: "${VIDEO_ROOT_OVERRIDES:?set VIDEO_ROOT_OVERRIDES to space-separated SOURCE=/path mappings}"
: "${SYNCPRO_DATA_DIR:?set SYNCPRO_DATA_DIR to the downloaded SyncPro-189k dataset directory}"

PYTHON="${PYTHON:-python3}"
DATA_DIR="${RL_DATA_DIR:-$SYNCPRO_DATA_DIR/rl}"
TRAIN_JSONL="${RL_TRAIN_DATA:-$DATA_DIR/train.jsonl}"
VAL_JSONL="${RL_VAL_DATA:-$DATA_DIR/validation.jsonl}"
MANIFEST="${RL_MANIFEST:-$DATA_DIR/metadata/schedule_manifest.json}"
CLIP_CATALOG="${RL_CLIP_CATALOG:-$DATA_DIR/metadata/train_clip_catalog_f60.jsonl}"
CLIP_MANIFEST="${RL_CLIP_MANIFEST:-$DATA_DIR/metadata/train_clip_catalog_f60.manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/rl_balanced_hard_f1}"

for path in "$SFT_CHECKPOINT" "$TRAIN_JSONL" "$VAL_JSONL" "$MANIFEST" "$CLIP_CATALOG" "$CLIP_MANIFEST"; do
    test -e "$path" || { echo "missing required input: $path" >&2; exit 2; }
done
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python not found: $PYTHON" >&2; exit 2; }

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-16}"
GROUP_SIZE="${GROUP_SIZE:-4}"
for value in "$NPROC_PER_NODE" "$NNODES" "$NODE_RANK" "$MASTER_PORT" "$QUERY_BATCH_SIZE" "$GROUP_SIZE"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "topology/batch values must be integers" >&2; exit 2; }
done
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
if (( WORLD_SIZE < 1 || QUERY_BATCH_SIZE < WORLD_SIZE || QUERY_BATCH_SIZE % WORLD_SIZE != 0 )); then
    echo "QUERY_BATCH_SIZE=$QUERY_BATCH_SIZE must be >= and divisible by WORLD_SIZE=$WORLD_SIZE" >&2
    exit 2
fi
LOCAL_TRAJECTORIES=$((QUERY_BATCH_SIZE / WORLD_SIZE * GROUP_SIZE))
ROLLOUT_MICROBATCH="${ROLLOUT_MICROBATCH:-$LOCAL_TRAJECTORIES}"
if (( ROLLOUT_MICROBATCH > 8 )); then
    ROLLOUT_MICROBATCH=8
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export STEP_MASK_WINDOW=1
export STEP_FAST_ENCODE="${STEP_FAST_ENCODE:-1}"
export MSS_DUAL_DECODE_MODE="${MSS_DUAL_DECODE_MODE:-batched}"
export MSS_MERGE_MODE=additive
export MSS_MERGE_GATE_MODE=shared
export MSS_MERGE_GAMMA=0.5
export MSS_FLEX_KERNEL_OPTS="${MSS_FLEX_KERNEL_OPTS:-BLOCK_M=32,BLOCK_N=32,num_stages=1,num_warps=4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

read -r -a ROOT_MAPPINGS <<< "$VIDEO_ROOT_OVERRIDES"
VIDEO_ARGS=()
for mapping in "${ROOT_MAPPINGS[@]}"; do
    [[ "$mapping" == *=* ]] || { echo "invalid VIDEO_ROOT_OVERRIDES item: $mapping" >&2; exit 2; }
    VIDEO_ARGS+=(--video_root "$mapping")
done

mkdir -p "$OUTPUT_DIR"
"$PYTHON" examples/rl/verify_screened_timing_schedule.py \
    --train "$TRAIN_JSONL" \
    --validation "$VAL_JSONL" \
    --manifest "$MANIFEST" \
    --clip_catalog "$CLIP_CATALOG" \
    --clip_catalog_manifest "$CLIP_MANIFEST" \
    --query_batch_size "$QUERY_BATCH_SIZE" \
    --max_steps "${MAX_STEPS:-50}" \
    --max_frames 60 \
    "${VIDEO_ARGS[@]}"

"$PYTHON" examples/rl/prepare_dual_grpo_shard_plan.py \
    --train_jsonl "$TRAIN_JSONL" \
    --manifest "$MANIFEST" \
    --clip_catalog "$CLIP_CATALOG" \
    --clip_catalog_manifest "$CLIP_MANIFEST" \
    --output "$OUTPUT_DIR/shard_plan.json" \
    --max_frames 60 \
    --world_size "$WORLD_SIZE" \
    --per_device_query_batch_size "$((QUERY_BATCH_SIZE / WORLD_SIZE))" \
    --seed "${SEED:-42}" \
    --batch_schedule fixed_epoch_v1 \
    --timing_data_variant full_positive_v1 \
    --timing_epoch_steps 0 \
    --rank_sharding balanced_frames_v1

echo "[RL] method=balanced+hard-window-F1 world_size=$WORLD_SIZE query_batch=$QUERY_BATCH_SIZE G=$GROUP_SIZE"
echo "[RL] checkpoint=$SFT_CHECKPOINT data=$DATA_DIR output=$OUTPUT_DIR"
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[RL] preflight passed"
    exit 0
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

REPORT_ARGS=(--no-report_to_swanlab --swanlab_mode disabled)
if [[ "${REPORT_TO_SWANLAB:-0}" == "1" ]]; then
    : "${SWANLAB_API_KEY:?set SWANLAB_API_KEY when REPORT_TO_SWANLAB=1}"
    REPORT_ARGS=(
        --report_to_swanlab
        --swanlab_mode "${SWANLAB_MODE:-cloud}"
        --swanlab_project "${SWANLAB_PROJECT:-mss-rl}"
        --swanlab_exp_name "${SWANLAB_EXP_NAME:-balanced-hard-f1}"
    )
fi

exec "$PYTHON" -m torch.distributed.run \
    --nproc-per-node "$NPROC_PER_NODE" \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    --module examples.rl.trainer_grpo_dual_stream \
    --model "$SFT_CHECKPOINT" \
    --timing_train_jsonl "$TRAIN_JSONL" \
    --timing_val_jsonl "$VAL_JSONL" \
    --timing_manifest "$MANIFEST" \
    "${VIDEO_ARGS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --query_batch_size "$QUERY_BATCH_SIZE" \
    --group_size "$GROUP_SIZE" \
    --rollout_trajectory_microbatch_size "$ROLLOUT_MICROBATCH" \
    --rollout_query_batch_mode throughput \
    --reuse_frozen_visual_features \
    --defer_actor_lm_head \
    --actor_fused_frozen_linear_backward \
    --actor_fused_trainable_linear_input_backward \
    --actor_segmented_rmsnorm_backward \
    --actor_fused_gated_rmsnorm_backward \
    --actor_fused_lm_head_input_backward \
    --batch_schedule fixed_epoch_v1 \
    --timing_data_variant full_positive_v1 \
    --timing_epoch_steps 0 \
    --rank_sharding balanced_frames_v1 \
    --actor_trajectory_microbatch_size 1 \
    --actor_packed_token_budget 0 \
    --actor_rank_rebalance post_rollout_groups_v1 \
    --actor_rank_rebalance_min_gain 0.05 \
    --actor_linear_mode inference_aligned \
    --gradient_checkpointing_policy linear_only \
    --max_frames 60 \
    --max_tokens 64 \
    --max_kv_len "${MAX_KV_LEN:-32768}" \
    --max_sequence_length "${MAX_SEQUENCE_LENGTH:-32768}" \
    --streaming_cache \
    --window 1 \
    --learning_rate "${LEARNING_RATE:-1e-6}" \
    --kl_coef 0 \
    --temperature "${TEMPERATURE:-1.0}" \
    --top_p "${TOP_P:-1.0}" \
    --seed "${SEED:-42}" \
    --max_steps "${MAX_STEPS:-50}" \
    --reward_format 0 \
    --reward_timing 1 \
    --timing_reward_mode hard_window_f1_v1 \
    --policy_decision_scope all \
    --num_train_epochs 1 \
    --eval_every "${EVAL_EVERY:-10}" \
    --eval_query_batch_size "${EVAL_QUERY_BATCH_SIZE:-1}" \
    --eval_max_tokens "${EVAL_MAX_TOKENS:-256}" \
    --no-eval_at_start \
    --distributed_timeout_seconds "${DISTRIBUTED_TIMEOUT_SECONDS:-7200}" \
    --deterministic_algorithms \
    --save_every "${SAVE_EVERY:-5}" \
    "${REPORT_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "$@"
