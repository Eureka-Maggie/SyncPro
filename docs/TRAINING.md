# Training guide

The two launchers encode the paper configurations and validate their inputs before allocating training workers.

## SFT: fixed γ, split loss, and ATR

Required inputs:

- Qwen3.5-4B model directory
- SyncPro SFT training JSONL
- user-selected validation JSONL in the same schema
- `data/system_prompt_dual.txt`

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export SFT_TRAIN_DATA=/path/to/train.jsonl
export SFT_VAL_DATA=/path/to/validation.jsonl
export OUTPUT_DIR=$PWD/outputs/sft_split_atr
bash scripts/train_sft.sh
```

Reference configuration:

| Setting | Value |
|---|---:|
| Merge | additive, γ = 0.5 fixed |
| Dual LoRA | rank 64, α 128, dropout 0.05 |
| Global batch size | 128 |
| Special-token weights | `<inner>` 2.5, `<silent>` 0.2, `<speak>` 3.0 |
| Action loss | restricted binary split loss |
| ATR | λ 0.05, margin 0.5, radius 4, one hard negative |

`scripts/train_sft.sh` writes `dual_infer_contract.json` into the output directory. The contract captures the architecture, fixed merge policy, trainable parameter expression, prompt digest, loss settings, and token weights. Resume is allowed only when the existing contract matches.

Run the configuration checks without launching workers:

```bash
PREFLIGHT_ONLY=1 bash scripts/train_sft.sh
```

## RL: balanced chronological hard-window F1

Obtain the referenced media, configure `VIDEO_ROOT_OVERRIDES` through `configs/video_roots.env`, and start from the final SFT checkpoint:

```bash
cp configs/video_roots.example.env configs/video_roots.env
source configs/video_roots.env

export SFT_CHECKPOINT=/path/to/final-sft-checkpoint
export OUTPUT_DIR=$PWD/outputs/rl_balanced_hard_f1
bash scripts/train_rl.sh
```

Reference configuration:

| Setting | Value |
|---|---:|
| Training annotations | 800 rows |
| Optimizer steps | 50 |
| Query batch size | 16 |
| Group size | 4 |
| Sampled frames | 60 |
| Behavior balance | 4 groups × 200 examples |
| Reward | chronological one-to-one hard-window F1 |

At every step, the fixed schedule contributes four queries from each behavior group. A predicted activation can match at most one ground-truth window in chronological order. With `TP` matches, `P` predictions, and `G` ground-truth windows, the reward is `2 × TP / (P + G)`.

RL preflight validates annotations, manifests, video availability, schedule balance, clip catalogs, and distributed sharding. It still needs the referenced videos because media validation is part of the contract:

```bash
PREFLIGHT_ONLY=1 bash scripts/train_rl.sh
```

## Distributed overrides

Both launchers default to eight local processes. Common overrides include:

```bash
export NPROC_PER_NODE=4
export NNODES=1
export MASTER_PORT=29500
```

SFT requires `TOTAL_BATCH_SIZE` to be divisible by `PER_DEVICE_BATCH_SIZE × NPROC_PER_NODE × NNODES`. RL requires `QUERY_BATCH_SIZE` to be divisible by the world size.

Experiment tracking is off by default. Enable it only through environment variables; never place credentials in repository files.
