# Training guide

The two launchers encode the paper configurations and validate their inputs before allocating training workers.

## SFT: fixed γ, split loss, and ATR

Required inputs:

- Qwen3.5-4B model directory
- eight SyncPro SFT release shards
- locally obtained source-frame trees referenced by those shards
- `data/system_prompt_dual.txt`

Download `EurekaTian/SyncPro-189k` from Hugging Face as described in
[DATA.md](DATA.md). The reference run split and cached each of the eight shards
independently; it did not use an arbitrary validation file. Recreate those
caches before training:

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export SFT_SHARD_DIR=/path/to/SyncPro-189k/sft
export SFT_MEDIA_ROOT=/path/to/source-frame-root
export SFT_CACHE_DIR=$PWD/.cache/syncpro-sft
bash scripts/prepare_sft_cache.sh

export OUTPUT_DIR=$PWD/outputs/sft_split_atr
bash scripts/train_sft.sh
```

Validate the paths, all eight gzip shards, and the framework entry point
without processing frames:

```bash
PREFLIGHT_ONLY=1 bash scripts/prepare_sft_cache.sh
```

Reference configuration:

| Setting | Value |
|---|---:|
| Merge | additive, γ = 0.5 fixed |
| Dual LoRA | rank 64, α 128, dropout 0.05 |
| Global batch size | 128 |
| Per-device batch / gradient accumulation | 4 / 4 on 8 GPUs |
| Epochs / seed / data seed | 1 / 42 / 42 |
| Learning rate / scheduler | 1e-4 / cosine |
| Optimizer / weight decay | AdamW fused / 0.1 |
| Warmup ratio / max gradient norm | 0.05 / 1.0 |
| Maximum sequence length | 32,768 |
| Image token limit | 128 |
| Validation split | 1% per shard, seed 42 |
| Save / evaluate / log interval | 10 / 50 / 5 steps |
| Distributed strategy | 8 GPUs, DeepSpeed ZeRO-2, BF16 |
| Attention / memory | FlashAttention 2, padding-free, gradient checkpointing |
| Trainable parameters | A/B LoRA adapters, token embeddings, and LM head |
| Frozen parameters | vision tower, base weights, and merge coefficient |
| Data workers | preprocessing 32; loader workers 8, prefetch 1 |
| Special-token weights | `<inner>` 2.5, `<silent>` 0.2, `<speak>` 3.0 |
| Action loss | restricted binary split loss |
| ATR | λ 0.05, margin 0.5, radius 4, one hard negative |
| Resume / checkpoint retention | exact data-state resume / latest 20 checkpoints |
| Experiment reporting | disabled by default |

`scripts/train_sft.sh` writes `dual_infer_contract.json` into the output directory. The contract captures the architecture, fixed merge policy, trainable parameter expression, prompt digest, loss settings, and token weights. Resume is allowed only when the existing contract matches.

Run the configuration checks without launching workers:

```bash
PREFLIGHT_ONLY=1 bash scripts/train_sft.sh
```

## RL: balanced chronological hard-window F1

The paper's RL annotations are not part of the public SyncPro-189k release.
With separately prepared annotations in the expected schema, obtain the
referenced media, configure `VIDEO_ROOT_OVERRIDES`, and start from the final
SFT checkpoint:

```bash
cp configs/video_roots.example.env configs/video_roots.env
source configs/video_roots.env

export SFT_CHECKPOINT=/path/to/sft-checkpoint-1448
export RL_DATA_DIR=/path/to/prepared/balanced-rl-schedule
export OUTPUT_DIR=$PWD/outputs/rl_balanced_hard_f1
bash scripts/train_rl.sh
```

Reference configuration:

| Setting | Value |
|---|---:|
| Training annotations | 960 ordered rows |
| Configured optimizer steps | 60 |
| Released checkpoint | step 50 |
| Query batch size | 16 |
| Group size | 4 |
| Sampled frames | 60 |
| Maximum generated tokens | 64 |
| Maximum KV / sequence length | 32,768 / 32,768 |
| Learning rate / seed | 1e-6 / 42 |
| Sampling temperature / top-p | 1.0 / 1.0 |
| KL / content / auxiliary action losses | 0 / 0 / 0 |
| Behavior balance | 4 groups × 240 examples |
| Schedule / rank sharding | `fixed_epoch_v1` / `balanced_frames_v1` |
| Rollout / actor trajectory microbatch | 8 / 1 per rank |
| Streaming cache / attention window | enabled / 1 |
| Save / evaluate interval | 10 / 10 steps |
| Evaluation batch / token limit | 1 query / 256 tokens |
| Evaluation at start | disabled |
| Deterministic algorithms | enabled |
| Distributed timeout | 7,200 seconds |
| Experiment reporting | disabled by default |
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
