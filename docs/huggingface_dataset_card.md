---
license: cc-by-nc-sa-4.0
language:
- en
task_categories:
- visual-question-answering
- video-text-to-text
tags:
- streaming-video
- proactive-video-understanding
- temporal-grounding
- annotations-only
configs:
- config_name: sft
  data_files:
  - split: train
    path: sft/train-*.jsonl.gz
- config_name: rl
  data_files:
  - split: train
    path: rl/train.jsonl
  - split: validation
    path: rl/validation.jsonl
---

# SyncPro-189k

SyncPro-189k is the annotation release for **SyncPro: Synchronous Reasoning and Responding for Proactive Streaming Video Understanding**. It supports timing-aware supervised fine-tuning and balanced reinforcement learning for causal streaming-video models.

> This repository contains annotations only. It does not distribute videos, extracted frames, benchmark media, or model checkpoints.

During double-blind review, this dataset repository is private. Authorized users must authenticate with Hugging Face before using `load_dataset` or `hf download`. The associated GitHub repository intentionally does not mirror the complete annotations.

## Dataset configurations

| Configuration | Split | Rows | Description |
|---|---:|---:|---|
| `sft` | train | 189,403 | Dual-stream streaming conversations with relative frame references |
| `rl` | train | 800 | Fixed-order balanced timing schedule, 50 batches of 16 |
| `rl` | validation | 58 | Portable timing annotations from three evaluation benchmarks |

```python
from datasets import load_dataset

sft = load_dataset("EurekaTian/SyncPro-189k", "sft", split="train")
rl_train = load_dataset("EurekaTian/SyncPro-189k", "rl", split="train")
rl_validation = load_dataset("EurekaTian/SyncPro-189k", "rl", split="validation")
```

## SFT annotations

Each SFT row contains `messages`, `images`, `data_source`, and `video_id`. The 189,403 trajectories reference 13,522,811 chronological frames. Image paths are portable references such as `magqa/frames/<video_id>/frame_000001.jpg`; users must resolve them against media obtained from the original publisher.

Assistant targets contain an internal reasoning branch followed by an external response branch:

```text
<inner>...</inner><|im_end|>
<silent>
```

or an activation such as:

```text
<silent><|im_end|>
<speak>...</speak>
```

## RL annotations

RL rows use the `proactive_timing_v1` schema. They contain a query, relative video reference, streaming interval, and one or more ground-truth activation windows. Training rows additionally retain the behavior group, source provenance, and fixed schedule slot. The training split has 200 examples in each of four behavior groups.

## Data sources

The SFT annotations draw from these research datasets:

| Source | Rows | Access and terms |
|---|---:|---|
| HACS | 59,331 | [Repository](https://github.com/hangzhaomit/HACS-dataset) |
| MAGQA / Shot2Story | 36,826 | [Dataset card](https://huggingface.co/datasets/mhan/shot2story) |
| ActivityNet Captions | 22,799 | [Project page](https://cs.stanford.edu/people/ranjaykrishna/densevid/) |
| COIN | 22,585 | [Annotations](https://github.com/coin-dataset/annotations) |
| QVHighlights | 19,001 | [Repository](https://github.com/jayleicn/moment_detr) |
| DiDeMo | 10,326 | [Repository](https://github.com/LisaAnne/TemporalLanguageRelease) |
| StreamingCoT | 7,142 | [Repository](https://github.com/Fleeting-hyh/StreamingCoT) |
| MovieNet | 3,463 | [Project page](https://movienet.github.io/) |
| Ego4D GoalStep | 3,296 | [Access guide](https://ego4d-data.org/docs/start-here/) |
| Perception Test | 1,929 | [Repository](https://github.com/google-deepmind/perception_test) |
| YouCook2 | 1,903 | [Download page](https://youcook2.eecs.umich.edu/download) |
| UCF-Crime | 802 | [Project page](https://www.crcv.ucf.edu/projects/real-world/) |

The RL validation annotations reference [StreamingBench](https://streamingbench.github.io/), [OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench), and [ProactiveVideoQA](https://github.com/yellow-binary-tree/ProactiveVideoQA).

Source names and identifiers are included for resolution and attribution. Users are responsible for obtaining each underlying dataset, accepting its terms, and complying with its access, citation, and use restrictions.

All released media references are relative. The export pipeline rejects absolute workstation paths and known internal infrastructure markers, and the uploaded files were independently checked for credentials and internal service identifiers.

## License

The SyncPro annotation files are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This license applies only to the annotation release. Underlying media and source annotations remain subject to their original terms; the more restrictive applicable terms continue to govern their use.

## Paper and code

- Code: [Eureka-Maggie/SyncPro](https://github.com/Eureka-Maggie/SyncPro)
- Paper: under double-blind review at ICLR 2027; link and citation will be added when public

## Citation

Citation metadata will be added when the paper is public. Until then, please cite the original source datasets used in your work.
