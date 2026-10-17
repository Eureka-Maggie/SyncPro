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
---

# SyncPro-189k

SyncPro-189k is the supervised-fine-tuning annotation release for **SyncPro: Synchronous Reasoning and Responding for Proactive Streaming Video Understanding**.

> This repository contains annotations only. It does not distribute videos, extracted frames, benchmark media, or model checkpoints.

The associated GitHub repository intentionally does not mirror the complete annotations. The RL annotations used in the paper are not part of this dataset release.

## Dataset configurations

| Configuration | Split | Rows | Description |
|---|---:|---:|---|
| `sft` | train | 189,403 | Dual-stream streaming conversations with relative frame references |

```python
from datasets import load_dataset

sft = load_dataset("EurekaTian/SyncPro-189k", "sft", split="train")
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

Source names and identifiers are included for resolution and attribution. Users are responsible for obtaining each underlying dataset, accepting its terms, and complying with its access, citation, and use restrictions.

All released media references are relative. The export pipeline rejects absolute workstation paths and known internal infrastructure markers, and the uploaded files were independently checked for credentials and internal service identifiers.

## License

The released SyncPro SFT annotation files are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This license applies only to the annotation release. Underlying media and source annotations remain subject to their original terms; the more restrictive applicable terms continue to govern their use.

## Paper and code

- Code: [Eureka-Maggie/SyncPro](https://github.com/Eureka-Maggie/SyncPro)
- Project page: [SyncPro](https://eureka-maggie.github.io/SyncPro/)
- Paper: [SyncPro: Synchronous Reasoning and Responding for Proactive Streaming Video Understanding](https://eureka-maggie.github.io/SyncPro/paper/SyncPro.pdf)
- BibTeX: [citation file](https://eureka-maggie.github.io/SyncPro/paper/SyncPro.bib)

## Citation

Please use the linked paper citation and also cite the original source datasets used in your work.
