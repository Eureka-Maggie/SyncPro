# Data guide

SyncPro-189k is an annotation-only release. It does not contain videos, extracted frames, benchmark media, or model-generated copies of third-party assets. Download each source from its publisher, accept any required terms, and resolve the portable paths locally.

The complete annotations are hosted only in the private Hugging Face dataset repository during review; they are not included in the GitHub repository or its training-code checkout:

```bash
hf auth login
hf download EurekaTian/SyncPro-189k \
  --repo-type dataset \
  --local-dir /path/to/SyncPro-189k
```

The download contains eight compressed SFT shards under `sft/`, the RL train and validation splits under `rl/`, and the RL schedule metadata under `rl/metadata/`. The files under `data/examples/` in the GitHub repository are illustrative schema examples only.

## SFT schema

Each line in the `sft` configuration is one streaming trajectory:

```json
{
  "messages": [
    {"role": "user", "content": "Query: ...<image>"},
    {"role": "assistant", "content": "<inner>...</inner><|im_end|>\n<silent>"}
  ],
  "images": ["magqa/frames/video-id/000000.jpg"],
  "data_source": "MAGQA",
  "video_id": "video-id"
}
```

`images` is chronological and has one entry per assistant turn. Paths are relative references in the form `<source>/frames/<video_id>/<frame>.jpg`; they are not download URLs. Assistant targets contain internal branch B followed by external branch A:

- Branch B: `<silent>` or `<inner>...</inner>`
- Branch A: `<silent>` or `<speak>...</speak>`

The MAGQA portion was filtered for repeated positive responses: 6,318 trajectories were changed and 8,928 duplicate `<speak>` turns were converted to silence. The released SFT file contains 189,403 trajectories and 13,522,811 referenced frames.

## SFT sources

| Source | Trajectories | Publisher |
|---|---:|---|
| HACS | 59,331 | [HACS repository](https://github.com/hangzhaomit/HACS-dataset) |
| MAGQA / Shot2Story | 36,826 | [Shot2Story on Hugging Face](https://huggingface.co/datasets/mhan/shot2story) |
| ActivityNet Captions | 22,799 | [Dense-Captioning Events in Videos](https://cs.stanford.edu/people/ranjaykrishna/densevid/) |
| COIN | 22,585 | [COIN annotations](https://github.com/coin-dataset/annotations) |
| QVHighlights | 19,001 | [Moment-DETR / QVHighlights](https://github.com/jayleicn/moment_detr) |
| DiDeMo | 10,326 | [Temporal Language Release](https://github.com/LisaAnne/TemporalLanguageRelease) |
| StreamingCoT | 7,142 | [StreamingCoT repository](https://github.com/Fleeting-hyh/StreamingCoT) |
| MovieNet | 3,463 | [MovieNet](https://movienet.github.io/) |
| Ego4D GoalStep | 3,296 | [Ego4D access guide](https://ego4d-data.org/docs/start-here/) |
| Perception Test | 1,929 | [Perception Test repository](https://github.com/google-deepmind/perception_test) |
| YouCook2 | 1,903 | [YouCook2 downloads](https://youcook2.eecs.umich.edu/download) |
| UCF-Crime | 802 | [UCF-Crime project](https://www.crcv.ucf.edu/projects/real-world/) |
| **Total** | **189,403** | |

Source names and source-provided video IDs are retained so users with authorized copies can resolve the annotations. They do not imply redistribution rights. All released media references are relative; workstation paths, credentials, and internal service identifiers are excluded by the exporter and release audit.

## RL schema

The `rl` configuration uses the `proactive_timing_v1` schema:

```json
{
  "schema_version": "proactive_timing_v1",
  "sample_id": "schedule:balanced:...",
  "source": "screened_ego4d",
  "task": "PO",
  "query": "Query: ...",
  "video": {
    "root_key": "screened_ego4d",
    "path": "videos/ego4d/example.mp4",
    "group_id": "screened_ego4d:example"
  },
  "stream": {"start_s": 0.0, "end_s": 47.0, "fps": 1.0},
  "gt_windows": [{"start_s": 10.0, "end_s": 14.0}]
}
```

Training rows also include the four-way behavior category, source provenance, and fixed schedule slot. The 800-row training split contains 200 examples from each behavior group. The 58-row validation split references [StreamingBench](https://streamingbench.github.io/), [OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench), and [ProactiveVideoQA](https://github.com/yellow-binary-tree/ProactiveVideoQA).

## Resolving media paths

The repository includes `configs/video_roots.example.env`. Copy it to the ignored local file and edit the common dataset root:

```bash
cp configs/video_roots.example.env configs/video_roots.env
source configs/video_roots.env
```

The RL loader combines each `video.root_key` with its relative `video.path`. The SFT loader resolves each relative image path from the working directory or the data root configured by the user. Absolute workstation paths are rejected by the release exporter.

## License

The SyncPro annotations are released under CC BY-NC-SA 4.0 because incorporated annotation sources include non-commercial ShareAlike terms. This license covers the SyncPro annotation files only. Every source dataset keeps its own terms, attribution requirements, and access procedure; users must satisfy both the SyncPro annotation license and the applicable source terms.
