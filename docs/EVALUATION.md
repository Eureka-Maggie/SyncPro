# Evaluation guide

SyncPro is evaluated in causal streaming mode: frames arrive chronologically, future frames are unavailable, and the model must decide whether to remain silent or activate a response.

| Benchmark | Task used in the paper | Primary reported metric | Official resource |
|---|---|---|---|
| StreamingBench | Proactive Output (PO) | Acc@2s | [Project page](https://streamingbench.github.io/) |
| OVO-Bench | Fine-grained Active Responding (FAR) | Mean F1 | [Repository](https://github.com/JoeLeelyf/OVO-Bench) |
| ProactiveVideoQA | Video Anomaly Detection (VAD) | PAUC | [Repository](https://github.com/yellow-binary-tree/ProactiveVideoQA) |

The submitted paper reports 41.2 Acc@2s, 34.19 mean F1, and 32.43 PAUC, respectively. Obtain the benchmark media and annotations from their maintainers and follow the official evaluation protocols. This repository does not redistribute benchmark videos.

The portable validation file under `data/rl/balanced/val.jsonl` contains only the temporal/query annotations needed by the training-side evaluator. It is not a replacement for each benchmark's official test set or evaluation package.
