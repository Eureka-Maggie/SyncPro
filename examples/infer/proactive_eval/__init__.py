"""Proactive video understanding evaluation harness.

Loads our SFT-LoRA Qwen3.5 ckpts (post swift-merge) and runs them on the
three proactive benchmarks reported by Em-Garde:

  - OVO-Bench FAR subset       (CRR / REC / SSR, F1@2s)
  - StreamingBench PO subset   (Recall@2s / Accuracy@2s)
  - ProactiveVideoQA EGO/WEB/VAD (model_response_list -> external PAUC@0.5)

Our model is a single VLM that emits per-frame <silent> / <inner>...</inner>
/ <speak>...</speak> tokens at 1 fps. Em-Garde's StreamingModel (proposer +
detector + responder) is replaced; only the JSONL schema and metric formulas
are reused.
"""
