# SyncPro environment

SyncPro uses the framework snapshot included in this repository. Install this
checkout in editable mode rather than combining the released training code
with a different framework revision.

## Reference stack

The released SFT and RL recipes were validated with the following stack:

| Component | Reference value |
| --- | --- |
| Platform | Linux x86_64 |
| Python | 3.11.14 |
| GPUs | 8 × NVIDIA L20Z 80 GB |
| NVIDIA driver | 580.105.08 |
| PyTorch | 2.10.0 |
| PyTorch CUDA runtime | 12.8 |
| cuDNN | 9.10.0.2 |
| Transformers | 5.3.0 |
| DeepSpeed | 0.17.6 |
| FlashAttention | 2.8.3 |
| Flash Linear Attention | 0.4.1 |
| causal-conv1d | 1.6.1 |

The reference launchers use one eight-GPU node. Lower-memory devices may
require smaller per-device batches, shorter sequence limits, or different
rollout microbatches.

## Recreate the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install setuptools==80.10.1 wheel==0.46.3 \
  packaging==26.1 ninja==1.13.0
python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --no-build-isolation
python -m pip install -e . --no-deps
```

PyTorch must be installed before FlashAttention and the other compiled CUDA
extensions. If the target driver requires a different PyTorch build, keep the
PyTorch, torchvision, FlashAttention, and CUDA extension builds mutually
compatible rather than mixing wheels from different CUDA runtimes.

The root `requirements.txt` is the canonical dependency entry point. It
includes the component files under `requirements/` automatically; users do
not need to choose among them or install them separately.

## External requirements

- Download `Qwen3.5-4B` and set `MODEL_PATH` for SFT.
- Download the SFT-only `EurekaTian/SyncPro-189k` release and map its source
  frames locally as described in `docs/DATA.md` and `docs/TRAINING.md`.
- RL annotations are not included in the public dataset release; the RL
  launcher requires a separately prepared local schedule.
- Experiment tracking is optional and disabled by default. Keep any tracking
  credentials in environment variables rather than config files.

## Verify

```bash
python - <<'PY'
import torch, transformers

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("gpu_count", torch.cuda.device_count())
PY

python -m unittest discover -s tests -p 'test_release_contract.py'
```
