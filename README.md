# mini-lora

Toy project: learn LoRA by retraining an open-weight model (SmolLM2-360M-Instruct) to speak pirate. You drive, the notebook explains.

## Setup (RX 7900 XTX / gfx1100, Python 3.14)

PyPI `torch` is CUDA/CPU-only, so torch comes from AMD's wheel index:

```bash
uv sync --python 3.14
./scripts/install-rocm-torch.sh   # re-run after EVERY uv sync/add
uv run --no-sync python scripts/check_env.py   # expect hip_available: True, bf16
```

Two env rules (AMD gfx1100 quirks, see `scripts/install-rocm-torch.sh`):
- `accelerate` drags PyPI torch (CUDA/CPU-only) into the lock, so the AMD
  build must be re-installed after every `uv sync`/`uv add`.
- Always run with `uv run --no-sync` - plain `uv run` re-syncs PyPI torch
  back over the AMD build and HIP silently disappears.

Open the notebook with the project venv kernel:

```bash
uv run --no-sync python -m ipykernel install --user --name mini-lora --display-name "mini-lora (.venv)"
uv run --no-sync --with jupyter jupyter lab notebooks/01-pirate-lora.ipynb
```

CPU-only fallback works (no GPU required) - the env-check cell prints the precision decision (`bf16` on HIP, `fp32` on CPU).
