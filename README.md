# mini-lora

Toy project: learn LoRA by retraining an open-weight model (SmolLM2-360M-Instruct) to speak pirate. You drive, the notebook explains.

## Setup (RX 7900 XTX / gfx1100, Python 3.14)

PyPI `torch` is CUDA/CPU-only, so torch comes from AMD's wheel index:

```bash
uv sync --python 3.14
./scripts/install-rocm-torch.sh
uv run python scripts/check_env.py   # expect hip_available: True, bf16
```

Open the notebook with the project venv kernel:

```bash
uv run python -m ipykernel install --user --name mini-lora --display-name "mini-lora (.venv)"
uv run --with jupyter jupyter lab notebooks/01-pirate-lora.ipynb
```

CPU-only fallback works (no GPU required) - the env-check cell prints the precision decision (`bf16` on HIP, `fp32` on CPU).
