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

## Training from the CLI (same code the notebook calls)

```bash
uv run --no-sync python scripts/pirate_lora.py --rank 8 --before-only  # baseline samples -> data/before.json
uv run --no-sync python scripts/pirate_lora.py --rank 8                # 1 epoch -> adapters/pirate-r8/
uv run --no-sync python scripts/pirate_lora.py --rank 32               # 1 epoch -> adapters/pirate-r32/
uv run --no-sync python scripts/pirate_lora.py --rank 8 --epochs 3 --suffix=-e3
```

## What this toy teaches

- LoRA mechanics: frozen 363M base + rank-r detour on `q,k,v,o` (r=8: 1.6M params, 0.45%; r=32: 6.6M, 1.8%), alpha scaling, `merge_and_unload` back to one file.
- Reading curves: train/val loss, cosine LR schedule, adapter grad-norm.
- Two earned findings: **loss is not style** (1 epoch: r=32 fits better yet sounds blander) and **epochs buy style but tax facts** (3 epochs: unmistakable pirate voice, factual drift, no val overfit).

## Map

- `notebooks/01-pirate-lora.ipynb` - the whole story, cell by cell.
- `scripts/pirate_lora.py` - sampling + training (notebook imports it; CLI runs it).
- `scripts/prepare_data.py` - freezes `data/holdouts.json` + `data/split.json`.
- `data/before.json` - frozen base-model baseline for all comparisons.
- `adapters/pirate-r{8,32}[-e3]/` + `train_log_*.json` - all four adapters and their curves.
- Issues #1-#6 on GitHub - the design trail, including the misses.
