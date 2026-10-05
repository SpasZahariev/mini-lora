#!/usr/bin/env bash
# Install AMD ROCm torch for gfx110X (RX 7900 XTX) into the project venv.
# PyPI torch wheels are CUDA/CPU-only, so we pull torch/torchvision from
# AMD's wheel index. Everything else still resolves from PyPI.
#
# Usage: ./scripts/install-rocm-torch.sh
#
# IMPORTANT: re-run this script after EVERY `uv sync` / `uv add`.
# `accelerate` pulls PyPI torch (CUDA/CPU-only) into the lock, and plain
# `uv run` re-syncs the venv back to it. So: run everything with
# `uv run --no-sync ...` (or .venv/bin/python directly), which skips sync.
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_VER="2.10.0+rocm7.13.0"
VISION_VER="0.25.0+rocm7.13.0"
AMD_INDEX="https://repo.amd.com/rocm/whl/gfx110X-all/"

uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  "torch==${TORCH_VER}" "torchvision==${VISION_VER}" \
  --index-url "${AMD_INDEX}" \
  --extra-index-url https://pypi.org/simple
