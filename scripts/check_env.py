"""Environment check for the mini-lora toy project.

Run: uv run python scripts/check_env.py

Reports Python/torch versions, HIP (ROCm) availability, device names,
and the precision decision (bf16 vs fp32) used by later training cells.
Exits 0 whether GPU is present or not - CPU-only is a supported fallback,
it just means tiny models and slower runs.
"""

import platform
import sys


def main() -> int:
    import torch

    print(f"python: {platform.python_version()} ({sys.executable})")
    print(f"torch: {torch.__version__}")

    hip = torch.cuda.is_available()
    print(f"hip_available (torch.cuda.is_available): {hip}")
    if hip:
        print(f"device_count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"device[{i}]: {torch.cuda.get_device_name(i)}")
        use_bf16 = torch.cuda.is_bf16_supported()
    else:
        print("device: CPU (fallback - no HIP/ROCm device visible)")
        print("hint: for AMD GPUs install torch from the AMD index:")
        print("  ./scripts/install-rocm-torch.sh")
        use_bf16 = False

    print(f"use_bf16: {use_bf16}")
    print("precision decision:", "bf16" if use_bf16 else "fp32")

    import matplotlib

    print(f"matplotlib: {matplotlib.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
