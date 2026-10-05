"""Freeze the data artifacts for the pirate-LoRA toy.

Run: uv run python scripts/prepare_data.py

- Loads quill-voice/pirate (797 rows, instruction -> pirate output).
- Removes the 5 holdout prompts (demo-only, in NEITHER train nor val).
- Splits the rest 90/10 with seed 42.
- Writes data/holdouts.json + data/split.json (frozen for issues #3/#4).

Tokenization itself (ChatML, max 512, assistant-only labels) lives in the
notebook where it can be read and tweaked - this script only freezes *which*
rows go where, so every LoRA run compares on identical data.
"""

import json
from pathlib import Path

SEED = 42
DATASET = "quill-voice/pirate"

# Diverse modern topics: the base model answers these plainly, so the
# before/after style contrast is unmistakable. Matched verbatim to dataset rows.
HOLDOUT_INSTRUCTIONS = [
    "Implement SQL query circuit breaker.",
    "HTML entity for copyright symbol.",
    "What is 220 divided by 22?",
    "What is parasitism?",
    "What is ablation?",
]

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train")
    by_instr = {r["instruction"]: r for r in ds}
    missing = [h for h in HOLDOUT_INSTRUCTIONS if h not in by_instr]
    if missing:
        raise SystemExit(f"holdouts not found verbatim in dataset: {missing}")

    holdout_set = set(HOLDOUT_INSTRUCTIONS)
    rest = [r for r in ds if r["instruction"] not in holdout_set]
    print(f"dataset rows: {len(ds)} | holdouts: {len(holdout_set)} | pool: {len(rest)}")

    from datasets import Dataset

    pool = Dataset.from_list(rest)
    split = pool.train_test_split(test_size=0.1, seed=SEED)
    train, val = split["train"], split["test"]
    print(f"train: {len(train)} | val: {len(val)} | seed: {SEED}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "holdouts.json").write_text(
        json.dumps(HOLDOUT_INSTRUCTIONS, indent=2) + "\n"
    )
    (OUT_DIR / "split.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "seed": SEED,
                "test_size": 0.1,
                "train_n": len(train),
                "val_n": len(val),
                "holdout_n": len(HOLDOUT_INSTRUCTIONS),
                "total_n": len(ds),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT_DIR / 'holdouts.json'} and {OUT_DIR / 'split.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
