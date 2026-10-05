"""Shared LoRA helpers + training CLI for the pirate toy.

Used two ways (same bytes, no drift):
- notebook section 2+: `sys.path.insert(0, '..'); from scripts.pirate_lora import ...`
- overnight/headless: `uv run --no-sync python scripts/pirate_lora.py --rank 8`

Pipeline: load base (bf16 on HIP, fp32 on CPU) -> sample holdouts ->
tokenize with assistant-only labels -> Trainer + peft LoRA ->
adapters/pirate-r{N}/ + adapters/train_log_r{N}.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Single-GPU toy: hide the Raphael iGPU (device 1, tiny shared memory) so
# Trainer never DataParallels onto it - that segfaults (memory fault).
# Must be set before torch is first imported (all torch imports below
# are function-local for exactly this reason).
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATASET = "quill-voice/pirate"
SEED = 42
MAX_LENGTH = 512
LR = 2e-4

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ADAPTERS_DIR = ROOT / "adapters"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def get_device_and_dtype():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return "cpu", torch.float32


def get_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base():
    """Base instruct model in bf16 on HIP (fp32 CPU fallback)."""
    import torch
    from transformers import AutoModelForCausalLM

    device, dtype = get_device_and_dtype()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    if device == "cuda":
        model = model.to("cuda")
    return model


def generate(model, tok, instruction, max_new_tokens=128, temperature=0.7, seed=SEED):
    """Decode one assistant reply; returns text AFTER the prompt (no template tags)."""
    import torch

    if temperature > 0:
        torch.manual_seed(seed)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs.input_ids.shape[1]:])
    return text.split("<|im_end|>")[0].strip()


def sample_holdouts(model, tok, temperature=0.7):
    holdouts = json.loads((DATA_DIR / "holdouts.json").read_text())
    return [
        {"prompt": h, "reply": generate(model, tok, h, temperature=temperature)}
        for h in holdouts
    ]


def tokenize_row(tok, instruction, output, max_length=MAX_LENGTH):
    """ChatML format + tokenize; labels mask the prompt with -100 (see notebook §1)."""
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tok.apply_chat_template(
        [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ],
        tokenize=False,
    )
    input_ids = tok(full, truncation=True, max_length=max_length)["input_ids"]
    pre_len = len(tok(prefix, truncation=True, max_length=max_length)["input_ids"])
    pre_len = min(pre_len, len(input_ids))
    labels = [-100] * pre_len + input_ids[pre_len:]
    return {"input_ids": input_ids, "labels": labels}


def build_tokenized(tok):
    """Same 90/10 seed-42 split as notebook §1 (asserts frozen counts).

    Returns plain lists of dicts - deliberately NOT datasets.map: map's
    Arrow schema inference once mangled input_ids into strings, and a
    list of {input_ids, attention_mask, labels} is exactly what Trainer
    indexes. Transparent beats clever.
    """
    from datasets import load_dataset

    split_info = json.loads((DATA_DIR / "split.json").read_text())
    holdouts = json.loads((DATA_DIR / "holdouts.json").read_text())
    ds = load_dataset(split_info["dataset"], split="train")
    HOLD = set(holdouts)
    # Identical op to notebook §1 (same lib, same seed) -> identical members.
    pool = ds.filter(lambda r: r["instruction"] not in HOLD)
    parts = pool.train_test_split(test_size=0.1, seed=split_info["seed"])
    assert (len(parts["train"]), len(parts["test"])) == (
        split_info["train_n"],
        split_info["val_n"],
    )

    def rows(rs):
        out = []
        for r in rs:
            t = tokenize_row(tok, r["instruction"], r["output"])
            t["attention_mask"] = [1] * len(t["input_ids"])
            out.append(t)
        return out

    return rows(parts["train"]), rows(parts["test"])


def pad_collator(tok):
    """Pad a batch: input_ids with pad_token, attention 0/1, labels with -100.

    Ten lines on purpose - DataCollatorForLanguageModeling would silently
    overwrite our assistant-only labels with shifted input_ids, un-teaching
    notebook §1. Explicit beats magic here.
    """
    import torch

    pad_id = tok.pad_token_id

    def collate(features):
        longest = max(len(f["input_ids"]) for f in features)
        batch_ids, batch_mask, batch_labels = [], [], []
        for f in features:
            pad = longest - len(f["input_ids"])
            batch_ids.append(f["input_ids"] + [pad_id] * pad)
            batch_mask.append([1] * len(f["input_ids"]) + [0] * pad)
            batch_labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    return collate


def train(rank, alpha=None, epochs=1):
    """Train one LoRA adapter; saves adapters/pirate-r{rank}/ + train_log json."""
    import torch
    from transformers import TrainingArguments, Trainer, set_seed
    from peft import LoraConfig, get_peft_model

    alpha = 2 * rank if alpha is None else alpha
    set_seed(SEED)
    device, dtype = get_device_and_dtype()
    use_bf16 = dtype == torch.bfloat16

    tok = get_tokenizer()
    train_ds, val_ds = build_tokenized(tok)
    model = load_base()
    model.config.use_cache = False  # gradient checkpointing-style compat for training

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA r={rank} alpha={alpha} targets={TARGET_MODULES}")
    print(f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    out = ADAPTERS_DIR / f"runs" / f"r{rank}"
    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=use_bf16,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=pad_collator(tok),
    )
    trainer.train()

    dest = ADAPTERS_DIR / f"pirate-r{rank}"
    model.save_pretrained(dest)
    tok.save_pretrained(dest)

    # Trainer already logs learning_rate + grad_norm alongside loss -
    # persist all three so issue #4 can plot loss/LR/grad-norm comparisons.
    train_losses = [
        {
            "step": e["step"],
            "loss": e["loss"],
            "lr": e.get("learning_rate"),
            "grad_norm": e.get("grad_norm"),
        }
        for e in trainer.state.log_history
        if "loss" in e
    ]
    eval_losses = [
        {"step": e["step"], "eval_loss": e["eval_loss"]}
        for e in trainer.state.log_history
        if "eval_loss" in e
    ]
    log = {
        "rank": rank,
        "alpha": alpha,
        "lr": LR,
        "epochs": epochs,
        "effective_batch": 16,
        "seed": SEED,
        "precision": "bf16" if use_bf16 else "fp32",
        "device": device,
        "trainable": trainable,
        "total": total,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
    }
    ADAPTERS_DIR.mkdir(exist_ok=True)
    (ADAPTERS_DIR / f"train_log_r{rank}.json").write_text(json.dumps(log, indent=2))
    print(f"saved adapter -> {dest}")
    return log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--before-only", action="store_true",
                    help="only decode holdouts on base model into data/before.json")
    ns = ap.parse_args(argv)

    tok = get_tokenizer()
    if ns.before_only:
        model = load_base()
        samples = sample_holdouts(model, tok)
        DATA_DIR.mkdir(exist_ok=True)
        (DATA_DIR / "before.json").write_text(json.dumps(samples, indent=2) + "\n")
        for s in samples:
            print(f"PROMPT: {s['prompt']}\nREPLY: {s['reply']}\n")
        return 0

    train(ns.rank, ns.alpha, ns.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
