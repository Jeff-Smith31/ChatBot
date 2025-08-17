#!/usr/bin/env python3
"""
Incremental training script for a simple conversational language model.
- Fetches training text from URLs and/or local text files.
- Fine-tunes a base causal LM (default: gpt2) for a small number of steps/epochs.
- Saves a new model directory each run (timestamped).

Usage examples:
  python train.py --urls https://example.com/page1 https://example.com/page2 --steps 200
  python train.py --files data/my_text.txt data/notes.txt --epochs 1 --batch_size 2
  python train.py --urls https://example.com --files data/a.txt --base_model gpt2-medium

Notes:
- This script aims to be CPU-friendly by default but will utilize GPU if available.
- Internet access is used to fetch training data via the provided URLs when specified.
- Conversations are NOT persisted; only trained model snapshots are saved.
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import time
import math
from pathlib import Path
from datetime import datetime
from typing import List, Iterable

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DEFAULT_BASE_MODEL = os.environ.get("CHATBOT_BASE_MODEL", "gpt2")
DEFAULT_OUTPUT_ROOT = "models"
USER_AGENT = "ChatBotTrainer/1.0 (+https://example.com)"


def fetch_url_text(url: str, timeout: int = 20) -> str:
    """Fetches a URL and returns visible text, lightly cleaned."""
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = re.sub(r"\t+", "\t", text)
        text = text.strip()
        return text
    except Exception as e:
        print(f"[warn] Failed to fetch {url}: {e}")
        return ""


def read_file_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[warn] Failed to read {path}: {e}")
        return ""


def chunk_text(text: str, min_chars: int = 200) -> List[str]:
    """Simple paragraph-ish chunking to reduce extremely long sequences prior to tokenization."""
    parts = re.split(r"\n{2,}", text)
    chunks = []
    buf = []
    total = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < min_chars:
            buf.append(p)
            total += len(p)
            if total >= min_chars:
                chunks.append("\n\n".join(buf))
                buf, total = [], 0
        else:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, total = [], 0
            chunks.append(p)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def build_dataset(texts: Iterable[str], tokenizer, block_size: int = 256) -> Dataset:
    """Tokenize and group texts into fixed-size blocks for causal LM training."""
    cleaned = [t for t in (t.strip() for t in texts) if t]
    if not cleaned:
        raise ValueError("No training text found after cleaning.")

    def tokenize_fn(examples):
        # Tokenize each text separately in the batch to keep list-of-lists shape
        return tokenizer(
            examples["text"],
            return_attention_mask=False,
            add_special_tokens=False,
            truncation=False,
        )

    # Wrap in dataset first
    raw_ds = Dataset.from_dict({"text": cleaned})

    # Tokenize (map can batch)
    tokenized = raw_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Group into blocks
    def group_texts(examples):
        # Concatenate only fields that are lists of lists (e.g., input_ids)
        concatenated = {}
        for k, v in examples.items():
            if len(v) > 0 and isinstance(v[0], list):
                concatenated[k] = sum(v, [])
        total_length = len(concatenated.get("input_ids", []))
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_ds = tokenized.map(group_texts, batched=True, desc="Grouping into blocks")
    return lm_ds


def latest_model_dir(root: str) -> str | None:
    root_path = Path(root)
    if not root_path.exists():
        return None
    candidates = [p for p in root_path.iterdir() if p.is_dir() and p.name.startswith("model_")]
    if not candidates:
        return None
    return str(sorted(candidates)[-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental fine-tuning for a simple chatbot model")
    p.add_argument("--urls", nargs="*", help="One or more URLs to fetch and use as training data")
    p.add_argument("--files", nargs="*", help="One or more local text files to use as training data")
    p.add_argument("--preset", choices=["oasst1", "dolly15k", "dailydialog"], help="Convenience presets for popular datasets")
    p.add_argument("--hf_dataset", help="Hugging Face dataset name (e.g., OpenAssistant/oasst1)")
    p.add_argument("--hf_split", default="train", help="Dataset split to use (default: train)")
    p.add_argument("--hf_text_field", default="text", help="Text field name in the dataset (default: text)")
    p.add_argument("--hf_role_field", default=None, help="Optional role field (e.g., role) to prefix lines with role -> 'User:' / 'Assistant:'")
    p.add_argument("--hf_user_tag", default="user", help="Value in role field indicating user (default: user)")
    p.add_argument("--hf_assistant_tag", default="assistant", help="Value in role field indicating assistant (default: assistant)")
    p.add_argument("--streaming", action="store_true", help="Stream the HF dataset (for very large datasets)")
    p.add_argument("--max_samples", type=int, default=0, help="Optional cap on number of HF samples (0 = unlimited)")
    p.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="Base model name or path (default: gpt2)")
    p.add_argument("--use_latest_as_base", action="store_true", help="If set, use the latest saved model under ./models as the base, if present")
    p.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT, help="Directory to store trained models (default: models)")

    # Training hyperparameters (kept modest by default)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=0, help="If > 0, limit total training steps and override epochs")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--block_size", type=int, default=256, help="Token block size for causal LM")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--eval_ratio", type=float, default=0.05, help="Holdout ratio for simple eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Use fp16 if available")

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    texts: List[str] = []

    # 1) Hugging Face dataset (optional)
    # Preset convenience mapping
    if args.preset and not args.hf_dataset:
        if args.preset == "oasst1":
            args.hf_dataset = "OpenAssistant/oasst1"
            args.hf_split = args.hf_split or "train"
            args.hf_text_field = "text"
            args.hf_role_field = "role"
        elif args.preset == "dolly15k":
            args.hf_dataset = "databricks/databricks-dolly-15k"
            args.hf_split = args.hf_split or "train"
            # custom fields, handled below
        elif args.preset == "dailydialog":
            args.hf_dataset = "daily_dialog"
            args.hf_split = args.hf_split or "train"
            # custom fields, handled below

    def _example_to_text(ex):
        # Custom mappings for presets/datasets
        if args.preset == "dolly15k":
            instr = ex.get("instruction")
            resp = ex.get("response")
            if instr and resp:
                return f"User: {instr}\n\nAssistant: {resp}"
            return None
        if args.preset == "dailydialog" and "dialog" in ex and isinstance(ex["dialog"], (list, tuple)):
            lines = []
            for i, utt in enumerate(ex["dialog"]):
                role = "User" if i % 2 == 0 else "Assistant"
                lines.append(f"{role}: {utt}")
            return "\n".join(lines)
        # Generic mapping: optional role prefix + text
        txt = ex.get(args.hf_text_field, "")
        if not txt:
            return None
        if args.hf_role_field and ex.get(args.hf_role_field) is not None:
            role_raw = str(ex[args.hf_role_field]).lower()
            if role_raw == str(args.hf_user_tag).lower():
                role_name = "User"
            elif role_raw == str(args.hf_assistant_tag).lower():
                role_name = "Assistant"
            else:
                role_name = role_raw.title()
            txt = f"{role_name}: {txt}"
        return txt

    if args.hf_dataset:
        try:
            if args.streaming:
                ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
                count = 0
                for ex in ds:
                    txt = _example_to_text(ex)
                    if not txt:
                        continue
                    texts.extend(chunk_text(txt))
                    count += 1
                    if args.max_samples and count >= args.max_samples:
                        break
            else:
                ds = load_dataset(args.hf_dataset, split=args.hf_split)
                if args.max_samples and len(ds) > args.max_samples:
                    ds = ds.select(range(args.max_samples))
                for ex in ds:
                    txt = _example_to_text(ex)
                    if txt:
                        texts.extend(chunk_text(txt))
        except Exception as e:
            print(f"[warn] Failed to load HF dataset {args.hf_dataset}: {e}")

    # 2) URLs (optional)
    if args.urls:
        for url in tqdm(args.urls, desc="Fetching URLs"):
            t = fetch_url_text(url)
            if t:
                # add chunked to reduce extreme length pieces
                texts.extend(chunk_text(t))

    # 3) Local files (optional)
    if args.files:
        for fp in args.files:
            t = read_file_text(fp)
            if t:
                texts.extend(chunk_text(t))

    if not texts:
        print("[error] No training data found. Use --hf_dataset and/or --urls and/or --files with text content.")
        sys.exit(1)

    # Choose base model
    base_model = args.base_model
    if args.use_latest_as_base:
        lm = latest_model_dir(args.output_root)
        if lm:
            print(f"Using latest saved model as base: {lm}")
            base_model = lm
        else:
            print("[info] No prior saved model found, using provided base model.")

    print(f"Loading tokenizer and model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # Ensure pad token exists for batch training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Build dataset
    ds = build_dataset(texts, tokenizer=tokenizer, block_size=args.block_size)

    # Simple train/eval split
    eval_size = max(1, int(len(ds) * args.eval_ratio)) if len(ds) > 10 else min(10, len(ds) // 5 or 1)
    split_idx = max(1, len(ds) - eval_size)
    train_ds = ds.select(range(0, split_idx))
    eval_ds = ds.select(range(split_idx, len(ds))) if len(ds) > 1 else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"model_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_steps = args.steps if args.steps > 0 else None

    # Build TrainingArguments with compatibility for older Transformers versions
    import inspect as _inspect
    _params = set(_inspect.signature(TrainingArguments.__init__).parameters.keys())

    # Base kwargs common across versions
    ta_kwargs = dict(
        output_dir=str(out_dir / "hf_runs"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        num_train_epochs=1 if total_steps else args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=1,
        fp16=(args.fp16 and torch.cuda.is_available()),
        bf16=False,
        dataloader_num_workers=0,
        save_total_limit=1,
        seed=args.seed,
        logging_steps=10,
        save_steps=(max(50, total_steps // 4) if (total_steps and total_steps >= 200) else 200),
    )

    # Evaluation-related kwargs, only if supported
    if eval_ds is not None:
        if "evaluation_strategy" in _params:
            ta_kwargs["evaluation_strategy"] = "steps"
        if "eval_steps" in _params:
            ta_kwargs["eval_steps"] = (max(10, total_steps // 5) if (total_steps and total_steps >= 20) else 50)
    else:
        if "evaluation_strategy" in _params:
            ta_kwargs["evaluation_strategy"] = "no"
        elif "do_eval" in _params:
            ta_kwargs["do_eval"] = False

    # Optional reporting key
    if "report_to" in _params:
        ta_kwargs["report_to"] = []

    # Support setting a hard max_steps when requested
    if total_steps and "max_steps" in _params:
        ta_kwargs["max_steps"] = int(total_steps)

    # Filter only supported keys for maximum compatibility
    supported_kwargs = {k: v for k, v in ta_kwargs.items() if k in _params}
    training_args = TrainingArguments(**supported_kwargs)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # Start training; for broad compatibility, rely on TrainingArguments for max_steps when supported
    trainer.train()

    # Save the fine-tuned model snapshot
    print(f"Saving model to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Also write a small metadata file
    meta_path = out_dir / "TRAINING_INFO.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(
            f"Base model: {base_model}\n"
            f"Epochs: {args.epochs}\n"
            f"Steps: {args.steps}\n"
            f"Batch size: {args.batch_size}\n"
            f"LR: {args.lr}\n"
            f"Block size: {args.block_size}\n"
            f"Timestamp: {timestamp}\n"
            f"Sources (URLs): {args.urls or []}\n"
            f"Sources (Files): {args.files or []}\n"
        )

    print("Training complete.")
    print(f"New model saved at: {out_dir}")
    print("You can now run: python chat.py --model_path", out_dir)


if __name__ == "__main__":
    main()
