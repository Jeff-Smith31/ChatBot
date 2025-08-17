#!/usr/bin/env python3
"""
Console chatbot app that loads a (fine-tuned) causal language model and chats without saving history.

Examples:
  python chat.py                # uses latest saved model under ./models if present, else base gpt2
  python chat.py --model_path models/model_20250101_120000
  python chat.py --base_model gpt2-medium

Type 'exit' or 'quit' to end the chat.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

# Safety/stability settings for macOS and Python multiprocessing/tokenizers
# Set environment variables BEFORE importing torch/transformers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid semaphore leaks/warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")             # reduce thread contention
os.environ.setdefault("MKL_NUM_THREADS", "1")             # if MKL present
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")        # OpenBLAS threads
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")      # Apple Accelerate/vecLib threads
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")         # numexpr threads
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # be lenient on Apple backends

DEFAULT_BASE_MODEL = os.environ.get("CHATBOT_BASE_MODEL", "gpt2")
DEFAULT_OUTPUT_ROOT = "models"


def latest_model_dir(root: str) -> str | None:
    root_path = Path(root)
    if not root_path.exists():
        return None
    candidates = [p for p in root_path.iterdir() if p.is_dir() and p.name.startswith("model_")]
    if not candidates:
        return None
    return str(sorted(candidates)[-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple console chatbot using a local causal LM")
    p.add_argument("--model_path", help="Path to a fine-tuned model directory to load")
    p.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="Base model to use if no model_path provided")
    p.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT, help="Where to look for latest model if none provided")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Force device selection (default: auto)")
    p.add_argument("--sampling", choices=["auto", "true", "false"], default="auto", help="Sampling mode: auto (default), true (enable sampling), false (greedy)")
    return p.parse_args()


def main():
    # Set spawn start method to avoid macOS fork issues; safe no-op if already set
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # Import heavy libraries AFTER env setup
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Further reduce threading within PyTorch for stability
    try:
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
        # On some macOS setups, disabling MKLDNN avoids low-level crashes
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = False
    except Exception:
        pass

    args = parse_args()

    model_path = args.model_path
    if not model_path:
        lm = latest_model_dir(args.output_root)
        if lm:
            print(f"Loading latest saved model: {lm}")
            model_path = lm
        else:
            print("No saved model found. Falling back to base model.")
            model_path = args.base_model

    print(f"Loading model from: {model_path}")
    # Use slow (pure Python) tokenizer to avoid potential macOS SIGBUS from Rust fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Device selection with explicit avoidance of MPS
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda" and hasattr(torch, "cuda") and torch.cuda.is_available():
        device = "cuda"
    else:
        # auto: prefer CUDA if available; otherwise CPU. We avoid MPS explicitly for stability.
        device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"

    if hasattr(torch.backends, "mps") and getattr(torch.backends.mps, "is_available", lambda: False)():
        # We intentionally do not use MPS to avoid macOS crashes; inform user once.
        print("[info] Detected Apple MPS backend; defaulting to CPU for stability. Use --device cuda if you have CUDA.")
        device = "cpu" if device != "cuda" else device

    model.to(device)
    model.eval()

    # Decide sampling mode
    import sys as _sys
    is_darwin = _sys.platform == "darwin"
    if args.sampling == "true":
        do_sample = True
    elif args.sampling == "false":
        do_sample = False
    else:
        # auto: on CUDA enable sampling; on CPU/macOS default to greedy for stability
        do_sample = (device == "cuda") and not is_darwin

    print("Chat started. Type 'exit' or 'quit' to end.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user:
            continue

        # Stateless prompt style: simple instruction with last user message only
        prompt = (
            "You are a helpful, concise assistant.\n\n"
            f"User: {user}\n"
            "Assistant:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Cap max_new_tokens to model_max_length to avoid overflow on small context windows
        context_len = inputs["input_ids"].shape[1]
        max_allowed = max(16, int(getattr(tokenizer, "model_max_length", 1024)) - context_len - 1)
        safe_new_tokens = max(16, min(args.max_new_tokens, max_allowed))
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=safe_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    num_beams=1,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            # Fallback with smaller generation if something goes wrong (e.g., memory)
            try:
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max(32, min(64, safe_new_tokens)),
                        do_sample=do_sample,
                        temperature=min(1.0, max(0.7, args.temperature)) if do_sample else None,
                        top_p=min(0.95, max(0.8, args.top_p)) if do_sample else None,
                        num_beams=1,
                        use_cache=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            except Exception as e2:
                print(f"[error] Generation failed: {e2}")
                continue
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract assistant portion (after last 'Assistant:')
        if "Assistant:" in text:
            reply = text.split("Assistant:")[-1].strip()
        else:
            # Fallback: strip the prompt
            reply = text[len(prompt):].strip()
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
