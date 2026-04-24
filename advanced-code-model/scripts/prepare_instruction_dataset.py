#!/usr/bin/env python3
"""Prepare instruction-tuning dataset by combining CoT + CodeAlpaca.

Converts CodeAlpaca JSON to chat format, combines with existing CoT data,
applies dedup + low-entropy filtering, and writes a unified instruction dataset.

Chat format:
  <|im_start|>system
  You are a coding assistant...
  <|im_start|>user
  [instruction or question]
  <|im_start|>assistant
  [response]
  <|im_end|>
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from tokenizers import Tokenizer

# Chat template tokens (must match tokenizer)
SYSTEM_TOKEN = "<|im_start|>system\n"
USER_TOKEN = "<|im_start|>user\n"
ASSISTANT_TOKEN = "<|im_start|>assistant\n"
END_TOKEN = "<|im_end|>\n"

SYSTEM_PROMPTS = {
    "bash": "You are a bash scripting expert. Write clear, efficient shell scripts.",
    "python": "You are a Python programming expert. Write clean, well-documented code.",
    "general": "You are a helpful coding assistant. Write clear, correct code in any language.",
}


def format_chat(instruction: str, response: str, domain: str = "general") -> str:
    """Format instruction/response pair as chat."""
    system = SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["general"])
    return f"{SYSTEM_TOKEN}{system}{USER_TOKEN}{instruction}{ASSISTANT_TOKEN}{response}{END_TOKEN}"


def load_codealpaca(path: Path, tokenizer: Tokenizer, max_samples: int = None) -> np.ndarray:
    """Load and convert CodeAlpaca JSON to tokenized chat format."""
    print(f"Loading CodeAlpaca from {path}...")
    with open(path) as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    sequences = []
    for item in data:
        instruction = item.get("instruction", "")
        response = item.get("response", "")

        # Detect domain from content (simple heuristic)
        instr_lower = instruction.lower()
        if "bash" in instr_lower or "shell" in instr_lower or "script" in instr_lower:
            domain = "bash"
        elif "python" in instr_lower:
            domain = "python"
        else:
            domain = "general"

        chat_text = format_chat(instruction, response, domain)
        tokens = tokenizer.encode(chat_text).ids

        # Truncate or pad to seq_len
        seq_len = 1024
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens = tokens + [0] * (seq_len - len(tokens))

        sequences.append(tokens)

    return np.array(sequences, dtype=np.int32)


def load_cot(path: Path) -> np.ndarray:
    """Load existing CoT data (already in chat format)."""
    print(f"Loading CoT from {path}...")
    return np.load(path)


def hash_rows(arr: np.ndarray) -> np.ndarray:
    """Hash each row for deduplication."""
    import hashlib
    out = np.empty(arr.shape[0], dtype=np.uint64)
    contig = np.ascontiguousarray(arr)
    view = contig.view(np.uint8).reshape(arr.shape[0], -1)
    for i in range(arr.shape[0]):
        out[i] = int.from_bytes(hashlib.blake2b(view[i].tobytes(), digest_size=8).digest(), "little")
    return out


def dedup_rows(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop exact duplicate rows."""
    hashes = hash_rows(arr)
    _, first_idx = np.unique(hashes, return_index=True)
    first_idx.sort()
    n_dropped = arr.shape[0] - first_idx.shape[0]
    return arr[first_idx], n_dropped


def low_entropy_mask(arr: np.ndarray, min_uniq_ratio: float, pad_id: int,
                     min_content_tokens: int = 64) -> np.ndarray:
    """Filter low-entropy rows (same logic as clean rebuild)."""
    n_rows = arr.shape[0]
    keep = np.empty(n_rows, dtype=bool)
    for i in range(n_rows):
        row = arr[i]
        content = row[row != pad_id]
        if content.size < min_content_tokens:
            keep[i] = False
        else:
            keep[i] = len(np.unique(content)) >= min_uniq_ratio * content.size
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codealpaca", default="data/cot/opensource_bash_instructions.json")
    ap.add_argument("--cot-train", default="data/processed/cot_train.npy")
    ap.add_argument("--cot-val", default="data/processed/cot_val.npy")
    ap.add_argument("--gsm8k", default="data/processed/gsm8k_cot_train.npy")
    ap.add_argument("--min-uniq-ratio", type=float, default=0.10)
    ap.add_argument("--max-codealpaca", type=int, default=None)
    ap.add_argument("--out-prefix", default="mixed_instruction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_dir = Path(__file__).parent.parent
    processed = base_dir / "data" / "processed"

    tokenizer = Tokenizer.from_file(str(base_dir / "data" / "tokenizer" / "tokenizer.json"))

    print("=" * 70)
    print("INSTRUCTION DATASET PREP")
    print("=" * 70)

    # Load sources
    all_train, all_val = [], []

    # CodeAlpaca (convert JSON → chat format)
    ca_path = base_dir / args.codealpaca
    if ca_path.exists():
        ca_train = load_codealpaca(ca_path, tokenizer, args.max_codealpaca)
        ca_val = ca_train[:min(1000, len(ca_train)//10)]
        ca_train = ca_train[1000:] if len(ca_train) > 1000 else ca_train[len(ca_val):]
        print(f"  CodeAlpaca: train {len(ca_train):,}, val {len(ca_val):,}")
        all_train.append(ca_train)
        all_val.append(ca_val)
    else:
        print(f"  CodeAlpaca: not found at {ca_path}")

    # CoT (already in chat format)
    cot_train_path = base_dir / args.cot_train
    if cot_train_path.exists():
        cot_train = load_cot(cot_train_path)
        cot_val_path = base_dir / args.cot_val
        cot_val = load_cot(cot_val_path) if cot_val_path.exists() else cot_train[:1000]
        print(f"  CoT: train {len(cot_train):,}, val {len(cot_val):,}")
        all_train.append(cot_train)
        all_val.append(cot_val)
    else:
        print(f"  CoT: not found at {cot_train_path}")

    # GSM8K CoT
    gsm_path = base_dir / args.gsm8k
    if gsm_path.exists():
        gsm = load_cot(gsm_path)
        print(f"  GSM8K: {len(gsm):,}")
        all_train.append(gsm)

    if not all_train:
        raise SystemExit("no instruction data found")

    # Combine
    combined_train = np.concatenate(all_train, axis=0)
    combined_val = np.concatenate(all_val, axis=0) if len(all_val) > 1 else all_val[0][:1000]

    print(f"\nBefore filtering: train {combined_train.shape[0]:,}, val {combined_val.shape[0]:,}")

    # Filter low entropy
    keep_train = low_entropy_mask(combined_train, args.min_uniq_ratio, 0)
    combined_train = combined_train[keep_train]
    keep_val = low_entropy_mask(combined_val, args.min_uniq_ratio, 0)
    combined_val = combined_val[keep_val]

    print(f"After filter: train {combined_train.shape[0]:,}, val {combined_val.shape[0]:,}")

    # Dedup
    combined_train, n_dup_train = dedup_rows(combined_train)
    combined_val, n_dup_val = dedup_rows(combined_val)

    print(f"After dedup: train {combined_train.shape[0]:,} (-{n_dup_train:,}), val {combined_val.shape[0]:,} (-{n_dup_val:,})")

    # Shuffle
    rng = np.random.RandomState(args.seed)
    rng.shuffle(combined_train)
    rng.shuffle(combined_val)

    # Save
    out_train = processed / f"{args.out_prefix}_train.npy"
    out_val = processed / f"{args.out_prefix}_val.npy"
    np.save(out_train, combined_train)
    np.save(out_val, combined_val)

    tokens = combined_train.shape[0] * combined_train.shape[1]
    print(f"\n{'=' * 70}")
    print(f"WROTE {out_train}")
    print(f"  Train: {combined_train.shape[0]:,} × {combined_train.shape[1]} = {tokens/1e9:.2f}B tokens")
    print(f"  Val:   {combined_val.shape[0]:,}")
    print(f"  File size: {combined_train.nbytes / 1e9:.1f} GB")
    print(f"\nChat format: <|im_start|>system/user/assistant + <|im_end|>")


if __name__ == "__main__":
    main()
