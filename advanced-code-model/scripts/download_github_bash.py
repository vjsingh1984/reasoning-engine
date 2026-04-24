#!/usr/bin/env python3
"""
Download and prepare comprehensive bash script dataset.

Sources:
  1. GShadow/github-shell (1.38M shell scripts from GitHub)
  2. Crawler for additional bash-heavy repos

Usage:
    python scripts/download_github_bash.py
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


def download_github_shell_dataset(tokenizer, max_seq_len, pad_id, eos_id,
                                   num_samples=None, output_dir=None):
    """Download GShadow/github-shell dataset."""
    print("\n" + "=" * 60)
    print("Downloading GitHub Shell Scripts Dataset")
    print("=" * 60)

    ds = load_dataset("GShadow/github-shell", split="train")

    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    print(f"  Dataset size: {len(ds):,} scripts")

    all_tokens = []
    total_chars = 0
    skipped = 0

    for item in tqdm(ds, desc="Tokenizing shell scripts"):
        code = item.get("code", "")

        # Filter out very short or very long scripts
        if len(code) < 50 or len(code) > 50000:
            skipped += 1
            continue

        tokens = tokenizer.encode(code).ids
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)
        total_chars += len(code)

    num_seqs = len(all_tokens) // max_seq_len
    if num_seqs == 0:
        print("  WARNING: Not enough tokens")
        return None

    sequences = np.array(all_tokens[:num_seqs * max_seq_len], dtype=np.int32).reshape(num_seqs, max_seq_len)

    print(f"\n  Tokens: {len(all_tokens):,} -> {num_seqs:,} sequences")
    print(f"  Total chars: {total_chars/1e6:.0f}M (~{len(all_tokens)/1e6:.0f}M tokens)")
    print(f"  Skipped: {skipped:,} scripts")

    if output_dir:
        split = int(num_seqs * 0.95)
        np.save(output_dir / "github_shell_train.npy", sequences[:split])
        np.save(output_dir / "github_shell_val.npy", sequences[split:])
        print(f"  Saved: {split:,} train, {num_seqs - split:,} val")

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Download GitHub bash dataset")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to download (default: all)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer_path = base_dir / "data" / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found: {tokenizer_path}")
        return
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Loaded tokenizer: vocab={tokenizer.get_vocab_size()}")

    vocab = tokenizer.get_vocab()
    pad_id = vocab.get("<PAD>", 0)
    eos_id = vocab.get("<EOS>", 3)
    max_seq_len = 1024

    total_new_tokens = 0

    # Download GitHub shell scripts
    seqs = download_github_shell_dataset(
        tokenizer, max_seq_len, pad_id, eos_id, args.num_samples, output_dir)
    if seqs is not None:
        total_new_tokens += seqs.shape[0] * max_seq_len

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"Total new tokens: {total_new_tokens/1e9:.2f}B")

    print(f"\nNext steps:")
    print(f"  1. python scripts/rebuild_code_cot_data.py")
    print(f"  2. Start training with bash-enriched data")


if __name__ == "__main__":
    main()
