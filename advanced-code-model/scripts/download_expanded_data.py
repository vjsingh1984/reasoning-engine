#!/usr/bin/env python3
"""
Download expanded datasets to hit Chinchilla-optimal token count for 393M model.

Target: ~5B additional unique tokens (currently have ~1.2B unique, need ~7.9B)

Sources:
  - OpenWebText: ~8B tokens available (language data)
  - StarCoderData: massive code corpus (multi-language)
  - Full TinyStories: ~0.5B tokens (currently using only 0.11B)

Usage:
    python scripts/download_expanded_data.py
    python scripts/download_expanded_data.py --lang-samples 500000 --code-samples 300000
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def download_and_tokenize_openwebtext(tokenizer, max_seq_len, pad_id, eos_id,
                                       num_samples=500000, output_dir=None):
    """Download OpenWebText and tokenize into sequences."""
    print("\n" + "=" * 60)
    print(f"Downloading OpenWebText ({num_samples:,} samples)")
    print("=" * 60)

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    all_tokens = []
    total_chars = 0
    pbar = tqdm(total=num_samples, desc="OpenWebText")

    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        text = item.get("text", "")
        if len(text) < 100:
            continue
        tokens = tokenizer.encode(text).ids
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)
        total_chars += len(text)
        pbar.update(1)

    pbar.close()

    # Chunk into sequences
    num_seqs = len(all_tokens) // max_seq_len
    if num_seqs == 0:
        print("  WARNING: Not enough tokens for even 1 sequence")
        return None

    sequences = np.array(all_tokens[:num_seqs * max_seq_len], dtype=np.int32).reshape(num_seqs, max_seq_len)

    print(f"  Tokens: {len(all_tokens):,} -> {num_seqs:,} sequences of {max_seq_len}")
    print(f"  Total chars: {total_chars/1e6:.0f}M (~{len(all_tokens)/1e6:.0f}M tokens)")

    if output_dir:
        # Split 95/5
        split = int(num_seqs * 0.95)
        np.save(output_dir / "openwebtext_train.npy", sequences[:split])
        np.save(output_dir / "openwebtext_val.npy", sequences[split:])
        print(f"  Saved: {split:,} train, {num_seqs - split:,} val")

    return sequences


def download_and_tokenize_starcoderdata(tokenizer, max_seq_len, pad_id, eos_id,
                                         num_samples=300000, output_dir=None):
    """Download StarCoderData and tokenize into sequences."""
    print("\n" + "=" * 60)
    print(f"Downloading StarCoderData ({num_samples:,} samples)")
    print("=" * 60)

    ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)

    all_tokens = []
    total_chars = 0
    lang_counts = {}
    pbar = tqdm(total=num_samples, desc="StarCoderData")

    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        content = item.get("content", "")
        if len(content) < 50 or len(content) > 50000:
            continue

        lang = item.get("lang", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        tokens = tokenizer.encode(content).ids
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)
        total_chars += len(content)
        pbar.update(1)

    pbar.close()

    # Print language breakdown
    print("\n  Language breakdown (top 10):")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {lang:15s} {count:,}")

    # Chunk into sequences
    num_seqs = len(all_tokens) // max_seq_len
    if num_seqs == 0:
        print("  WARNING: Not enough tokens for even 1 sequence")
        return None

    sequences = np.array(all_tokens[:num_seqs * max_seq_len], dtype=np.int32).reshape(num_seqs, max_seq_len)

    print(f"\n  Tokens: {len(all_tokens):,} -> {num_seqs:,} sequences of {max_seq_len}")
    print(f"  Total chars: {total_chars/1e6:.0f}M (~{len(all_tokens)/1e6:.0f}M tokens)")

    if output_dir:
        split = int(num_seqs * 0.95)
        np.save(output_dir / "starcoderdata_train.npy", sequences[:split])
        np.save(output_dir / "starcoderdata_val.npy", sequences[split:])
        print(f"  Saved: {split:,} train, {num_seqs - split:,} val")

    return sequences


def download_full_tinystories(tokenizer, max_seq_len, pad_id, eos_id, output_dir=None):
    """Download full TinyStories (currently only using ~5% of it)."""
    print("\n" + "=" * 60)
    print("Downloading Full TinyStories")
    print("=" * 60)

    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"  Full dataset: {len(ds):,} stories")

    all_tokens = []
    for item in tqdm(ds, desc="TinyStories"):
        text = item["text"]
        if len(text) < 20:
            continue
        tokens = tokenizer.encode(text).ids
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)

    num_seqs = len(all_tokens) // max_seq_len
    sequences = np.array(all_tokens[:num_seqs * max_seq_len], dtype=np.int32).reshape(num_seqs, max_seq_len)

    print(f"  Tokens: {len(all_tokens):,} -> {num_seqs:,} sequences of {max_seq_len}")

    if output_dir:
        split = int(num_seqs * 0.95)
        np.save(output_dir / "tinystories_full_train.npy", sequences[:split])
        np.save(output_dir / "tinystories_full_val.npy", sequences[split:])
        print(f"  Saved: {split:,} train, {num_seqs - split:,} val")

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Download expanded training data")
    parser.add_argument("--lang-samples", type=int, default=500000,
                        help="OpenWebText samples (default: 500K, ~1.5B tokens)")
    parser.add_argument("--code-samples", type=int, default=300000,
                        help="StarCoderData samples (default: 300K, ~1B tokens)")
    parser.add_argument("--skip-tinystories", action="store_true",
                        help="Skip full TinyStories download")
    parser.add_argument("--skip-openwebtext", action="store_true",
                        help="Skip OpenWebText download")
    parser.add_argument("--skip-starcoderdata", action="store_true",
                        help="Skip StarCoderData download")
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

    # 1. Full TinyStories
    if not args.skip_tinystories:
        seqs = download_full_tinystories(tokenizer, max_seq_len, pad_id, eos_id, output_dir)
        if seqs is not None:
            total_new_tokens += seqs.shape[0] * max_seq_len

    # 2. OpenWebText
    if not args.skip_openwebtext:
        seqs = download_and_tokenize_openwebtext(
            tokenizer, max_seq_len, pad_id, eos_id, args.lang_samples, output_dir)
        if seqs is not None:
            total_new_tokens += seqs.shape[0] * max_seq_len

    # 3. StarCoderData
    if not args.skip_starcoderdata:
        seqs = download_and_tokenize_starcoderdata(
            tokenizer, max_seq_len, pad_id, eos_id, args.code_samples, output_dir)
        if seqs is not None:
            total_new_tokens += seqs.shape[0] * max_seq_len

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"New tokens: {total_new_tokens/1e9:.2f}B")
    print(f"\nExisting data:")
    for f in sorted(output_dir.glob("*.npy")):
        d = np.load(f)
        print(f"  {f.name:35s} {d.shape[0]:>10,} seqs = {d.shape[0]*d.shape[1]/1e9:.2f}B tokens")

    print(f"\nNext: python scripts/rebuild_mixed_data.py")


if __name__ == "__main__":
    main()
