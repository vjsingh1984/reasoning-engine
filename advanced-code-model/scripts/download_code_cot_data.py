#!/usr/bin/env python3
"""
Download additional code and CoT data to improve model quality.

Focus areas:
  - More bash/Python code (primary target)
  - Chain-of-thought reasoning
  - Code explanation pairs
  - Problem-solving with reasoning

Target: Add 3-5B more tokens focused on code + reasoning

Usage:
    python scripts/download_code_cot_data.py
    python scripts/download_code_cot_data.py --skip-reasoning
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def download_more_starcoderdata(tokenizer, max_seq_len, pad_id, eos_id,
                                 num_samples=1000000, output_dir=None):
    """Download more samples from StarCoderData (all code languages)."""
    print("\n" + "=" * 60)
    print(f"Downloading MORE StarCoderData ({num_samples:,} samples)")
    print("=" * 60)

    ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)

    all_tokens = []
    total_chars = 0
    lang_counts = {}
    samples_collected = 0
    pbar = tqdm(total=num_samples, desc="StarCoderData (all code)")

    for i, item in enumerate(ds):
        if samples_collected >= num_samples:
            break
        content = item.get("content", "")
        lang = item.get("lang", "unknown")

        # Skip non-code content
        if lang in ["markdown", "tex", "html", "xml"]:
            continue

        if len(content) < 50 or len(content) > 50000:
            continue

        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        tokens = tokenizer.encode(content).ids
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)
        total_chars += len(content)
        samples_collected += 1
        pbar.update(1)

    pbar.close()

    print("\n  Language breakdown:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang:15s} {count:,}")

    num_seqs = len(all_tokens) // max_seq_len
    if num_seqs == 0:
        print("  WARNING: Not enough tokens")
        return None

    sequences = np.array(all_tokens[:num_seqs * max_seq_len], dtype=np.int32).reshape(num_seqs, max_seq_len)

    print(f"\n  Tokens: {len(all_tokens):,} -> {num_seqs:,} sequences")
    print(f"  Total chars: {total_chars/1e6:.0f}M (~{len(all_tokens)/1e6:.0f}M tokens)")

    if output_dir:
        split = int(num_seqs * 0.95)
        np.save(output_dir / "starcoderdata_pybash_train.npy", sequences[:split])
        np.save(output_dir / "starcoderdata_pybash_val.npy", sequences[split:])
        print(f"  Saved: {split:,} train, {num_seqs - split:,} val")

    return sequences


def download_gsm8k_cot(tokenizer, max_seq_len, pad_id, eos_id, output_dir=None):
    """Download GSM8K with chain-of-thought reasoning."""
    print("\n" + "=" * 60)
    print("Downloading GSM8K (Grade School Math with CoT)")
    print("=" * 60)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"  Dataset: {len(ds):,} problems")

    all_tokens = []
    for item in tqdm(ds, desc="GSM8K CoT"):
        question = item.get("question", "")
        answer = item.get("answer", "")

        # Format as Q&A with reasoning
        text = f"Question: {question}\n\nAnswer: {answer}"

        tokens = tokenizer.encode(text).ids
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)

    num_seqs = len(all_tokens) // max_seq_len
    sequences = np.array(all_tokens[:num_seqs * max_seq_len], dtype=np.int32).reshape(num_seqs, max_seq_len)

    print(f"  Tokens: {len(all_tokens):,} -> {num_seqs:,} sequences")

    if output_dir:
        split = int(num_seqs * 0.95)
        np.save(output_dir / "gsm8k_cot_train.npy", sequences[:split])
        np.save(output_dir / "gsm8k_cot_val.npy", sequences[split:])
        print(f"  Saved: {split:,} train, {num_seqs - split:,} val")

    return sequences


def download_codeparrot(tokenizer, max_seq_len, pad_id, eos_id,
                        num_samples=500000, output_dir=None):
    """Download CodeParrot (more Python code)."""
    print("\n" + "=" * 60)
    print(f"Downloading CodeParrot ({num_samples:,} samples)")
    print("=" * 60)

    try:
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Skipping CodeParrot dataset")
        return None

    all_tokens = []
    total_chars = 0
    pbar = tqdm(total=num_samples, desc="CodeParrot")

    for i, item in enumerate(ds):
        if i >= num_samples:
            break

        content = item.get("content", "")
        if len(content) < 50 or len(content) > 50000:
            continue

        tokens = tokenizer.encode(content).ids
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)
        total_chars += len(content)
        pbar.update(1)

    pbar.close()

    num_seqs = len(all_tokens) // max_seq_len
    if num_seqs == 0:
        print("  WARNING: Not enough tokens")
        return None

    sequences = np.array(all_tokens[:num_seqs * max_seq_len], dtype=np.int32).reshape(num_seqs, max_seq_len)

    print(f"  Tokens: {len(all_tokens):,} -> {num_seqs:,} sequences")
    print(f"  Total chars: {total_chars/1e6:.0f}M (~{len(all_tokens)/1e6:.0f}M tokens)")

    if output_dir:
        split = int(num_seqs * 0.95)
        np.save(output_dir / "codeparrot_train.npy", sequences[:split])
        np.save(output_dir / "codeparrot_val.npy", sequences[split:])
        print(f"  Saved: {split:,} train, {num_seqs - split:,} val")

    return sequences


def download_stackexchange(tokenizer, max_seq_len, pad_id, eos_id,
                           num_samples=200000, output_dir=None):
    """Download StackExchange (code Q&A with explanations)."""
    print("\n" + "=" * 60)
    print(f"Downloading StackExchange ({num_samples:,} samples)")
    print("=" * 60)

    try:
        # Try alternative dataset name
        ds = load_dataset("eli5", split="train_eli5", streaming=True)
        dataset_name = "eli5"
    except:
        try:
            ds = load_dataset("flax-community/pubmed-qa", split="train", streaming=True)
            dataset_name = "pubmed-qa"
        except:
            print("  WARNING: No suitable Q&A dataset found, skipping...")
            return None

    all_tokens = []
    pbar = tqdm(total=num_samples, desc=f"{dataset_name}")

    for i, item in enumerate(ds):
        if i >= num_samples:
            break

        # Try different field names
        question = item.get("question", "") or item.get("title", "") or item.get("text", "")
        answer = item.get("answer", "") or item.get("answers", "") or item.get("response", "")

        if len(question) < 20 or len(answer) < 10:
            continue

        text = f"Q: {question}\n\nA: {answer}"

        tokens = tokenizer.encode(text).ids
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)
        pbar.update(1)

    pbar.close()

    num_seqs = len(all_tokens) // max_seq_len
    if num_seqs == 0:
        print("  WARNING: Not enough tokens")
        return None

    sequences = np.array(all_tokens[:num_seqs * max_seq_len], dtype=np.int32).reshape(num_seqs, max_seq_len)

    print(f"  Tokens: {len(all_tokens):,} -> {num_seqs:,} sequences")

    if output_dir:
        split = int(num_seqs * 0.95)
        np.save(output_dir / "stackexchange_train.npy", sequences[:split])
        np.save(output_dir / "stackexchange_val.npy", sequences[split:])
        print(f"  Saved: {split:,} train, {num_seqs - split:,} val")

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Download code + CoT focused data")
    parser.add_argument("--starcoder-samples", type=int, default=1000000,
                        help="StarCoderData Python/Bash samples (default: 1M)")
    parser.add_argument("--math-samples", type=int, default=100000,
                        help="MATH dataset samples (default: 100K)")
    parser.add_argument("--codeparrot-samples", type=int, default=500000,
                        help="CodeParrot samples (default: 500K)")
    parser.add_argument("--stackexchange-samples", type=int, default=200000,
                        help="StackExchange samples (default: 200K)")
    parser.add_argument("--skip-starcoder", action="store_true",
                        help="Skip StarCoderData download")
    parser.add_argument("--skip-gsm8k", action="store_true",
                        help="Skip GSM8K download")
    parser.add_argument("--skip-codeparrot", action="store_true",
                        help="Skip CodeParrot download")
    parser.add_argument("--skip-stackexchange", action="store_true",
                        help="Skip StackExchange download")
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

    # 1. More StarCoderData (Python + Bash focused)
    if not args.skip_starcoder:
        seqs = download_more_starcoderdata(
            tokenizer, max_seq_len, pad_id, eos_id, args.starcoder_samples, output_dir)
        if seqs is not None:
            total_new_tokens += seqs.shape[0] * max_seq_len

    # 2. GSM8K (math with chain-of-thought)
    if not args.skip_gsm8k:
        seqs = download_gsm8k_cot(tokenizer, max_seq_len, pad_id, eos_id, output_dir)
        if seqs is not None:
            total_new_tokens += seqs.shape[0] * max_seq_len

    # 3. CodeParrot (more Python code)
    if not args.skip_codeparrot:
        seqs = download_codeparrot(
            tokenizer, max_seq_len, pad_id, eos_id, args.codeparrot_samples, output_dir)
        if seqs is not None:
            total_new_tokens += seqs.shape[0] * max_seq_len

    # 4. StackExchange (code Q&A)
    if not args.skip_stackexchange:
        seqs = download_stackexchange(
            tokenizer, max_seq_len, pad_id, eos_id, args.stackexchange_samples, output_dir)
        if seqs is not None:
            total_new_tokens += seqs.shape[0] * max_seq_len

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"New tokens: {total_new_tokens/1e9:.2f}B")

    print(f"\nNext steps:")
    print(f"  1. python scripts/rebuild_code_cot_data.py")
    print(f"  2. Review data distribution")
    print(f"  3. Start training with expanded data")


if __name__ == "__main__":
    main()
