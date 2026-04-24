#!/usr/bin/env python3
"""
Rebuild mixed_train.npy from all available data sources with minimal upsampling.

With expanded data, we need much less upsampling — real data > repeated data.

Usage:
    python scripts/rebuild_mixed_data.py
    python scripts/rebuild_mixed_data.py --code-upsample 2 --cot-upsample 1
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rebuild expanded mixed dataset")
    parser.add_argument("--code-upsample", type=int, default=2,
                        help="Code data upsample factor (default: 2, was 8)")
    parser.add_argument("--cot-upsample", type=int, default=1,
                        help="CoT data upsample factor (default: 1, was 3)")
    parser.add_argument("--split-ratio", type=float, default=0.95)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    processed = base_dir / "data" / "processed"

    print("=" * 60)
    print("Rebuilding Mixed Dataset (Expanded)")
    print("=" * 60)

    all_train = []
    all_val = []

    # Load all available data sources
    sources = {
        # Original sources
        "language": {"upsample": 1, "type": "language"},
        "code": {"upsample": args.code_upsample, "type": "code"},
        "cot": {"upsample": args.cot_upsample, "type": "code"},
        # Expanded sources (no upsampling — fresh data)
        "tinystories_full": {"upsample": 1, "type": "language"},
        "openwebtext": {"upsample": 1, "type": "language"},
        "starcoderdata": {"upsample": 1, "type": "code"},
    }

    total_train = 0
    total_val = 0

    for name, cfg in sources.items():
        train_path = processed / f"{name}_train.npy"
        val_path = processed / f"{name}_val.npy"

        if not train_path.exists():
            print(f"  {name:25s}  -- not found, skipping")
            continue

        train = np.load(train_path)
        val = np.load(val_path) if val_path.exists() else None

        upsample = cfg["upsample"]
        if upsample > 1:
            train_expanded = np.tile(train, (upsample, 1))
        else:
            train_expanded = train

        all_train.append(train_expanded)
        n_train = train_expanded.shape[0]
        total_train += n_train

        if val is not None:
            all_val.append(val)
            n_val = val.shape[0]
            total_val += n_val
        else:
            n_val = 0

        tokens_b = n_train * train.shape[1] / 1e9
        print(f"  {name:25s}  {train.shape[0]:>10,} seqs × {upsample}x = "
              f"{n_train:>10,} train, {n_val:>8,} val  ({tokens_b:.2f}B tokens)")

    if not all_train:
        print("\nERROR: No data found!")
        return

    # Combine and shuffle
    print(f"\nCombining...")
    combined_train = np.concatenate(all_train, axis=0)
    combined_val = np.concatenate(all_val, axis=0) if all_val else combined_train[:1000]

    print(f"Shuffling {combined_train.shape[0]:,} training sequences...")
    rng = np.random.RandomState(42)
    rng.shuffle(combined_train)

    # Save
    # Backup old mixed data
    old_train = processed / "mixed_train.npy"
    old_val = processed / "mixed_val.npy"
    if old_train.exists():
        backup = processed / "mixed_train_v1.npy"
        if not backup.exists():
            print(f"Backing up old mixed_train.npy -> mixed_train_v1.npy")
            old_train.rename(backup)
            old_val.rename(processed / "mixed_val_v1.npy")

    np.save(processed / "mixed_train.npy", combined_train)
    np.save(processed / "mixed_val.npy", combined_val)

    train_tokens = combined_train.shape[0] * combined_train.shape[1]
    val_tokens = combined_val.shape[0] * combined_val.shape[1]

    print(f"\n{'=' * 60}")
    print(f"Mixed Dataset Rebuilt!")
    print(f"{'=' * 60}")
    print(f"  Train: {combined_train.shape[0]:,} seqs = {train_tokens/1e9:.2f}B tokens")
    print(f"  Val:   {combined_val.shape[0]:,} seqs = {val_tokens/1e9:.2f}B tokens")
    print(f"  Seq len: {combined_train.shape[1]}")

    # Chinchilla analysis
    params = 393_000_000
    chinchilla = params * 20
    print(f"\n  Chinchilla target: {chinchilla/1e9:.2f}B unique tokens")
    print(f"  Current unique:    {train_tokens/1e9:.2f}B tokens")
    ratio = train_tokens / chinchilla
    if ratio >= 0.8:
        print(f"  Status: {ratio:.0%} of Chinchilla optimal -- GOOD")
    else:
        print(f"  Status: {ratio:.0%} of Chinchilla optimal -- need more data or epochs")

    print(f"\n  With 2 epochs: {train_tokens*2/1e9:.2f}B total tokens seen")
    print(f"  With 3 epochs: {train_tokens*3/1e9:.2f}B total tokens seen")


if __name__ == "__main__":
    main()
