#!/usr/bin/env python3
"""
Rebuild mixed dataset with focus on code and CoT data.

Strategy:
  - Upsample code data heavily (primary training target)
  - Include CoT reasoning data
  - Less upsampling of language data (secondary)
  - Target: 5B+ tokens with 70% code/reasoning focus

Usage:
    python scripts/rebuild_code_cot_data.py
    python scripts/rebuild_code_cot_data.py --code-upsample 4 --cot-upsample 2
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rebuild code+CoT focused dataset")
    parser.add_argument("--code-upsample", type=int, default=4,
                        help="Code data upsample factor (default: 4)")
    parser.add_argument("--cot-upsample", type=int, default=2,
                        help="CoT data upsample factor (default: 2)")
    parser.add_argument("--lang-upsample", type=int, default=1,
                        help="Language data upsample factor (default: 1)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    processed = base_dir / "data" / "processed"

    print("=" * 60)
    print("Rebuilding Mixed Dataset (Code + CoT Focus)")
    print("=" * 60)

    all_train = []
    all_val = []

    # Data sources with priorities
    sources = {
        # Code data (primary - heavy upsampling)
        "code": {"upsample": args.code_upsample, "type": "code", "priority": 1},
        "starcoderdata_pybash": {"upsample": args.code_upsample, "type": "code", "priority": 1},
        "starcoderdata": {"upsample": args.code_upsample, "type": "code", "priority": 1},

        # CoT/reasoning data (secondary - moderate upsampling)
        "cot": {"upsample": args.cot_upsample, "type": "cot", "priority": 2},
        "gsm8k_cot": {"upsample": args.cot_upsample, "type": "cot", "priority": 2},
        "math_cot": {"upsample": args.cot_upsample, "type": "cot", "priority": 2},
        "stackexchange": {"upsample": args.cot_upsample, "type": "cot", "priority": 2},

        # Language data (tertiary - minimal upsampling)
        "language": {"upsample": args.lang_upsample, "type": "language", "priority": 3},
        "tinystories_full": {"upsample": args.lang_upsample, "type": "language", "priority": 3},
        "openwebtext": {"upsample": args.lang_upsample, "type": "language", "priority": 3},
    }

    total_train = 0
    total_val = 0
    code_tokens = 0
    cot_tokens = 0
    lang_tokens = 0

    for priority in [1, 2, 3]:
        print(f"\n{'=' * 60}")
        print(f"Priority {priority}:")
        print(f"{'=' * 60}")

        for name, cfg in sources.items():
            if cfg["priority"] != priority:
                continue

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

            # Track token counts by type
            tokens = n_train * train.shape[1]
            if cfg["type"] == "code":
                code_tokens += tokens
            elif cfg["type"] == "cot":
                cot_tokens += tokens
            else:
                lang_tokens += tokens

            if val is not None:
                all_val.append(val)
                n_val = val.shape[0]
                total_val += n_val
            else:
                n_val = 0

            tokens_b = n_train * train.shape[1] / 1e9
            type_str = f"[{cfg['type'].upper()}]"
            print(f"  {name:25s} {train.shape[0]:>10,} seqs × {upsample}x = "
                  f"{n_train:>10,} train, {n_val:>8,} val  {type_str}  ({tokens_b:.2f}B tokens)")

    if not all_train:
        print("\nERROR: No data found!")
        return

    # Combine and shuffle
    print(f"\n{'=' * 60}")
    print(f"Combining and shuffling...")
    print(f"{'=' * 60}")

    combined_train = np.concatenate(all_train, axis=0)
    combined_val = np.concatenate(all_val, axis=0) if all_val else combined_train[:1000]

    print(f"Shuffling {combined_train.shape[0]:,} training sequences...")
    rng = np.random.RandomState(42)
    rng.shuffle(combined_train)

    # Save
    old_train = processed / "mixed_train.npy"
    old_val = processed / "mixed_val.npy"
    if old_train.exists():
        backup = processed / "mixed_train_v2_code_cot.npy"
        if not backup.exists():
            print(f"Backing up old mixed_train.npy -> mixed_train_v2_code_cot.npy")
            old_train.rename(backup)
            old_val.rename(processed / "mixed_val_v2_code_cot.npy")

    np.save(processed / "mixed_train.npy", combined_train)
    np.save(processed / "mixed_val.npy", combined_val)

    train_tokens = combined_train.shape[0] * combined_train.shape[1]
    val_tokens = combined_val.shape[0] * combined_val.shape[1]

    print(f"\n{'=' * 60}")
    print(f"Mixed Dataset Rebuilt (Code + CoT Focus)!")
    print(f"{'=' * 60}")
    print(f"  Train: {combined_train.shape[0]:,} seqs = {train_tokens/1e9:.2f}B tokens")
    print(f"  Val:   {combined_val.shape[0]:,} seqs = {val_tokens/1e9:.2f}B tokens")
    print(f"  Seq len: {combined_train.shape[1]}")

    print(f"\nToken Distribution:")
    print(f"  Code:        {code_tokens/1e9:.2f}B tokens ({code_tokens/train_tokens:.0%})")
    print(f"  CoT/Reason:  {cot_tokens/1e9:.2f}B tokens ({cot_tokens/train_tokens:.0%})")
    print(f"  Language:    {lang_tokens/1e9:.2f}B tokens ({lang_tokens/train_tokens:.0%})")

    # Chinchilla analysis
    params = 393_000_000
    chinchilla = params * 20
    print(f"\nChinchilla Analysis:")
    print(f"  Target: {chinchilla/1e9:.2f}B unique tokens")
    print(f"  Current: {train_tokens/1e9:.2f}B unique tokens ({train_tokens/chinchilla:.0%})")
    print(f"  With 5 epochs: {train_tokens*5/1e9:.2f}B total tokens seen")
    print(f"  With 10 epochs: {train_tokens*10/1e9:.2f}B total tokens seen")


if __name__ == "__main__":
    main()
