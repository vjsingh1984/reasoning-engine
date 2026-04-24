#!/usr/bin/env python3
"""
Create Chinchilla-optimal dataset for 400M model (~7.86B tokens).

Strategy:
  - Target: 7.86B tokens (20x per parameter)
  - Heavy bash focus: 70% bash/code
  - Minimal upsampling to reduce memory footprint
  - Estimated file size: ~30GB

Usage:
    python scripts/rebuild_chinchilla_dataset.py
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rebuild Chinchilla-optimal dataset")
    parser.add_argument("--bash-upsample", type=float, default=1.5,
                        help="Bash upsample factor (default: 1.5)")
    parser.add_argument("--code-upsample", type=float, default=1.5,
                        help="Code upsample factor (default: 1.5)")
    parser.add_argument("--cot-upsample", type=float, default=1.0,
                        help="CoT upsample factor (default: 1.0)")
    parser.add_argument("--lang-upsample", type=float, default=0.5,
                        help="Language upsample factor (default: 0.5)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    processed = base_dir / "data" / "processed"

    print("=" * 60)
    print("CHINCHILLA-OPTIMAL DATASET (7.86B tokens)")
    print("=" * 60)
    print(f"Bash upsample: {args.bash_upsample}x")
    print(f"Code upsample: {args.code_upsample}x")
    print(f"CoT upsample: {args.cot_upsample}x")
    print(f"Lang upsample: {args.lang_upsample}x")
    print()

    all_train = []
    all_val = []

    # Prioritized sources for Chinchilla optimal
    sources = {
        # Priority 1: Pure bash
        "github_shell": {"upsample": args.bash_upsample, "type": "bash", "priority": 1},

        # Priority 2: Code
        "starcoderdata_pybash": {"upsample": args.code_upsample, "type": "code", "priority": 2},
        "codeparrot": {"upsample": args.code_upsample, "type": "code", "priority": 2},
        "code": {"upsample": args.code_upsample, "type": "code", "priority": 2},
        "starcoderdata": {"upsample": args.code_upsample, "type": "code", "priority": 2},

        # Priority 3: CoT
        "cot": {"upsample": args.cot_upsample, "type": "cot", "priority": 3},
        "gsm8k_cot": {"upsample": args.cot_upsample, "type": "cot", "priority": 3},

        # Priority 4: Language (minimal - we already have too much)
        "tinystories_full": {"upsample": args.lang_upsample, "type": "language", "priority": 4},
        "openwebtext": {"upsample": args.lang_upsample, "type": "language", "priority": 4},
        "language": {"upsample": args.lang_upsample, "type": "language", "priority": 4},
    }

    total_train = 0
    total_val = 0
    bash_tokens = 0
    code_tokens = 0
    cot_tokens = 0
    lang_tokens = 0

    for priority in [1, 2, 3, 4]:
        print(f"{'=' * 60}")
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

            # Handle fractional upsampling
            if upsample < 1:
                # Sample a subset
                sample_size = int(len(train) * upsample)
                indices = np.random.RandomState(42).permutation(len(train))[:sample_size]
                train_expanded = train[indices]
            else:
                # Regular upsampling
                upsample_int = int(upsample)
                if upsample_int > 1:
                    train_expanded = np.tile(train, (upsample_int, 1))
                    # Handle fractional part
                    if upsample > upsample_int:
                        extra_frac = upsample - upsample_int
                        extra_size = int(len(train) * extra_frac)
                        indices = np.random.RandomState(43).permutation(len(train))[:extra_size]
                        train_expanded = np.concatenate([train_expanded, train[indices]])
                else:
                    train_expanded = train

            all_train.append(train_expanded)
            n_train = train_expanded.shape[0]
            total_train += n_train

            # Track token counts by type
            tokens = n_train * train.shape[1]
            data_type = cfg["type"]
            if data_type == "bash":
                bash_tokens += tokens
            elif data_type == "code":
                code_tokens += tokens
            elif data_type == "cot":
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
            type_str = f"[{data_type.upper()}]"
            marker = "🔥" if data_type == "bash" else ""
            print(f"  {marker} {name:25s} {train.shape[0]:>10,} seqs × {upsample:.1f}x = "
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
        backup = processed / "mixed_train_v4_chinchilla.npy"
        if not backup.exists():
            print(f"Backing up old mixed_train.npy -> mixed_train_v4_chinchilla.npy")
            old_train.rename(backup)
            old_val.rename(processed / "mixed_val_v4_chinchilla.npy")

    np.save(processed / "mixed_train.npy", combined_train)
    np.save(processed / "mixed_val.npy", combined_val)

    train_tokens = combined_train.shape[0] * combined_train.shape[1]
    val_tokens = combined_val.shape[0] * combined_val.shape[1]

    print(f"\n{'=' * 60}")
    print(f"CHINCHILLA-OPTIMAL DATASET CREATED!")
    print(f"{'=' * 60}")
    print(f"  Train: {combined_train.shape[0]:,} seqs = {train_tokens/1e9:.2f}B tokens")
    print(f"  Val:   {combined_val.shape[0]:,} seqs = {val_tokens/1e9:.2f}B tokens")
    print(f"  Seq len: {combined_train.shape[1]}")
    print(f"  File size: {combined_train.nbytes / 1e9:.1f}GB")

    print(f"\n🔥 Token Distribution:")
    print(f"  Bash/Shell:  {bash_tokens/1e9:.2f}B tokens ({bash_tokens/train_tokens:.0%})")
    print(f"  Code:        {code_tokens/1e9:.2f}B tokens ({code_tokens/train_tokens:.0%})")
    print(f"  CoT/Reason:  {cot_tokens/1e9:.2f}B tokens ({cot_tokens/train_tokens:.0%})")
    print(f"  Language:    {lang_tokens/1e9:.2f}B tokens ({lang_tokens/train_tokens:.0%})")

    bash_code_total = bash_tokens + code_tokens
    print(f"\n  Bash+Code:   {bash_code_total/1e9:.2f}B tokens ({bash_code_total/train_tokens:.0%}) 🔥🔥")

    # Chinchilla analysis
    params = 393_000_000
    chinchilla = params * 20
    print(f"\nChinchilla Analysis:")
    print(f"  Target: {chinchilla/1e9:.2f}B unique tokens")
    print(f"  Current: {train_tokens/1e9:.2f}B unique tokens ({train_tokens/chinchilla*100:.0f}%)")
    print(f"  Status: {'✅ OPTIMAL' if 0.9 <= train_tokens/chinchilla <= 1.1 else 'Close'}")

    print(f"\nTraining Projections:")
    print(f"  With 5 epochs: {train_tokens*5/1e9:.2f}B total tokens seen")
    print(f"  With 10 epochs: {train_tokens*10/1e9:.2f}B total tokens seen")


if __name__ == "__main__":
    main()
