#!/usr/bin/env python3
"""
Prepare a single mixed training dataset that combines language, code, and CoT data
with configurable upsampling to balance the mix.

Solves catastrophic forgetting by training on everything simultaneously.

Usage:
    python scripts/prepare_mixed_data.py
    python scripts/prepare_mixed_data.py --code-upsample 8 --cot-upsample 3
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create mixed training dataset")
    parser.add_argument("--code-upsample", type=int, default=8,
                        help="How many times to repeat code data (default: 8)")
    parser.add_argument("--cot-upsample", type=int, default=3,
                        help="How many times to repeat CoT data (default: 3)")
    parser.add_argument("--language-downsample", type=float, default=0.5,
                        help="Fraction of language data to keep (default: 0.5)")
    parser.add_argument("--split-ratio", type=float, default=0.95)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    processed = base_dir / "data" / "processed"

    # Load all datasets
    datasets = {}
    for stage in ["language", "code", "cot"]:
        train_path = processed / f"{stage}_train.npy"
        val_path = processed / f"{stage}_val.npy"
        if train_path.exists():
            datasets[stage] = {
                "train": np.load(train_path),
                "val": np.load(val_path),
            }
            print(f"Loaded {stage}: {datasets[stage]['train'].shape[0]:,} train, "
                  f"{datasets[stage]['val'].shape[0]:,} val")

    if not datasets:
        print("ERROR: No datasets found in data/processed/")
        return

    # Apply sampling
    all_train = []
    all_val = []

    if "language" in datasets:
        lang = datasets["language"]["train"]
        n_keep = int(len(lang) * args.language_downsample)
        idx = np.random.RandomState(42).choice(len(lang), n_keep, replace=False)
        lang_sampled = lang[idx]
        all_train.append(lang_sampled)
        all_val.append(datasets["language"]["val"])
        print(f"Language: {len(lang):,} -> {len(lang_sampled):,} (downsample {args.language_downsample}x)")

    if "code" in datasets:
        code = datasets["code"]["train"]
        code_up = np.tile(code, (args.code_upsample, 1))
        all_train.append(code_up)
        all_val.append(datasets["code"]["val"])
        print(f"Code: {len(code):,} -> {len(code_up):,} (upsample {args.code_upsample}x)")

    if "cot" in datasets:
        cot = datasets["cot"]["train"]
        cot_up = np.tile(cot, (args.cot_upsample, 1))
        all_train.append(cot_up)
        all_val.append(datasets["cot"]["val"])
        print(f"CoT: {len(cot):,} -> {len(cot_up):,} (upsample {args.cot_upsample}x)")

    # Merge and shuffle
    train_merged = np.concatenate(all_train, axis=0)
    val_merged = np.concatenate(all_val, axis=0)

    np.random.RandomState(42).shuffle(train_merged)
    np.random.RandomState(42).shuffle(val_merged)

    print(f"\nMixed dataset:")
    print(f"  Train: {len(train_merged):,} sequences ({len(train_merged) * train_merged.shape[1] / 1e6:.0f}M tokens)")
    print(f"  Val: {len(val_merged):,} sequences")

    # Show balance
    total = len(train_merged)
    if "language" in datasets:
        lang_n = int(len(datasets["language"]["train"]) * args.language_downsample)
        print(f"  Language: {lang_n/total*100:.0f}%")
    if "code" in datasets:
        code_n = len(datasets["code"]["train"]) * args.code_upsample
        print(f"  Code: {code_n/total*100:.0f}%")
    if "cot" in datasets:
        cot_n = len(datasets["cot"]["train"]) * args.cot_upsample
        print(f"  CoT: {cot_n/total*100:.0f}%")

    # Save
    np.save(processed / "mixed_train.npy", train_merged)
    np.save(processed / "mixed_val.npy", val_merged)
    print(f"\nSaved to data/processed/mixed_{{train,val}}.npy")
    print(f"\nTrain with:")
    print(f"  python scripts/train.py --stage code --model-size 250m --device rocm \\")
    print(f"    --data-file mixed_train.npy --val-file mixed_val.npy \\")
    print(f"    --num-epochs 3 --batch-size 4 --learning-rate 3e-4")


if __name__ == "__main__":
    main()
