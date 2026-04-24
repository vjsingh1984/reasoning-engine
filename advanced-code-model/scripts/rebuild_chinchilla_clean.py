#!/usr/bin/env python3
"""Rebuild Chinchilla dataset with dedup + low-entropy filtering.

Addresses gaps found in mixed_train.npy audit:
  1. Exact-row duplicates from np.tile upsampling -> use shuffled resampling.
  2. Hex-dump / byte-run / CSV-number contamination -> drop rows whose
     unique-token ratio across the whole 1024-row is below --min-uniq-ratio.
     (Calibrated on codeparrot: threshold 0.10 drops 1.8% junk, keeps normal code.)
  3. No dedup between sources -> hash each row, drop repeats.

Writes a new `mixed_train_clean.npy` + `mixed_val_clean.npy` alongside the
existing files (does not overwrite).
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np


SOURCES = {
    "github_shell":         {"upsample": 1.5, "type": "bash"},
    "starcoderdata_pybash": {"upsample": 1.5, "type": "code"},
    "codeparrot":           {"upsample": 1.5, "type": "code"},
    "code":                 {"upsample": 1.5, "type": "code"},
    "starcoderdata":        {"upsample": 1.5, "type": "code"},
    "cot":                  {"upsample": 1.0, "type": "cot"},
    "gsm8k_cot":            {"upsample": 1.0, "type": "cot"},
    "tinystories_full":     {"upsample": 0.5, "type": "language"},
    "openwebtext":          {"upsample": 0.5, "type": "language"},
    "language":             {"upsample": 0.5, "type": "language"},
}


def low_entropy_mask(arr: np.ndarray, min_uniq_ratio: float, pad_id: int,
                     min_content_tokens: int = 32) -> np.ndarray:
    """Return boolean mask of rows whose non-PAD content is sufficiently diverse.

    True = keep row. For each row:
      - Strip PAD tokens; if fewer than min_content_tokens remain, drop.
      - Otherwise keep iff unique_tokens(content) / len(content) >= min_uniq_ratio.

    This correctly drops CSV/hex-dump junk while preserving short but legit
    code that just happens to be heavily padded.
    """
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


def hash_rows(arr: np.ndarray) -> np.ndarray:
    """Return a uint64 hash per row (fast, collision-resistant enough for dedup)."""
    out = np.empty(arr.shape[0], dtype=np.uint64)
    contig = np.ascontiguousarray(arr)
    view = contig.view(np.uint8).reshape(arr.shape[0], -1)
    for i in range(arr.shape[0]):
        out[i] = int.from_bytes(hashlib.blake2b(view[i].tobytes(), digest_size=8).digest(), "little")
    return out


def dedup_rows(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop exact-duplicate rows. Returns (deduped_array, n_dropped)."""
    hashes = hash_rows(arr)
    _, first_idx = np.unique(hashes, return_index=True)
    first_idx.sort()
    n_dropped = arr.shape[0] - first_idx.shape[0]
    return arr[first_idx], n_dropped


def resample(arr: np.ndarray, factor: float, seed: int) -> np.ndarray:
    """Upsample by factor via shuffled permutations — no tile duplicates.

    For factor=1.5, emits one full shuffled copy + a random half-subset.
    For factor<1, returns a subset.
    """
    rng = np.random.RandomState(seed)
    n = arr.shape[0]
    if factor <= 0:
        return arr[:0]
    if factor < 1:
        idx = rng.permutation(n)[: int(n * factor)]
        return arr[idx]

    whole = int(factor)
    frac = factor - whole
    parts = []
    for i in range(whole):
        # Each whole copy gets its own shuffled permutation.
        perm = rng.permutation(n)
        parts.append(arr[perm])
    if frac > 0:
        perm = rng.permutation(n)[: int(n * frac)]
        parts.append(arr[perm])
    return np.concatenate(parts, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-uniq-ratio", type=float, default=0.10,
                    help="Drop rows with unique-token ratio below this "
                         "(0.10 drops ~1.8%% of codeparrot — the CSV/hex-dump junk)")
    ap.add_argument("--pad-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="mixed_clean")
    args = ap.parse_args()

    processed = Path(__file__).parent.parent / "data" / "processed"

    all_train, all_val = [], []
    type_tokens = {"bash": 0, "code": 0, "cot": 0, "language": 0}
    stats = []

    print("=" * 70)
    print("CHINCHILLA CLEAN REBUILD")
    print(f"  min uniq-token ratio: {args.min_uniq_ratio}")
    print(f"  PAD id: {args.pad_id}")
    print("=" * 70)

    for name, cfg in SOURCES.items():
        train_path = processed / f"{name}_train.npy"
        if not train_path.exists():
            print(f"  {name:26s}  -- not found, skipping")
            continue

        train = np.load(train_path)
        val_path = processed / f"{name}_val.npy"
        val = np.load(val_path) if val_path.exists() else None
        n0 = train.shape[0]

        # 1. filter low-entropy rows (hex dumps, byte runs, tiny fragments)
        keep = low_entropy_mask(train, args.min_uniq_ratio, args.pad_id)
        train = train[keep]
        n_after_filter = train.shape[0]

        # 2. dedup exact rows
        train, n_dup = dedup_rows(train)
        n_after_dedup = train.shape[0]

        # 3. upsample via shuffled resampling (no tile)
        train = resample(train, cfg["upsample"], seed=hash(name) & 0xFFFFFFFF)
        n_final = train.shape[0]

        all_train.append(train)
        if val is not None:
            all_val.append(val)

        tokens = n_final * train.shape[1]
        type_tokens[cfg["type"]] += tokens
        stats.append((name, cfg["type"], n0, n_after_filter, n_after_dedup, n_final, tokens))

        print(f"  {name:26s} {n0:>10,} -> filter {n_after_filter:>10,} "
              f"-> dedup {n_after_dedup:>10,} -> sample {n_final:>10,}  "
              f"[{cfg['type']}]")

    if not all_train:
        raise SystemExit("no data loaded")

    print(f"\nConcatenating + global shuffle...")
    combined = np.concatenate(all_train, axis=0)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(combined)
    combined_val = np.concatenate(all_val, axis=0) if all_val else combined[:1000]

    out_train = processed / f"{args.out_prefix}_train.npy"
    out_val = processed / f"{args.out_prefix}_val.npy"
    np.save(out_train, combined)
    np.save(out_val, combined_val)

    total = combined.size
    print(f"\n{'=' * 70}")
    print(f"WROTE {out_train}")
    print(f"  Train: {combined.shape[0]:,} × {combined.shape[1]} = {total/1e9:.2f}B tokens")
    print(f"  Val:   {combined_val.shape[0]:,}")
    print(f"  File size: {combined.nbytes / 1e9:.1f} GB")
    print()
    for t, tok in type_tokens.items():
        print(f"  {t:10s} {tok/1e9:6.2f}B ({tok/total:5.1%})")

    print(f"\nPer-source drop summary:")
    print(f"  {'source':26s} {'initial':>10s} {'filtered':>10s} {'deduped':>10s} {'final':>10s}")
    for name, t, n0, nf, nd, nfinal, _ in stats:
        drop_filter = n0 - nf
        drop_dup = nf - nd
        print(f"  {name:26s} {n0:>10,} -{drop_filter:>9,} -{drop_dup:>9,} {nfinal:>10,}")


if __name__ == "__main__":
    main()
