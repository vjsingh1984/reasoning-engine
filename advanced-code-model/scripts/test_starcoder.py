#!/usr/bin/env python3
"""Test script to check what's available in StarCoderData"""

from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np

base_dir = Path(__file__).parent.parent
tokenizer_path = base_dir / "data" / "tokenizer" / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_path))
eos_id = tokenizer.get_vocab().get("<EOS>", 3)

print("Testing StarCoderData...")
ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)

lang_counts = {}
total_samples = 0
total_tokens = []

for i, item in enumerate(tqdm(ds, desc="Scanning")):
    if i >= 10000:  # Check first 10K samples
        break

    lang = item.get("lang", "unknown")
    content = item.get("content", "")

    if len(content) < 50 or len(content) > 50000:
        continue

    lang_counts[lang] = lang_counts.get(lang, 0) + 1

    # Test tokenization for a few samples
    if i < 100:
        tokens = tokenizer.encode(content).ids
        total_tokens.append(len(tokens))

    total_samples += 1

print(f"\nTotal valid samples (first 10K): {total_samples}")
print(f"\nLanguage breakdown (top 20):")
for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {lang:20s} {count:5d}")

if total_tokens:
    avg_tokens = np.mean(total_tokens)
    print(f"\nAverage tokens per sample: {avg_tokens:.0f}")
    print(f"Estimated 1M samples: {1_000_000 * avg_tokens / 1e9:.2f}B tokens")
