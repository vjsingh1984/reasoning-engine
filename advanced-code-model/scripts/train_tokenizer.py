"""
Train BPE tokenizer on combined language + code + optional CoT corpus.

Uses HuggingFace tokenizers library with ByteLevel pre-tokenizer
for proper handling of code (preserves whitespace, special chars).

Special tokens:
  0: <PAD>    - Padding
  1: <UNK>    - Unknown
  2: <BOS>    - Begin of sequence
  3: <EOS>    - End of sequence
  4: <|im_start|> - ChatML start
  5: <|im_end|>   - ChatML end

Usage:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab-size 16000 --include-cot
"""

import argparse
import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


def iter_language_data(data_dir: Path) -> Iterator[str]:
    """Yield text chunks from language corpus (TinyStories batch files)."""
    raw_dir = data_dir / "raw"
    batch_files = sorted(raw_dir.glob("batch_*.txt"))
    print(f"Found {len(batch_files)} language batch files")

    for batch_file in batch_files:
        try:
            with open(batch_file, "r", encoding="utf-8") as f:
                content = f.read()
            stories = content.strip().split("\n\n")
            for story in stories:
                if story.strip():
                    yield story.strip()
        except Exception as e:
            print(f"Error reading {batch_file}: {e}")
            continue


def iter_code_data(data_dir: Path) -> Iterator[str]:
    """Yield script contents from bash corpus."""
    scripts_dir = data_dir / "raw" / "scripts"
    script_files = sorted(scripts_dir.glob("*.sh"))
    print(f"Found {len(script_files)} bash scripts")

    for script_file in script_files:
        try:
            with open(script_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if content.strip():
                yield content.strip()
        except Exception as e:
            print(f"Error reading {script_file}: {e}")
            continue


def iter_cot_data(cot_path: Path) -> Iterator[str]:
    """Yield ChatML-formatted strings from CoT JSON data."""
    if not cot_path.exists():
        print(f"CoT data not found: {cot_path}")
        return

    with open(cot_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Found {len(samples)} CoT samples")

    for sample in samples:
        instruction = sample.get("instruction", "")
        response = sample.get("response", "")
        if instruction and response:
            chatml = (
                f"<|im_start|>system\nYou are a bash scripting expert.<|im_end|>\n"
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
            )
            yield chatml


def data_iterator(base_dir: Path, include_cot: bool = False) -> Iterator[str]:
    """Combined iterator over all data sources for memory-efficient training."""
    language_dir = base_dir / "data" / "language"
    code_dir = base_dir / "data" / "bash"

    lang_count = 0
    for text in iter_language_data(language_dir):
        lang_count += 1
        yield text
    print(f"  Yielded {lang_count:,} language examples")

    code_count = 0
    for text in iter_code_data(code_dir):
        code_count += 1
        yield text
    print(f"  Yielded {code_count:,} code examples")

    if include_cot:
        cot_path = base_dir / "data" / "cot" / "all_bash_cot.json"
        cot_count = 0
        for text in iter_cot_data(cot_path):
            cot_count += 1
            yield text
        print(f"  Yielded {cot_count:,} CoT examples")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on language + code corpus")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Target vocabulary size (default: 32000)")
    parser.add_argument("--include-cot", action="store_true",
                        help="Include CoT data (data/cot/all_bash_cot.json) formatted as ChatML")
    parser.add_argument("--output-dir", type=str, default="data/tokenizer",
                        help="Output directory relative to advanced-code-model/ (default: data/tokenizer)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / args.output_dir

    print("=" * 60)
    print("Training BPE Tokenizer (ByteLevel)")
    print("=" * 60)
    print()
    print(f"Vocabulary size: {args.vocab_size:,}")
    print(f"Include CoT: {args.include_cot}")
    print(f"Output: {output_dir}")
    print()

    # Special tokens (order determines IDs: 0, 1, 2, 3, 4, 5)
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<|im_start|>", "<|im_end|>"]

    # Initialize tokenizer with ByteLevel for proper code handling
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train from iterator (memory efficient)
    print("Training tokenizer from data iterator...")
    print()
    tokenizer.train_from_iterator(
        data_iterator(base_dir, include_cot=args.include_cot),
        trainer=trainer,
    )

    print()
    print("Tokenizer training complete!")
    print()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_file = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    print(f"Saved tokenizer to {tokenizer_file}")

    # Verify special token IDs
    vocab = tokenizer.get_vocab()
    print()
    print("Special token IDs:")
    for tok in special_tokens:
        tok_id = vocab.get(tok, "MISSING")
        print(f"  {tok}: {tok_id}")

    # Test encoding
    print()
    print("Test encodings:")
    test_cases = [
        "Hello, world!",
        '#!/bin/bash\necho "Hello"',
        "for i in $(seq 1 10); do echo $i; done",
        "<|im_start|>user\nWrite a script<|im_end|>",
    ]
    for text in test_cases:
        encoded = tokenizer.encode(text)
        print(f"  Input:  {text!r}")
        print(f"  Tokens: {encoded.tokens[:15]}{'...' if len(encoded.tokens) > 15 else ''}")
        print(f"  IDs:    {encoded.ids[:15]}{'...' if len(encoded.ids) > 15 else ''}")
        print()

    # Save metadata
    metadata = {
        "vocab_size": len(vocab),
        "special_tokens": {tok: vocab.get(tok) for tok in special_tokens},
        "include_cot": args.include_cot,
        "pre_tokenizer": "ByteLevel",
    }
    metadata_file = output_dir / "tokenizer_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")

    print()
    print("=" * 60)
    print("Done! Next: python scripts/prepare_data.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
