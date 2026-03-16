"""
Prepare datasets for all training stages: language, code, and CoT.

Replaces: prepare_datasets.py, prepare_cot_data.py, generate_cot_local.py

Handles:
  - Language pretraining data (TinyStories batch files)
  - Code fine-tuning data (bash scripts, individually tokenized)
  - CoT instruction-tuning data (download from HF or generate locally)

Usage:
    # Prepare all stages
    python scripts/prepare_data.py

    # Prepare only code data
    python scripts/prepare_data.py --stages code

    # Prepare CoT with HF download + local generation
    python scripts/prepare_data.py --stages cot --download-cot --generate-cot-count 500

    # Custom sequence length
    python scripts/prepare_data.py --max-seq-len 512
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
from tokenizers import Tokenizer


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """Load trained tokenizer."""
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("Run first: python scripts/train_tokenizer.py")
        raise SystemExit(1)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Loaded tokenizer: {tokenizer_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def get_special_ids(tokenizer: Tokenizer) -> dict:
    """Get special token IDs from tokenizer."""
    vocab = tokenizer.get_vocab()
    return {
        "pad": vocab.get("<PAD>", 0),
        "unk": vocab.get("<UNK>", 1),
        "bos": vocab.get("<BOS>", 2),
        "eos": vocab.get("<EOS>", 3),
        "im_start": vocab.get("<|im_start|>", 4),
        "im_end": vocab.get("<|im_end|>", 5),
    }


def tokenize_and_chunk(texts: List[str], tokenizer: Tokenizer, max_seq_len: int,
                       pad_id: int, eos_id: int, concatenate: bool = True) -> np.ndarray:
    """
    Tokenize texts and create fixed-length sequences.

    Args:
        texts: List of text strings
        tokenizer: Trained tokenizer
        max_seq_len: Maximum sequence length
        pad_id: Padding token ID
        eos_id: EOS token ID
        concatenate: If True, concatenate all tokens and chunk (for language data).
                     If False, tokenize each text independently and pad/truncate (for code/CoT).

    Returns:
        numpy array of shape (num_sequences, max_seq_len)
    """
    if concatenate:
        # Concatenate all tokens into one stream, then chunk
        all_tokens = []
        for text in texts:
            encoded = tokenizer.encode(text)
            all_tokens.extend(encoded.ids)
            all_tokens.append(eos_id)

        # Chunk into fixed-length sequences
        num_sequences = len(all_tokens) // max_seq_len
        if num_sequences == 0:
            print(f"  Warning: Not enough tokens ({len(all_tokens)}) for even one sequence of length {max_seq_len}")
            return np.array([], dtype=np.int32).reshape(0, max_seq_len)

        all_tokens = all_tokens[:num_sequences * max_seq_len]
        sequences = np.array(all_tokens, dtype=np.int32).reshape(num_sequences, max_seq_len)
    else:
        # Tokenize each text independently, pad/truncate to max_seq_len
        sequences = []
        for text in texts:
            encoded = tokenizer.encode(text)
            token_ids = encoded.ids[:max_seq_len - 1] + [eos_id]  # Leave room for EOS

            # Pad to max_seq_len
            if len(token_ids) < max_seq_len:
                token_ids = token_ids + [pad_id] * (max_seq_len - len(token_ids))

            sequences.append(token_ids)

        if not sequences:
            return np.array([], dtype=np.int32).reshape(0, max_seq_len)

        sequences = np.array(sequences, dtype=np.int32)

    return sequences


def split_data(sequences: np.ndarray, split_ratio: float):
    """Split sequences into train and val sets."""
    num_train = int(len(sequences) * split_ratio)
    # Shuffle before splitting
    indices = np.random.permutation(len(sequences))
    sequences = sequences[indices]
    return sequences[:num_train], sequences[num_train:]


def prepare_language_dataset(base_dir: Path, tokenizer: Tokenizer, max_seq_len: int,
                              split_ratio: float, output_dir: Path):
    """Prepare language pretraining data from TinyStories batch files."""
    print()
    print("=" * 60)
    print("Preparing LANGUAGE dataset")
    print("=" * 60)

    raw_dir = base_dir / "data" / "language" / "raw"
    batch_files = sorted(raw_dir.glob("batch_*.txt"))
    print(f"Found {len(batch_files)} batch files")

    if not batch_files:
        print("ERROR: No language data found. Run download script first.")
        return

    # Collect all texts
    texts = []
    for bf in batch_files:
        try:
            content = bf.read_text(encoding="utf-8")
            stories = content.strip().split("\n\n")
            texts.extend([s.strip() for s in stories if s.strip()])
        except Exception as e:
            print(f"  Error reading {bf.name}: {e}")
    print(f"Loaded {len(texts):,} text segments")

    special = get_special_ids(tokenizer)
    sequences = tokenize_and_chunk(texts, tokenizer, max_seq_len,
                                   pad_id=special["pad"], eos_id=special["eos"],
                                   concatenate=True)
    print(f"Created {len(sequences):,} sequences of length {max_seq_len}")

    if len(sequences) == 0:
        print("No sequences created. Skipping.")
        return

    train, val = split_data(sequences, split_ratio)
    print(f"Train: {len(train):,} | Val: {len(val):,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "language_train.npy", train)
    np.save(output_dir / "language_val.npy", val)
    print(f"Saved to {output_dir}/language_{{train,val}}.npy")


def prepare_code_dataset(base_dir: Path, tokenizer: Tokenizer, max_seq_len: int,
                          split_ratio: float, output_dir: Path):
    """Prepare code fine-tuning data from bash scripts (individually tokenized)."""
    print()
    print("=" * 60)
    print("Preparing CODE dataset")
    print("=" * 60)

    scripts_dir = base_dir / "data" / "bash" / "raw" / "scripts"
    script_files = sorted(scripts_dir.glob("*.sh"))
    print(f"Found {len(script_files)} bash scripts")

    if not script_files:
        print("ERROR: No code data found. Run download script first.")
        return

    # Read scripts individually
    texts = []
    for sf in script_files:
        try:
            content = sf.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                texts.append(content.strip())
        except Exception as e:
            print(f"  Error reading {sf.name}: {e}")
    print(f"Loaded {len(texts):,} scripts")

    special = get_special_ids(tokenizer)
    sequences = tokenize_and_chunk(texts, tokenizer, max_seq_len,
                                   pad_id=special["pad"], eos_id=special["eos"],
                                   concatenate=False)
    print(f"Created {len(sequences):,} sequences of length {max_seq_len}")

    if len(sequences) == 0:
        print("No sequences created. Skipping.")
        return

    train, val = split_data(sequences, split_ratio)
    print(f"Train: {len(train):,} | Val: {len(val):,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "code_train.npy", train)
    np.save(output_dir / "code_val.npy", val)
    print(f"Saved to {output_dir}/code_{{train,val}}.npy")


def download_cot_from_hf() -> List[dict]:
    """Download CoT-style instruction data from HuggingFace, filter for bash."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        return []

    all_samples = []

    # Dataset 1: CodeAlpaca-20k
    print("  Downloading sahil2801/CodeAlpaca-20k...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        bash_keywords = ["bash", "shell", "script", "command line", "terminal",
                         "linux", "unix", "grep", "awk", "sed", "chmod", "cron"]
        for row in ds:
            instruction = row.get("instruction", "")
            output = row.get("output", "")
            if any(kw in instruction.lower() for kw in bash_keywords):
                all_samples.append({"instruction": instruction, "response": output})
        print(f"    Got {len(all_samples)} bash-related samples")
    except Exception as e:
        print(f"    Failed: {e}")

    # Dataset 2: Evol-Instruct-Code-80k
    count_before = len(all_samples)
    print("  Downloading nickrosh/Evol-Instruct-Code-80k-v1...")
    try:
        ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
        bash_keywords = ["bash", "shell", "script", "command line", "terminal",
                         "linux", "unix", "grep", "awk", "sed"]
        for row in ds:
            instruction = row.get("instruction", "")
            output = row.get("output", "")
            if any(kw in instruction.lower() for kw in bash_keywords):
                all_samples.append({"instruction": instruction, "response": output})
        print(f"    Got {len(all_samples) - count_before} bash-related samples")
    except Exception as e:
        print(f"    Failed: {e}")

    # Dataset 3: glaive-code-assistant-v2
    count_before = len(all_samples)
    print("  Downloading glaiveai/glaive-code-assistant-v2...")
    try:
        ds = load_dataset("glaiveai/glaive-code-assistant-v2", split="train")
        bash_keywords = ["bash", "shell", "script", "command line", "terminal",
                         "linux", "unix"]
        for row in ds:
            question = row.get("question", "")
            answer = row.get("answer", "")
            if any(kw in question.lower() for kw in bash_keywords):
                all_samples.append({"instruction": question, "response": answer})
        print(f"    Got {len(all_samples) - count_before} bash-related samples")
    except Exception as e:
        print(f"    Failed: {e}")

    print(f"  Total downloaded: {len(all_samples)} samples")
    return all_samples


def format_chatml(instruction: str, response: str,
                  system_prompt: str = "You are a bash scripting expert.") -> str:
    """Format instruction/response pair as ChatML string."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def prepare_cot_dataset(base_dir: Path, tokenizer: Tokenizer, max_seq_len: int,
                         split_ratio: float, output_dir: Path,
                         download_cot: bool = False, generate_cot_count: int = 0):
    """Prepare CoT instruction-tuning data."""
    print()
    print("=" * 60)
    print("Preparing COT dataset")
    print("=" * 60)

    all_samples = []

    # Source 1: Existing local CoT JSON (bash-specific)
    cot_path = base_dir / "data" / "cot" / "all_bash_cot.json"
    if cot_path.exists():
        with open(cot_path, "r", encoding="utf-8") as f:
            local_samples = json.load(f)
        print(f"Loaded {len(local_samples)} bash CoT samples from {cot_path.name}")
        all_samples.extend(local_samples)

    # Source 2: Open-source bash instruction data
    bash_instruct_path = base_dir / "data" / "cot" / "opensource_bash_instructions.json"
    if bash_instruct_path.exists():
        with open(bash_instruct_path, "r", encoding="utf-8") as f:
            bash_samples = json.load(f)
        print(f"Loaded {len(bash_samples)} open-source bash instruction samples")
        all_samples.extend(bash_samples)

    # Source 3: SlimOrca general instruction data
    slimorca_path = base_dir / "data" / "cot" / "slimorca_300k.json"
    if slimorca_path.exists():
        with open(slimorca_path, "r", encoding="utf-8") as f:
            slimorca_samples = json.load(f)
        print(f"Loaded {len(slimorca_samples)} SlimOrca samples")
        all_samples.extend(slimorca_samples)

    # Source 4: Download from HuggingFace
    if download_cot:
        print("Downloading CoT data from HuggingFace...")
        hf_samples = download_cot_from_hf()
        all_samples.extend(hf_samples)

    # Source 5: Generate local template-based samples
    if generate_cot_count > 0:
        print(f"Generating {generate_cot_count} template-based CoT samples...")
        try:
            from utils.cot_templates import generate_cot_samples
            generated = generate_cot_samples(generate_cot_count)
            all_samples.extend(generated)
            print(f"  Generated {len(generated)} samples")
        except ImportError as e:
            print(f"  ERROR: Could not import cot_templates: {e}")
            print("  Skipping local generation.")

    if not all_samples:
        print("No CoT data available. Skipping.")
        print("Use --download-cot or --generate-cot-count to get data.")
        return

    print(f"Total CoT samples: {len(all_samples)}")

    # Deduplicate by instruction
    seen = set()
    unique_samples = []
    for s in all_samples:
        key = s.get("instruction", "")[:200]
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)
    print(f"After dedup: {len(unique_samples)} unique samples")

    # Format as ChatML
    texts = []
    for sample in unique_samples:
        instruction = sample.get("instruction", "")
        response = sample.get("response", "")
        if instruction and response:
            texts.append(format_chatml(instruction, response))

    print(f"Formatted {len(texts)} ChatML strings")

    # Shuffle
    random.shuffle(texts)

    special = get_special_ids(tokenizer)
    sequences = tokenize_and_chunk(texts, tokenizer, max_seq_len,
                                   pad_id=special["pad"], eos_id=special["eos"],
                                   concatenate=False)
    print(f"Created {len(sequences):,} sequences of length {max_seq_len}")

    if len(sequences) == 0:
        print("No sequences created. Skipping.")
        return

    train, val = split_data(sequences, split_ratio)
    print(f"Train: {len(train):,} | Val: {len(val):,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cot_train.npy", train)
    np.save(output_dir / "cot_val.npy", val)
    print(f"Saved to {output_dir}/cot_{{train,val}}.npy")

    # Also save the raw ChatML JSON for reference
    cot_dir = base_dir / "data" / "cot"
    cot_dir.mkdir(parents=True, exist_ok=True)
    with open(cot_dir / "all_bash_cot.json", "w", encoding="utf-8") as f:
        json.dump(unique_samples, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for all training stages")
    parser.add_argument("--stages", type=str, nargs="+",
                        choices=["language", "code", "cot", "all"], default=["all"],
                        help="Which stages to prepare (default: all)")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                        help="Maximum sequence length (default: 1024)")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to tokenizer.json (default: data/tokenizer/tokenizer.json)")
    parser.add_argument("--download-cot", action="store_true",
                        help="Download CoT data from HuggingFace")
    parser.add_argument("--generate-cot-count", type=int, default=0,
                        help="Number of template-based CoT samples to generate locally")
    parser.add_argument("--split-ratio", type=float, default=0.95,
                        help="Train/val split ratio (default: 0.95)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Resolve tokenizer path
    if args.tokenizer_path:
        tokenizer_path = Path(args.tokenizer_path)
    else:
        tokenizer_path = base_dir / "data" / "tokenizer" / "tokenizer.json"

    tokenizer = load_tokenizer(tokenizer_path)
    output_dir = base_dir / "data" / "processed"

    # Determine which stages to run
    stages = set(args.stages)
    if "all" in stages:
        stages = {"language", "code", "cot"}

    print()
    print(f"Stages: {', '.join(sorted(stages))}")
    print(f"Sequence length: {args.max_seq_len}")
    print(f"Split ratio: {args.split_ratio}")

    if "language" in stages:
        prepare_language_dataset(base_dir, tokenizer, args.max_seq_len,
                                  args.split_ratio, output_dir)

    if "code" in stages:
        prepare_code_dataset(base_dir, tokenizer, args.max_seq_len,
                              args.split_ratio, output_dir)

    if "cot" in stages:
        prepare_cot_dataset(base_dir, tokenizer, args.max_seq_len,
                             args.split_ratio, output_dir,
                             download_cot=args.download_cot,
                             generate_cot_count=args.generate_cot_count)

    # Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "max_seq_len": args.max_seq_len,
        "split_ratio": args.split_ratio,
        "stages_prepared": sorted(stages),
        "tokenizer_path": str(tokenizer_path),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print()
    print("Next: python scripts/train.py --stage <language|code|cot>")


if __name__ == "__main__":
    main()
