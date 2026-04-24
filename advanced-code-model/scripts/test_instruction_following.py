#!/usr/bin/env python3
"""Test instruction-following capabilities.

Evaluates how well the model responds to instruction-style prompts
vs. completion-style prompts. Uses chat format tokens.
"""

import argparse
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent))
from model.config import ModelConfig
from model.transformer import create_model
from device import get_device as _get_device


SYSTEM_PROMPT = "<|im_start|>system\nYou are a helpful coding assistant. Write clear, correct code.<|im_start|>user\n"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
END_TOKEN = "<|im_end|>\n"


TESTS = [
    ("Simple function", "Write a Python function to calculate the factorial of a number."),
    ("Bash script", "Write a bash script that backs up all .log files older than 7 days."),
    ("Loop", "Write a for loop in Python that prints numbers 1 to 10."),
    ("Array sum", "Write a function to find the sum of an array of integers."),
    ("File read", "Write a bash one-liner to count lines in a file."),
    ("Conditional", "Write a Python function that returns 'even' or 'odd' for a given number."),
    ("Error handling", "Write a bash script that checks if a file exists and prints a message."),
    ("List comprehension", "Write a Python list comprehension to square even numbers from 1 to 10."),
]


def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    c = ckpt.get("config", {})
    cfg = ModelConfig(
        vocab_size=c.get("vocab_size", 32000),
        d_model=c.get("d_model", 1024),
        n_layers=c.get("n_layers", 26),
        n_heads=c.get("n_heads", 16),
        d_ff=c.get("d_ff", 4096),
        max_seq_len=c.get("max_seq_len", 1024),
    )
    model = create_model(cfg, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def generate_with_chat(model, tokenizer, instruction: str, max_tokens: int = 256,
                       temperature: float = 0.7, device: str = "cuda"):
    """Generate response to instruction using chat format."""
    prompt = SYSTEM_PROMPT + instruction + "\n" + ASSISTANT_PREFIX
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9
        )

    # Decode response (skip prompt)
    response_ids = output[0, len(ids):].tolist()
    response = tokenizer.decode(response_ids)

    # Clean up: stop at end token or next user message
    if END_TOKEN.strip() in response:
        response = response.split(END_TOKEN.strip())[0]
    if "<|im_start|>" in response:
        response = response.split("<|im_start|>")[0]

    return response.strip()


def run_tests(model, tokenizer, device):
    print("\n" + "=" * 70)
    print("INSTRUCTION-FOLLOWING TEST")
    print("=" * 70)

    results = []
    for name, instruction in TESTS:
        print(f"\n--- {name} ---")
        print(f"Instruction: {instruction}")

        try:
            response = generate_with_chat(model, tokenizer, instruction, device=device)
            print(f"Response:\n{response[:400]}")

            # Basic quality checks
            has_code = any(x in response for x in ["def ", "function", "#!/bin", "for ", "if ", "class "])
            not_empty = len(response.strip()) > 10
            not_repetitive = len(set(response.split())) > len(response.split()) * 0.5

            score = (has_code + not_empty + not_repetitive) / 3
            results.append((name, score, has_code, not_empty, not_repetitive))
            print(f"\nScore: {score:.1%} (code={has_code}, not_empty={not_empty}, not_repetitive={not_repetitive})")
        except Exception as e:
            print(f"Error: {e}")
            results.append((name, 0.0, False, False, False))

    return results


def print_summary(results):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_score = sum(r[1] for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.1%}")

    print(f"\nTest Results:")
    for name, score, code, empty, rep in results:
        status = "✓" if score > 0.5 else "~" if score > 0.2 else "✗"
        print(f"  {status} {name}: {score:.1%}")

    print(f"\nAssessment:")
    if avg_score > 0.7:
        print("  ✓ GOOD: Model follows instructions well")
    elif avg_score > 0.4:
        print("  ~ FAIR: Model shows some instruction-following ability")
    else:
        print("  ✗ POOR: Model struggles with instructions (needs SFT)")


def main():
    ap = argparse.ArgumentParser(description="Test instruction-following")
    ap.add_argument("--checkpoint", default="models/code_model_latest.pth")
    ap.add_argument("--device", default="rocm")
    args = ap.parse_args()

    device = _get_device(args.device)
    tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")
    model, _ = load_model(args.checkpoint, device)

    results = run_tests(model, tokenizer, device)
    print_summary(results)


if __name__ == "__main__":
    main()
