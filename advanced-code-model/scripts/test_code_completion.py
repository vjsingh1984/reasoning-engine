#!/usr/bin/env python3
"""
Test code completion capabilities (aligned with training data).

This test suite validates what the model ACTUALLY learned:
- Bash syntax completion
- Command pattern recognition
- Code structure understanding

NOT instruction-following (which requires different training data).
"""

import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent))
from model.config import get_config
from model.transformer import create_model
from device import get_device as _get_device


def load_model(checkpoint_path: str, device: str):
    """Load trained model."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    config_dict = checkpoint.get('config', {})
    model_size = checkpoint.get('model_size', '400m')

    # Create model config
    from model.config import ModelConfig
    model_cfg = ModelConfig(
        vocab_size=config_dict.get('vocab_size', 32000),
        d_model=config_dict.get('d_model', 1024),
        n_layers=config_dict.get('n_layers', 26),
        n_heads=config_dict.get('n_heads', 16),
        d_ff=config_dict.get('d_ff', 4096),
        max_seq_len=config_dict.get('max_seq_len', 1024),
    )

    model = create_model(model_cfg, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'Unknown')
    loss = checkpoint.get('val_loss', checkpoint.get('train_loss', checkpoint.get('avg_train_loss', 'Unknown')))

    print(f"  Epoch: {epoch}")
    if isinstance(loss, (int, float)):
        print(f"  Loss: {loss:.4f}")
    else:
        print(f"  Loss: {loss}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, epoch


def complete_code(model, tokenizer, prompt: str, max_tokens: int = 64,
                  temperature: float = 0.5, device: str = "cuda"):
    """Complete code snippet."""
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    prompt_ids = encoded.ids

    # Convert to tensor
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)

    # Generate using model's generate method
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9
        )

    # Decode - output includes prompt
    generated = tokenizer.decode(output_ids[0].tolist())

    # Remove prompt from output
    # The model returns full sequence, so we need to extract just the completion
    prompt_len = len(prompt_ids)
    completion_ids = output_ids[0][prompt_len:].tolist()
    completion = tokenizer.decode(completion_ids)

    return completion


def run_code_completion_tests(model, tokenizer, device: str):
    """Run code completion test suite."""
    print("\n" + "=" * 70)
    print("CODE COMPLETION TEST SUITE")
    print("=" * 70)

    tests = [
        {
            "name": "Loop Completion",
            "prompt": "for file in *.log; do",
            "expected_keywords": ["done", "mv", "rm", "echo"],
        },
        {
            "name": "Function Definition",
            "prompt": "function backup_files() {",
            "expected_keywords": ["}", "tar", "cp", "echo"],
        },
        {
            "name": "Conditional Statement",
            "prompt": "if [ -f \"config.txt\" ]; then",
            "expected_keywords": ["fi", "source", "cat", "echo"],
        },
        {
            "name": "Variable Assignment",
            "prompt": "LOGFILE=",
            "expected_keywords": ["\"", ".log", "/var/", "/tmp/"],
        },
        {
            "name": "Command Pipeline",
            "prompt": "cat file.txt |",
            "expected_keywords": ["grep", "awk", "sed", "sort"],
        },
        {
            "name": "Error Handling",
            "prompt": "command ||",
            "expected_keywords": ["echo", "exit", "rm", "false"],
        },
        {
            "name": "Shebang Start",
            "prompt": "#!/bin/bash\n",
            "expected_keywords": ["echo", "#", "if", "for", "function"],
        },
        {
            "name": "Comment Line",
            "prompt": "# Backup all log files\n",
            "expected_keywords": ["cp", "tar", "mv", "for", "while"],
        },
    ]

    results = []

    for i, test in enumerate(tests, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        print(f"Prompt: {test['prompt']}")

        completion = complete_code(model, tokenizer, test['prompt'], max_tokens=64)
        print(f"Completion:\n{completion}")

        # Check for expected keywords
        found_keywords = [kw for kw in test['expected_keywords'] if kw in completion]
        keyword_match = len(found_keywords) / len(test['expected_keywords'])

        result = {
            "test": test['name'],
            "prompt": test['prompt'],
            "completion": completion,
            "keyword_match": keyword_match,
            "found_keywords": found_keywords,
        }
        results.append(result)

        print(f"Keyword match: {keyword_match:.1%} ({found_keywords})")

    return results


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    avg_keyword_match = sum(r['keyword_match'] for r in results) / len(results)

    print(f"\nAverage keyword match: {avg_keyword_match:.1%}")

    print(f"\nTest Results:")
    for r in results:
        status = "✓" if r['keyword_match'] > 0.3 else "✗"
        print(f"  {status} {r['test']}: {r['keyword_match']:.1%}")

    print(f"\nAssessment:")
    if avg_keyword_match > 0.5:
        print("  ✓ GOOD: Model understands bash patterns")
    elif avg_keyword_match > 0.3:
        print("  ~ FAIR: Model shows some pattern recognition")
    else:
        print("  ✗ POOR: Model needs more training")


def main():
    parser = argparse.ArgumentParser(description="Test code completion capabilities")
    parser.add_argument("--checkpoint", default="models/code_model_periodic.pth",
                        help="Path to checkpoint")
    parser.add_argument("--model-size", default="400m", help="Model size")
    parser.add_argument("--device", default="rocm", help="Device (rocm/cuda/cpu)")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer_path = Path("data/tokenizer/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Load model
    device = _get_device(args.device)
    model, epoch = load_model(args.checkpoint, device)

    # Run tests
    results = run_code_completion_tests(model, tokenizer, device)

    # Print summary
    print_summary(results)

    print(f"\nTraining completion: Epoch {epoch}/10")
    print(f"Note: Results will improve after training completes (Epoch 10)")


if __name__ == "__main__":
    main()
