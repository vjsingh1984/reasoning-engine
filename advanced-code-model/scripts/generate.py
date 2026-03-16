"""
Generate bash scripts using a trained model with ChatML formatting.

Features:
  - Custom generation with repetition penalty (model.generate() lacks it)
  - Top-k + top-p (nucleus) sampling
  - ChatML prompt formatting
  - Single prompt, interactive, and test modes

Usage:
    # Single prompt
    python scripts/generate.py --checkpoint models/cot_model_best.pth \\
        --prompt "Write a script to find large files"

    # Interactive mode
    python scripts/generate.py --checkpoint models/cot_model_best.pth --interactive

    # Test mode (5 built-in prompts)
    python scripts/generate.py --checkpoint models/cot_model_best.pth --test
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Path setup: src first, then repo root (avoids MLX model/ conflict)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent))
from model.config import get_config
from model.transformer import create_model
from device import get_device as _get_device, print_device_info


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """Load trained tokenizer."""
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        raise SystemExit(1)
    return Tokenizer.from_file(str(tokenizer_path))


def format_prompt(instruction: str, system_prompt: str) -> str:
    """Format instruction as ChatML prompt for the model."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@torch.no_grad()
def generate_with_penalty(
    model,
    prompt_ids: list,
    tokenizer: Tokenizer,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: torch.device = None,
) -> str:
    """
    Generate text token-by-token with repetition penalty.

    The model's built-in generate() does not support repetition penalty,
    so we implement our own generation loop here.

    Args:
        model: Trained transformer model
        prompt_ids: Token IDs for the prompt
        tokenizer: Tokenizer for decoding
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Top-p (nucleus) filtering
        repetition_penalty: Penalty for repeating tokens (>1.0 penalizes)
        device: Compute device

    Returns:
        Generated text string (assistant response only)
    """
    vocab = tokenizer.get_vocab()
    stop_ids = set()
    for tok_name in ["<|im_end|>", "<EOS>", "<PAD>"]:
        if tok_name in vocab:
            stop_ids.add(vocab[tok_name])

    # Get model's max sequence length from the input embedding or config
    max_seq_len = getattr(model, "max_seq_len", 1024)
    if hasattr(model, "config"):
        max_seq_len = model.config.max_seq_len
    elif hasattr(model, "pos_embedding"):
        max_seq_len = model.pos_embedding.weight.shape[0]

    generated_ids = list(prompt_ids)

    for _ in range(max_tokens):
        # Truncate to max_seq_len (keep most recent tokens)
        input_ids = generated_ids[-max_seq_len:]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Forward pass
        logits = model(input_tensor)  # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply repetition penalty: divide logits of already-seen tokens
        if repetition_penalty != 1.0:
            seen_tokens = set(generated_ids)
            for token_id in seen_tokens:
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

        # Temperature scaling
        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            # Greedy
            next_token = next_logits.argmax().item()
            generated_ids.append(next_token)
            if next_token in stop_ids:
                break
            continue

        # Top-k filtering
        if top_k > 0:
            top_k_vals, top_k_idx = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            mask = torch.full_like(next_logits, float("-inf"))
            mask.scatter_(0, top_k_idx, top_k_vals)
            next_logits = mask

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")

            # Scatter back
            next_logits = torch.full_like(next_logits, float("-inf"))
            next_logits.scatter_(0, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_token)

        if next_token in stop_ids:
            break

    # Decode only the generated portion (after prompt)
    output_ids = generated_ids[len(prompt_ids):]
    # Remove stop token from output if present
    for i, tid in enumerate(output_ids):
        if tid in stop_ids:
            output_ids = output_ids[:i]
            break

    return tokenizer.decode(output_ids)


def run_single(model, tokenizer, prompt: str, system_prompt: str, device, args):
    """Generate for a single prompt."""
    formatted = format_prompt(prompt, system_prompt)
    prompt_ids = tokenizer.encode(formatted).ids

    print(f"\nPrompt: {prompt}")
    print("-" * 60)

    response = generate_with_penalty(
        model=model,
        prompt_ids=prompt_ids,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )

    print(response)
    print("-" * 60)
    return response


def run_interactive(model, tokenizer, system_prompt: str, device, args):
    """Interactive generation loop."""
    print("\nInteractive mode (type 'quit' or 'exit' to stop)")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        run_single(model, tokenizer, prompt, system_prompt, device, args)


def run_test(model, tokenizer, system_prompt: str, device, args):
    """Run 5 built-in test prompts."""
    test_prompts = [
        "Write a bash script that checks if a file exists and prints its size.",
        "Create a script to find all files larger than 100MB in the current directory.",
        "Write a bash function that validates an IP address.",
        "Create a script that monitors CPU usage and sends an alert if it exceeds 90%.",
        "Write a bash script to backup a MySQL database with timestamp in the filename.",
    ]

    print("\nRunning test generation with 5 built-in prompts")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/5]")
        run_single(model, tokenizer, prompt, system_prompt, device, args)
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate bash scripts with trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth file)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer.json (auto-detects if not specified)")
    parser.add_argument("--model-size", type=str, default="250m",
                        choices=["small", "tiny", "250m", "medium", "large", "xlarge"],
                        help="Model size config (default: 250m)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, rocm, cpu, or auto-detect")

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prompt", type=str, default=None,
                      help="Single prompt to generate from")
    mode.add_argument("--interactive", action="store_true",
                      help="Interactive chat mode")
    mode.add_argument("--test", action="store_true",
                      help="Run 5 built-in test prompts")

    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k filtering (default: 50)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p nucleus filtering (default: 0.9)")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="Repetition penalty (default: 1.2, 1.0=disabled)")
    parser.add_argument("--system-prompt", type=str,
                        default="You are a bash scripting expert. Write clean, well-commented bash scripts.",
                        help="System prompt for ChatML formatting")

    args = parser.parse_args()

    # Validate mode
    if not args.prompt and not args.interactive and not args.test:
        parser.error("Specify one of: --prompt, --interactive, or --test")

    # Setup device
    device = _get_device(args.device)
    print_device_info(device)
    print()

    base_dir = Path(__file__).parent.parent

    # Load tokenizer
    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
    else:
        # Auto-detect: check common locations
        candidates = [
            base_dir / "data" / "tokenizer" / "tokenizer.json",
            base_dir / "tokenizer.json",
        ]
        tokenizer_path = None
        for c in candidates:
            if c.exists():
                tokenizer_path = c
                break
        if tokenizer_path is None:
            print("ERROR: Could not find tokenizer.json. Use --tokenizer to specify path.")
            raise SystemExit(1)

    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Loaded tokenizer: {tokenizer_path} (vocab: {tokenizer.get_vocab_size()})")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        raise SystemExit(1)

    print(f"Loading model ({args.model_size})...")
    config = get_config(args.model_size)
    model = create_model(config, device=str(device))

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Handle torch.compile wrapped models
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        print("  Stripping _orig_mod. prefix from compiled model...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    if "metadata" in checkpoint:
        meta = checkpoint["metadata"]
        if "val_loss" in meta:
            print(f"  Val loss: {meta['val_loss']:.4f}")
        if "epoch" in meta:
            print(f"  Epoch: {meta['epoch']}")

    # Run selected mode
    if args.prompt:
        run_single(model, tokenizer, args.prompt, args.system_prompt, device, args)
    elif args.interactive:
        run_interactive(model, tokenizer, args.system_prompt, device, args)
    elif args.test:
        run_test(model, tokenizer, args.system_prompt, device, args)


if __name__ == "__main__":
    main()
