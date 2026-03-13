#!/usr/bin/env python3
"""CLI wrapper for model inference."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from finetune.config import load_config
from finetune.inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default=None, help="Device: cuda, rocm, or cpu")
    parser.add_argument("--adapter-path", default=None, help="Path to adapter")
    parser.add_argument("--prompt", default=None, help="Single prompt (otherwise interactive)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    overrides = {}
    if args.device:
        overrides["device"] = args.device

    config = load_config(args.config, overrides if overrides else None)
    engine = InferenceEngine(config, adapter_path=args.adapter_path)

    if args.prompt:
        response = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(response)
    else:
        engine.interactive()


if __name__ == "__main__":
    main()
