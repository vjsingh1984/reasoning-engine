#!/usr/bin/env python3
"""CLI wrapper for model evaluation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from finetune.config import load_config
from finetune.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default=None, help="Device: cuda, rocm, or cpu")
    parser.add_argument("--adapter-path", default=None, help="Path to adapter (default: from config)")
    parser.add_argument("--tasks", nargs="+", default=None, help="Eval tasks to run")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    overrides = {}
    if args.device:
        overrides["device"] = args.device

    config = load_config(args.config, overrides if overrides else None)
    evaluator = Evaluator(config)

    adapter_path = args.adapter_path or str(
        Path(config.training.output_dir) / "adapter"
    )

    results = evaluator.run(model_path=adapter_path, tasks=args.tasks)

    if args.output:
        evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
