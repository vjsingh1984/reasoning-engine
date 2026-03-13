#!/usr/bin/env python3
"""CLI wrapper for QLoRA fine-tuning."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from finetune.config import load_config
from finetune.trainer import CoTTrainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with QLoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default=None, help="Device: cuda, rocm, or cpu")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 for full)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.batch_size:
        overrides["training.per_device_batch_size"] = args.batch_size
    if args.lr:
        overrides["training.learning_rate"] = args.lr

    config = load_config(args.config, overrides if overrides else None)
    trainer = CoTTrainer(config)
    trainer.train(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
