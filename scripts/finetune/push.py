#!/usr/bin/env python3
"""CLI wrapper for publishing to HuggingFace Hub."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from finetune.config import load_config
from finetune.publisher import HubPublisher


def main():
    parser = argparse.ArgumentParser(description="Push fine-tuned model to HuggingFace Hub")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--adapter-path", default=None, help="Path to adapter")
    parser.add_argument("--no-merge", action="store_true", help="Push adapter only (don't merge)")
    args = parser.parse_args()

    config = load_config(args.config)
    publisher = HubPublisher(config, adapter_path=args.adapter_path)
    publisher.push(merge=not args.no_merge)


if __name__ == "__main__":
    main()
