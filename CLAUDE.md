# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational repository for building an LLM from scratch using PyTorch, with GPU acceleration on both NVIDIA CUDA and AMD ROCm. Implements the full ML pipeline: tokenization, transformer architecture, training, and text generation.

## Architecture

- **`device.py`**: Device abstraction layer — auto-detects CUDA/ROCm/CPU, provides `get_device(device_str)` used by all scripts via `--device` CLI flag. ROCm maps to `torch.cuda` API so common code is shared; only the PyTorch install differs.
- **`model/`**: GPT-style transformer (PyTorch `nn.Module`)
  - `embedding.py`: Token + positional embeddings (learned, sinusoidal, RoPE)
  - `attention.py`: MultiHeadAttention, GroupedQueryAttention, FlashAttention (SDPA)
  - `transformer.py`: GPTConfig, TransformerBlock, GPTModel, `create_model()` factory
- **`tokenizer/`**: Byte Pair Encoding (no PyTorch dependency, pure Python + regex)
  - `bpe.py`: BPETokenizer with train/encode/decode/save/load
  - `vocab.py`: Vocabulary with special tokens (pad, eos, unk, bos)
- **`training/`**: Training infrastructure
  - `data_loader.py`: PyTorch Dataset/DataLoader with pin_memory for GPU
  - `optimizer.py`: AdamW with proper weight decay groups + LR schedules
  - `trainer.py`: Training loop with AMP, gradient clipping, checkpointing
- **`finetune/`**: QLoRA fine-tuning framework (uses HuggingFace ecosystem)
  - `config.py`: FinetuneConfig dataclass + YAML loader, auto-detects device via `device.py`
  - `data.py`: DatasetLoader with pluggable formatters, outputs ChatML format
  - `quantization.py`: BitsAndBytes 4-bit config with ROCm/CUDA blocksize auto-detection
  - `lora.py`: LoRA adapter lifecycle (apply, save, load, merge)
  - `trainer.py`: CoTTrainer orchestrating quantization + LoRA + data + trl.SFTTrainer
  - `evaluator.py`: Wraps lm-eval for benchmark evaluation
  - `inference.py`: InferenceEngine with generate/stream/interactive modes
  - `publisher.py`: Merge adapter + push to HuggingFace Hub
  - `model_card.py`: Template-based model card generation
- **`scripts/`**: CLI entry points (all accept `--device cuda|rocm|cpu`)
  - `finetune/`: Thin CLI wrappers for fine-tuning (train, evaluate, inference, push)

## Common Commands

```bash
# Install PyTorch (pick one):
pip install torch --index-url https://download.pytorch.org/whl/cu124    # NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2  # AMD

# Install dependencies
pip install -r requirements.txt

# Test model on GPU
python scripts/test_model.py --device cuda

# Full pipeline
python scripts/download_data.py --num-samples 10000
python scripts/train_tokenizer.py --vocab-size 8000
python scripts/train.py --model-size tiny --batch-size 16 --max-steps 10000 --device cuda
python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time" --interactive

# Fine-tuning (QLoRA)
pip install -r requirements-finetune.txt
python scripts/finetune/train.py --config configs/qwen2.5-1.5b-cot.yaml --device rocm
python scripts/finetune/evaluate.py --config configs/qwen2.5-1.5b-cot.yaml --device rocm
python scripts/finetune/inference.py --config configs/qwen2.5-1.5b-cot.yaml --prompt "What is 15% of 240?" --device rocm
python scripts/finetune/push.py --config configs/qwen2.5-1.5b-cot.yaml
```

## Key Design Decisions

- **Shared code path**: CUDA and ROCm use identical code via `torch.cuda` — the `device.py` module handles detection and the `--device rocm` flag is an alias for `cuda`.
- **Checkpoints**: Saved as `.pt` files containing `model_state_dict`, `optimizer_state_dict`, training state. Cross-device compatible via `map_location`.
- **AMP**: Optional `--use-amp` flag enables float16 mixed precision on GPU (both CUDA and ROCm).
- **Flash Attention**: `--use-flash-attn` uses `F.scaled_dot_product_attention` which auto-selects the best kernel per hardware.
- **Fine-tuning**: Config-driven QLoRA with YAML configs in `configs/`. Modules in `finetune/` are importable; scripts in `scripts/finetune/` are thin CLI wrappers. Quantization blocksize auto-detected (128 for ROCm, 64 for CUDA).

## Git Conventions

- Large files (model checkpoints, datasets, `.pt`, `.npz`, `.safetensors`) are excluded via `.gitignore` — never commit them.
