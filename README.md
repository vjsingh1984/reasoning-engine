# LLM From Scratch - Learning Journey

Build and understand Large Language Models from the ground up, with GPU acceleration on NVIDIA CUDA and AMD ROCm.

## Project Goals

1. **Learn by Building**: Implement every component of an LLM from scratch
2. **Understand Fundamentals**: Tokenization, attention mechanisms, transformer architecture
3. **Practical Training**: Train models on your GPU with manageable parameter counts
4. **Document Everything**: Step-by-step guides for each component

## Technology Stack

- **Framework**: PyTorch 2.2+ (supports CUDA and ROCm via the same API)
- **Language**: Python 3.10+
- **Target Hardware**: NVIDIA GPUs (CUDA) or AMD GPUs (ROCm), with CPU fallback
- **Model Sizes**: ~23M (tiny) up to 774M+ parameters

## Project Structure

```
llm-from-scratch/
├── docs/               # Step-by-step documentation
├── device.py           # Device abstraction (CUDA/ROCm/CPU auto-detection)
├── tokenizer/          # BPE tokenizer implementation
│   ├── bpe.py
│   └── vocab.py
├── model/              # Transformer architecture
│   ├── attention.py    # Multi-head, GQA, and Flash attention
│   ├── embedding.py    # Token + positional embeddings (learned, sinusoidal, RoPE)
│   └── transformer.py  # TransformerBlock, GPTModel, GPTConfig
├── training/           # Training infrastructure
│   ├── trainer.py      # Training loop with AMP, checkpointing
│   ├── optimizer.py    # AdamW + LR schedules (cosine, linear, warmup)
│   └── data_loader.py  # PyTorch DataLoader integration
├── finetune/           # QLoRA fine-tuning framework
│   ├── config.py       # Config dataclass + YAML loader
│   ├── data.py         # Dataset loading & ChatML formatting
│   ├── quantization.py # BitsAndBytes 4-bit quantization
│   ├── lora.py         # LoRA adapter management
│   ├── trainer.py      # CoTTrainer (wraps trl.SFTTrainer)
│   ├── evaluator.py    # Benchmark evaluation (lm-eval)
│   ├── inference.py    # Generation + interactive chat
│   ├── publisher.py    # HuggingFace Hub publishing
│   └── model_card.py   # Model card generation
├── configs/            # YAML configs for fine-tuning
│   ├── qwen2.5-1.5b-cot.yaml
│   └── llama3.2-1b-cot.yaml
└── scripts/            # Executable scripts
    ├── download_data.py
    ├── train_tokenizer.py
    ├── train.py
    ├── generate.py
    ├── test_model.py
    └── finetune/       # Fine-tuning CLI wrappers
        ├── train.py
        ├── evaluate.py
        ├── inference.py
        └── push.py
```

## Quick Start

```bash
# 1. Install PyTorch for your GPU
# NVIDIA CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu124
# AMD ROCm:
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

# 2. Install other dependencies
pip install -r requirements.txt

# 3. Download data
python scripts/download_data.py --num-samples 10000

# 4. Train tokenizer
python scripts/train_tokenizer.py --vocab-size 8000

# 5. Test model architecture (auto-detects GPU)
python scripts/test_model.py
python scripts/test_model.py --device cuda   # Force NVIDIA
python scripts/test_model.py --device rocm   # Force AMD

# 6. Train model
python scripts/train.py \
    --model-size tiny \
    --batch-size 16 \
    --max-steps 10000 \
    --device cuda       # or rocm, or cpu

# 7. Generate text
python scripts/generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "Once upon a time" \
    --interactive
```

## Device Selection

All scripts accept a `--device` flag:

| Flag | GPU | Notes |
|------|-----|-------|
| `--device cuda` | NVIDIA (CUDA) | RTX 4000 Ada, etc. |
| `--device rocm` | AMD (ROCm) | RX 9700 XT AI Pro, etc. |
| `--device cpu` | CPU only | Fallback, slower |
| *(omitted)* | Auto-detect | Picks GPU if available |

Both CUDA and ROCm use PyTorch's `torch.cuda` API — the common code path is fully shared. Only the PyTorch installation differs.

## Learning Roadmap

### Phase 1: Foundations
- [x] Project setup
- [x] Understanding tokenization
- [x] Implementing BPE tokenizer
- [x] Building vocabulary from sample data

### Phase 2: Model Architecture
- [x] Embeddings and positional encoding (learned, sinusoidal, RoPE)
- [x] Multi-head self-attention with causal masking
- [x] Transformer blocks with Pre-LayerNorm
- [x] Complete GPT-style model (23M–774M+ params)
- [x] Grouped Query Attention support
- [x] Flash Attention via PyTorch SDPA

### Phase 3: Training
- [x] Data preprocessing pipeline
- [x] Training loop with AdamW optimizer
- [x] Learning rate scheduling (warmup + cosine decay)
- [x] Gradient clipping
- [x] Automatic checkpointing
- [x] Automatic mixed precision (AMP) support

### Phase 4: Scaling & Evaluation
- [x] Text generation with sampling strategies (top-k, top-p)
- [x] Generation script with interactive mode
- [x] Training scripts for all model sizes
- [x] Multi-GPU vendor support (NVIDIA + AMD)

### Phase 5: Fine-Tuning
- [x] QLoRA fine-tuning framework (4-bit NF4 + LoRA)
- [x] Config-driven training with YAML configs
- [x] Chain-of-thought reasoning (MetaMathQA + OpenOrca)
- [x] Benchmark evaluation (GSM8K, ARC-Challenge)
- [x] Interactive inference with streaming
- [x] HuggingFace Hub publishing with model cards

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ROCm Documentation](https://rocm.docs.amd.com/)

---

**Next Step**: Read `docs/00-OVERVIEW.md` to understand the architecture fundamentals.
