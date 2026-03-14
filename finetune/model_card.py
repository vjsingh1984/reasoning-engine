"""Model card generation from training config and eval results."""

from finetune.config import FinetuneConfig


class ModelCardGenerator:
    """Generates a HuggingFace model card from config and evaluation metrics."""

    @staticmethod
    def generate(
        config: FinetuneConfig,
        eval_results: dict = None,
        train_stats: dict = None,
    ) -> str:
        """Generate a model card markdown string."""
        sources = ", ".join(s.name for s in config.data.sources)
        sample_counts = " + ".join(
            f"{s.max_samples or 'all'} from {s.name}" for s in config.data.sources
        )

        eval_table = ""
        if eval_results and "results" in eval_results:
            rows = []
            for task, metrics in eval_results["results"].items():
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        rows.append(f"| {task} | {metric} | {value:.4f} |")
            if rows:
                eval_table = (
                    "## Evaluation Results\n\n"
                    "| Task | Metric | Score |\n"
                    "|------|--------|-------|\n"
                    + "\n".join(rows)
                    + "\n"
                )

        train_results = ""
        if train_stats:
            parts = []
            if train_stats.get("final_loss") is not None:
                parts.append(f"- **Final training loss**: {train_stats['final_loss']:.4f}")
            if train_stats.get("total_steps"):
                parts.append(f"- **Total steps**: {train_stats['total_steps']:,}")
            if train_stats.get("train_runtime"):
                hours = train_stats["train_runtime"] / 3600
                parts.append(f"- **Training time**: {hours:.1f} hours")
            if parts:
                train_results = "## Training Results\n\n" + "\n".join(parts) + "\n"

        datasets_yaml = "\n  - ".join(s.name for s in config.data.sources)
        hub_user = config.hub.model_id.split("/")[0] if "/" in config.hub.model_id else "user"

        card = f"""---
license: apache-2.0
base_model: {config.model.name}
tags:
  - chain-of-thought
  - reasoning
  - qlora
  - fine-tuned
datasets:
  - {datasets_yaml}
pipeline_tag: text-generation
---

# {config.hub.model_id.split('/')[-1]}

Fine-tuned **{config.model.name}** for chain-of-thought reasoning using QLoRA.

## Training Details

- **Method**: QLoRA (4-bit NF4 quantization + LoRA r={config.lora.r}, alpha={config.lora.alpha})
- **Data**: {sample_counts}
- **Epochs**: {config.training.num_epochs}
- **Effective batch size**: {config.training.per_device_batch_size * config.training.gradient_accumulation_steps} ({config.training.per_device_batch_size} x {config.training.gradient_accumulation_steps} gradient accumulation)
- **Learning rate**: {config.training.learning_rate} ({config.training.lr_scheduler} schedule)
- **Max sequence length**: {config.training.max_seq_length}
- **Precision**: {"bfloat16" if config.training.bf16 else "float32"}
- **Hardware**: AMD ROCm / NVIDIA CUDA (trained on AMD Radeon AI PRO R9700)

{train_results}
{eval_table}
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{config.hub.model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [
    {{"role": "system", "content": "{config.data.system_message}"}},
    {{"role": "user", "content": "What is 15% of 240?"}},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Framework

Built with the [llm-from-scratch](https://github.com/vjsingh1984/llm-from-scratch) fine-tuning framework.
"""
        return card
