"""Model card generation from training config and eval results."""

from finetune.config import FinetuneConfig


class ModelCardGenerator:
    """Generates a HuggingFace model card from config and evaluation metrics."""

    @staticmethod
    def generate(config: FinetuneConfig, eval_results: dict = None) -> str:
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
                )

        card = f"""---
license: apache-2.0
base_model: {config.model.name}
tags:
  - chain-of-thought
  - reasoning
  - qlora
  - fine-tuned
datasets:
  - {sources.replace(', ', chr(10) + '  - ')}
pipeline_tag: text-generation
---

# {config.hub.model_id.split('/')[-1]}

Fine-tuned **{config.model.name}** for chain-of-thought reasoning using QLoRA.

## Training Details

- **Method**: QLoRA (4-bit NF4 quantization + LoRA r={config.lora.r}, alpha={config.lora.alpha})
- **Data**: {sample_counts}
- **Epochs**: {config.training.num_epochs}
- **Batch size**: {config.training.per_device_batch_size} x {config.training.gradient_accumulation_steps} gradient accumulation
- **Learning rate**: {config.training.learning_rate} ({config.training.lr_scheduler} schedule)
- **Max sequence length**: {config.training.max_seq_length}
- **Precision**: {"bfloat16" if config.training.bf16 else "float32"}

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

Built with the [llm-from-scratch](https://github.com/vsingh/llm-from-scratch) fine-tuning framework.
"""
        return card
