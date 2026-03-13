"""LoRA adapter management for QLoRA fine-tuning."""

from pathlib import Path

from finetune.config import FinetuneConfig


class LoRAManager:
    """Manages LoRA adapter lifecycle: apply, save, load, merge."""

    @staticmethod
    def apply(model, config: FinetuneConfig):
        """Apply LoRA adapters to a model."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        return model

    @staticmethod
    def save_adapter(model, path: str):
        """Save LoRA adapter weights."""
        Path(path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)

    @staticmethod
    def load_adapter(model, path: str):
        """Load LoRA adapter weights onto a base model."""
        from peft import PeftModel

        return PeftModel.from_pretrained(model, path)

    @staticmethod
    def merge(model):
        """Merge LoRA adapters into the base model and unload."""
        return model.merge_and_unload()

    @staticmethod
    def print_trainable_params(model):
        """Print the number of trainable parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = 100 * trainable / total
        print(f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
