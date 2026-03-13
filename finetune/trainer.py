"""CoT training orchestration using trl.SFTTrainer."""

import json
from pathlib import Path

from transformers import TrainingArguments

from finetune.config import FinetuneConfig
from finetune.quantization import QuantizationManager
from finetune.lora import LoRAManager
from finetune.data import DatasetLoader


class CoTTrainer:
    """Orchestrates QLoRA fine-tuning: config → quantize → LoRA → data → train."""

    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.device_info = None
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def setup(self):
        """Initialize model, tokenizer, data, and LoRA."""
        _, self.device_info = self.config.resolve_device()
        print(f"Device: {self.device_info.get('name', 'CPU')} "
              f"({self.device_info['backend']})")

        print(f"Loading model: {self.config.model.name}")
        bnb_config = QuantizationManager.get_bnb_config(self.config)
        self.model = QuantizationManager.load_model(self.config, bnb_config)
        self.tokenizer = QuantizationManager.load_tokenizer(self.config)

        print("Applying LoRA adapters...")
        self.model = LoRAManager.apply(self.model, self.config)
        LoRAManager.print_trainable_params(self.model)

        print("Loading datasets...")
        loader = DatasetLoader(self.config)
        self.dataset = loader.load()

    def train(self, max_steps: int = -1):
        """Run the training loop."""
        from trl import SFTTrainer, SFTConfig

        if self.model is None:
            self.setup()

        tcfg = self.config.training
        output_dir = Path(tcfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=tcfg.num_epochs,
            per_device_train_batch_size=tcfg.per_device_batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.learning_rate,
            lr_scheduler_type=tcfg.lr_scheduler,
            warmup_ratio=tcfg.warmup_ratio,
            max_seq_length=tcfg.max_seq_length,
            bf16=tcfg.bf16,
            logging_steps=tcfg.logging_steps,
            save_steps=tcfg.save_steps,
            save_total_limit=3,
            max_steps=max_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
        )

        print("Starting training...")
        result = trainer.train()
        print(f"Training complete. Loss: {result.training_loss:.4f}")

        self.save(output_dir)
        return result

    def save(self, output_dir: Path):
        """Save adapter, tokenizer, and config."""
        adapter_dir = output_dir / "adapter"
        LoRAManager.save_adapter(self.model, str(adapter_dir))
        self.tokenizer.save_pretrained(str(adapter_dir))

        config_path = output_dir / "finetune_config.json"
        import dataclasses
        config_dict = dataclasses.asdict(self.config)
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Saved adapter to {adapter_dir}")
