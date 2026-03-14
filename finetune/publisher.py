"""Publish fine-tuned models to HuggingFace Hub."""

import json
from pathlib import Path

import torch
from finetune.config import FinetuneConfig
from finetune.lora import LoRAManager
from finetune.model_card import ModelCardGenerator


class HubPublisher:
    """Merge adapter, generate model card, and push to HuggingFace Hub."""

    def __init__(self, config: FinetuneConfig, adapter_path: str = None):
        self.config = config
        self.adapter_path = adapter_path or str(
            Path(config.training.output_dir) / "adapter"
        )

    def push(self, merge: bool = True):
        """Full publish pipeline: merge adapter → save → generate card → push."""
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer

        output_dir = Path(self.config.training.output_dir) / "merged"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load base model in float16 (not quantized) so merge produces valid weights
        model_dtype = getattr(torch, self.config.model.torch_dtype)
        print(f"Loading base model in {self.config.model.torch_dtype}: {self.config.model.name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            torch_dtype=model_dtype,
            device_map="auto",
            trust_remote_code=self.config.model.trust_remote_code,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading adapter: {self.adapter_path}")
        model = LoRAManager.load_adapter(model, self.adapter_path)

        if merge:
            print("Merging adapter into base model...")
            model = LoRAManager.merge(model)

        print(f"Saving merged model to {output_dir}")
        model.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))

        # Load eval results and training stats if available
        eval_path = Path(self.config.training.output_dir) / "eval_results.json"
        eval_results = None
        if eval_path.exists():
            with open(eval_path) as f:
                eval_results = json.load(f)

        train_stats = self._load_train_stats()

        card_content = ModelCardGenerator.generate(
            self.config, eval_results=eval_results, train_stats=train_stats
        )
        card_path = output_dir / "README.md"
        card_path.write_text(card_content)
        print("Generated model card")

        model_id = self.config.hub.model_id
        print(f"Pushing to HuggingFace Hub: {model_id}")
        api = HfApi()
        api.create_repo(model_id, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=model_id,
            commit_message=f"Upload fine-tuned {self.config.model.name} with QLoRA",
        )
        print(f"Published: https://huggingface.co/{model_id}")

    def _load_train_stats(self) -> dict:
        """Load training stats from the trainer_state.json of the last checkpoint."""
        output_dir = Path(self.config.training.output_dir)

        # Check checkpoints for trainer_state.json
        checkpoints = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
        if not checkpoints:
            return {}

        with open(checkpoints[-1]) as f:
            state = json.load(f)

        stats = {"total_steps": state.get("global_step")}
        log_history = state.get("log_history", [])

        # Look for the summary entry (has train_loss/train_runtime)
        for entry in reversed(log_history):
            if "train_loss" in entry:
                stats["final_loss"] = entry["train_loss"]
                stats["train_runtime"] = entry.get("train_runtime")
                stats["epoch"] = entry.get("epoch")
                return stats

        # No summary entry — compute average loss from last N log entries
        if log_history:
            recent = [e["loss"] for e in log_history[-10:] if "loss" in e]
            if recent:
                stats["final_loss"] = sum(recent) / len(recent)
            last = log_history[-1]
            stats["epoch"] = last.get("epoch")

        return stats
