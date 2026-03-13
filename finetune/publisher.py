"""Publish fine-tuned models to HuggingFace Hub."""

import json
from pathlib import Path

from finetune.config import FinetuneConfig
from finetune.quantization import QuantizationManager
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

        output_dir = Path(self.config.training.output_dir) / "merged"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading base model: {self.config.model.name}")
        bnb_config = QuantizationManager.get_bnb_config(self.config)
        model = QuantizationManager.load_model(self.config, bnb_config)
        tokenizer = QuantizationManager.load_tokenizer(self.config)

        print(f"Loading adapter: {self.adapter_path}")
        model = LoRAManager.load_adapter(model, self.adapter_path)

        if merge:
            print("Merging adapter into base model...")
            model = LoRAManager.merge(model)

        print(f"Saving merged model to {output_dir}")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Load eval results if available
        eval_path = Path(self.config.training.output_dir) / "eval_results.json"
        eval_results = None
        if eval_path.exists():
            with open(eval_path) as f:
                eval_results = json.load(f)

        card_content = ModelCardGenerator.generate(self.config, eval_results)
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
