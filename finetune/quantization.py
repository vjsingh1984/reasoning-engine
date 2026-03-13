"""Quantization management for QLoRA fine-tuning."""

import torch
from finetune.config import FinetuneConfig


class QuantizationManager:
    """Manages BitsAndBytes quantization configuration and model loading."""

    @staticmethod
    def get_bnb_config(config: FinetuneConfig):
        """Create a BitsAndBytesConfig from the fine-tune config."""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers is required for quantization. "
                "Install with: pip install -r requirements-finetune.txt"
            )

        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for 4-bit quantization. "
                "For ROCm: pip install bitsandbytes>=0.45.0 "
                "For CUDA: pip install bitsandbytes>=0.45.0"
            )

        qcfg = config.quantization
        compute_dtype = getattr(torch, qcfg.bnb_4bit_compute_dtype)

        kwargs = {
            "load_in_4bit": qcfg.load_in_4bit,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_quant_type": qcfg.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": qcfg.bnb_4bit_use_double_quant,
        }

        if qcfg.bnb_4bit_blocksize is not None:
            kwargs["bnb_4bit_blocksize"] = qcfg.bnb_4bit_blocksize

        return BitsAndBytesConfig(**kwargs)

    @staticmethod
    def load_model(config: FinetuneConfig, bnb_config, device_map="auto"):
        """Load a model with quantization applied."""
        from transformers import AutoModelForCausalLM

        model_dtype = getattr(torch, config.model.torch_dtype)

        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            quantization_config=bnb_config,
            dtype=model_dtype,
            device_map=device_map,
            trust_remote_code=config.model.trust_remote_code,
        )
        model.config.use_cache = False
        return model

    @staticmethod
    def load_tokenizer(config: FinetuneConfig):
        """Load the tokenizer for the configured model."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
