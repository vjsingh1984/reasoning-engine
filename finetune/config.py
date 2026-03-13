"""Configuration management for QLoRA fine-tuning."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from device import get_device, get_device_info


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_blocksize: Optional[int] = None  # auto-detected from device


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class DataSourceConfig:
    name: str = ""
    split: str = "train"
    max_samples: Optional[int] = None
    prompt_field: str = "query"
    response_field: str = "response"


@dataclass
class DataConfig:
    system_message: str = "You are a helpful assistant. Solve problems step by step."
    sources: list[DataSourceConfig] = field(default_factory=list)


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/finetune"
    num_epochs: int = 1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500


@dataclass
class EvalConfig:
    tasks: list[str] = field(default_factory=lambda: ["gsm8k", "arc_challenge"])
    num_fewshot: int = 5


@dataclass
class HubConfig:
    model_id: str = ""
    push: bool = False


@dataclass
class FinetuneConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    device: str = "auto"

    def resolve_device(self):
        """Resolve device string and set quantization blocksize."""
        device_str = None if self.device == "auto" else self.device
        torch_device = get_device(device_str)
        info = get_device_info(torch_device)

        if self.quantization.bnb_4bit_blocksize is None:
            if info.get("backend") == "ROCm":
                self.quantization.bnb_4bit_blocksize = 128
            else:
                self.quantization.bnb_4bit_blocksize = 64

        return torch_device, info


def _apply_overrides(d: dict, overrides: dict) -> dict:
    """Apply dot-separated overrides to a nested dict."""
    for key, value in overrides.items():
        parts = key.split(".")
        target = d
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return d


def load_config(yaml_path: str, overrides: Optional[dict] = None) -> FinetuneConfig:
    """Load a FinetuneConfig from a YAML file with optional overrides."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    if overrides:
        raw = _apply_overrides(raw, overrides)

    config = FinetuneConfig()

    if "model" in raw:
        config.model = ModelConfig(**raw["model"])

    if "quantization" in raw:
        config.quantization = QuantizationConfig(**raw["quantization"])

    if "lora" in raw:
        config.lora = LoRAConfig(**raw["lora"])

    if "data" in raw:
        data_raw = raw["data"]
        sources = [DataSourceConfig(**s) for s in data_raw.pop("sources", [])]
        config.data = DataConfig(**data_raw, sources=sources)

    if "training" in raw:
        config.training = TrainingConfig(**raw["training"])

    if "eval" in raw:
        config.eval = EvalConfig(**raw["eval"])

    if "hub" in raw:
        config.hub = HubConfig(**raw["hub"])

    if "device" in raw:
        config.device = raw["device"]

    return config
