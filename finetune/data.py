"""Dataset loading and formatting for ChatML-style fine-tuning."""

from typing import Callable

from datasets import load_dataset, concatenate_datasets

from finetune.config import DataSourceConfig, FinetuneConfig


def format_generic(sample: dict, source: DataSourceConfig, system_message: str) -> dict:
    """Format a sample into ChatML messages using configured field names."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": str(sample[source.prompt_field])})
    messages.append({"role": "assistant", "content": str(sample[source.response_field])})
    return {"messages": messages}


def format_openorca(sample: dict, source: DataSourceConfig, system_message: str) -> dict:
    """Format OpenOrca samples which have a system_prompt field."""
    messages = []
    sys_msg = sample.get("system_prompt", system_message)
    if sys_msg:
        messages.append({"role": "system", "content": str(sys_msg)})
    messages.append({"role": "user", "content": str(sample[source.prompt_field])})
    messages.append({"role": "assistant", "content": str(sample[source.response_field])})
    return {"messages": messages}


# Registry of dataset-specific formatters
FORMATTERS: dict[str, Callable] = {
    "Open-Orca/OpenOrca": format_openorca,
}


class DatasetLoader:
    """Loads, formats, and merges datasets for fine-tuning."""

    def __init__(self, config: FinetuneConfig):
        self.config = config

    def load(self):
        """Load and merge all configured data sources into a single dataset."""
        all_datasets = []

        for source in self.config.data.sources:
            print(f"Loading {source.name} (split={source.split})...")
            ds = load_dataset(source.name, split=source.split)

            if source.max_samples and source.max_samples < len(ds):
                ds = ds.shuffle(seed=42).select(range(source.max_samples))

            formatter = FORMATTERS.get(source.name, format_generic)
            system_msg = self.config.data.system_message

            ds = ds.map(
                lambda sample: formatter(sample, source, system_msg),
                remove_columns=ds.column_names,
                desc=f"Formatting {source.name}",
            )
            all_datasets.append(ds)
            print(f"  {source.name}: {len(ds)} samples")

        if not all_datasets:
            raise ValueError("No data sources configured")

        if len(all_datasets) == 1:
            merged = all_datasets[0]
        else:
            merged = concatenate_datasets(all_datasets)

        merged = merged.shuffle(seed=42)
        print(f"Total dataset: {len(merged)} samples")
        return merged
