"""Inference engine for fine-tuned models."""

import torch
from pathlib import Path

from finetune.config import FinetuneConfig
from finetune.quantization import QuantizationManager
from finetune.lora import LoRAManager


class InferenceEngine:
    """Load a fine-tuned model and generate text."""

    def __init__(self, config: FinetuneConfig, adapter_path: str = None):
        self.config = config
        self.adapter_path = adapter_path or str(
            Path(config.training.output_dir) / "adapter"
        )
        self.model = None
        self.tokenizer = None

    def setup(self):
        """Load base model with adapter."""
        self.config.resolve_device()

        print(f"Loading model: {self.config.model.name}")
        bnb_config = QuantizationManager.get_bnb_config(self.config)
        self.model = QuantizationManager.load_model(self.config, bnb_config)
        self.tokenizer = QuantizationManager.load_tokenizer(self.config)

        if Path(self.adapter_path).exists():
            print(f"Loading adapter: {self.adapter_path}")
            self.model = LoRAManager.load_adapter(self.model, self.adapter_path)
        else:
            print(f"No adapter found at {self.adapter_path}, using base model")

        self.model.eval()

        # Build terminator token IDs for chat models (e.g. <|im_end|>, <|endoftext|>)
        self._terminators = [self.tokenizer.eos_token_id]
        for name in ["<|im_end|>", "<|eot_id|>", "<|end|>"]:
            tok_id = self.tokenizer.convert_tokens_to_ids(name)
            if isinstance(tok_id, int) and tok_id != self.tokenizer.unk_token_id:
                self._terminators.append(tok_id)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate a response for a single prompt."""
        if self.model is None:
            self.setup()

        messages = [
            {"role": "system", "content": self.config.data.system_message},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._terminators,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def stream(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7):
        """Stream tokens one at a time using TextIteratorStreamer."""
        from threading import Thread
        from transformers import TextIteratorStreamer

        if self.model is None:
            self.setup()

        messages = [
            {"role": "system", "content": self.config.data.system_message},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": 0.9,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self._terminators,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()

    def interactive(self):
        """Run an interactive chat loop."""
        if self.model is None:
            self.setup()

        print("\nInteractive mode (type 'quit' to exit)")
        print("-" * 40)

        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if prompt.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            if not prompt:
                continue

            print("\nAssistant: ", end="", flush=True)
            for token in self.stream(prompt):
                print(token, end="", flush=True)
            print()
