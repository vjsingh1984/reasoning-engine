"""Evaluation using lm-evaluation-harness."""

import json
from pathlib import Path

from finetune.config import FinetuneConfig


class Evaluator:
    """Runs lm-eval benchmarks and compares base vs fine-tuned models."""

    def __init__(self, config: FinetuneConfig):
        self.config = config

    def run(self, model_path: str = None, tasks: list[str] = None) -> dict:
        """Run evaluation on configured tasks."""
        import lm_eval

        tasks = tasks or self.config.eval.tasks
        model_args = f"pretrained={self.config.model.name}"

        if model_path:
            model_args += f",peft={model_path}"

        model_args += f",trust_remote_code={self.config.model.trust_remote_code}"

        print(f"Evaluating on: {', '.join(tasks)}")
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            num_fewshot=self.config.eval.num_fewshot,
            batch_size="auto",
        )

        self._print_results(results)
        return results

    def compare(self, base_results: dict, finetuned_results: dict) -> str:
        """Generate a comparison table of base vs fine-tuned results."""
        lines = [
            f"{'Task':<20} {'Metric':<15} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}",
            "-" * 67,
        ]

        for task in base_results.get("results", {}):
            base_task = base_results["results"].get(task, {})
            ft_task = finetuned_results.get("results", {}).get(task, {})

            for metric, base_val in base_task.items():
                if metric.endswith(",none") or not isinstance(base_val, (int, float)):
                    continue
                ft_val = ft_task.get(metric, 0)
                if isinstance(ft_val, (int, float)):
                    delta = ft_val - base_val
                    sign = "+" if delta > 0 else ""
                    lines.append(
                        f"{task:<20} {metric:<15} {base_val:>10.4f} "
                        f"{ft_val:>12.4f} {sign}{delta:>9.4f}"
                    )

        table = "\n".join(lines)
        print(table)
        return table

    def save_results(self, results: dict, path: str):
        """Save evaluation results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {path}")

    @staticmethod
    def _print_results(results: dict):
        """Print evaluation results as a formatted table."""
        print(f"\n{'Task':<20} {'Metric':<15} {'Value':>10}")
        print("-" * 45)
        for task, metrics in results.get("results", {}).items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{task:<20} {metric:<15} {value:>10.4f}")
        print()
