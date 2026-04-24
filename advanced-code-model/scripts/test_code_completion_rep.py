#!/usr/bin/env python3
"""Test code completion with repetition penalty + temperature sweep.

Uses the same test prompts as test_code_completion.py but samples with
repetition penalty to see if degenerate loops are a sampling artifact
vs. a learned-representation issue.
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent))
from model.config import ModelConfig
from model.transformer import create_model
from device import get_device as _get_device


TESTS = [
    ("Loop Completion",      "for file in *.log; do",      ["done", "mv", "rm", "echo"]),
    ("Function Definition",  "function backup_files() {",  ["}", "tar", "cp", "echo"]),
    ("Conditional Statement", 'if [ -f "config.txt" ]; then', ["fi", "source", "cat", "echo"]),
    ("Variable Assignment",  "LOGFILE=",                    ['"', ".log", "/var/", "/tmp/"]),
    ("Command Pipeline",     "cat file.txt |",              ["grep", "awk", "sed", "sort"]),
    ("Error Handling",       "command ||",                  ["echo", "exit", "rm", "false"]),
    ("Shebang Start",        "#!/bin/bash\n",               ["echo", "#", "if", "for", "function"]),
    ("Comment Line",         "# Backup all log files\n",    ["cp", "tar", "mv", "for", "while"]),
]


def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    c = ckpt.get("config", {})
    cfg = ModelConfig(
        vocab_size=c.get("vocab_size", 32000),
        d_model=c.get("d_model", 1024),
        n_layers=c.get("n_layers", 26),
        n_heads=c.get("n_heads", 16),
        d_ff=c.get("d_ff", 4096),
        max_seq_len=c.get("max_seq_len", 1024),
    )
    model = create_model(cfg, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def generate_with_penalty(model, prompt_ids, max_new, temperature, top_k, top_p,
                           rep_penalty, device):
    """Standard top-k/top-p sampling + HF-style repetition penalty."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = ids.clone()
    max_seq_len = model.config.max_seq_len

    with torch.no_grad():
        for _ in range(max_new):
            ctx = generated if generated.shape[1] <= max_seq_len else generated[:, -max_seq_len:]
            logits = model(ctx)[:, -1, :]

            if rep_penalty != 1.0:
                for tok in set(generated[0].tolist()):
                    v = logits[0, tok]
                    logits[0, tok] = v / rep_penalty if v > 0 else v * rep_penalty

            logits = logits / temperature

            if top_k > 0:
                vals, _ = torch.topk(logits, top_k, dim=-1)
                logits = torch.where(logits < vals[:, -1:], torch.full_like(logits, float("-inf")), logits)

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            generated = torch.cat([generated, nxt], dim=1)

    return generated[0, len(prompt_ids):].tolist()


def run_suite(model, tok, device, temperature, rep_penalty, top_k=50, top_p=0.9):
    results = []
    for name, prompt, keywords in TESTS:
        ids = tok.encode(prompt).ids
        out_ids = generate_with_penalty(model, ids, 64, temperature, top_k, top_p,
                                         rep_penalty, device)
        completion = tok.decode(out_ids)
        found = [k for k in keywords if k in completion]
        results.append((name, prompt, completion, len(found) / len(keywords), found))
    return results


def print_run(label, results):
    print("\n" + "=" * 70)
    print(f"RUN: {label}")
    print("=" * 70)
    for name, prompt, completion, score, found in results:
        print(f"\n--- {name} ---")
        print(f"Prompt: {prompt.rstrip()}")
        print(f"Completion: {completion[:200]}")
        print(f"Match: {score:.0%} {found}")
    avg = sum(r[3] for r in results) / len(results)
    print(f"\n>>> {label}: average match = {avg:.1%}")
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/code_model_best.pth")
    ap.add_argument("--device", default="rocm")
    args = ap.parse_args()

    device = _get_device(args.device)
    tok = Tokenizer.from_file("data/tokenizer/tokenizer.json")
    model, _ = load_model(args.checkpoint, device)
    print(f"Loaded: {args.checkpoint} on {device}")

    torch.manual_seed(42)
    configs = [
        ("baseline (T=0.5, rep=1.0)", 0.5, 1.0),
        ("rep=1.2 (T=0.5)",           0.5, 1.2),
        ("rep=1.5 (T=0.5)",           0.5, 1.5),
        ("T=0.8 rep=1.2",             0.8, 1.2),
        ("T=1.0 rep=1.3",             1.0, 1.3),
    ]

    summary = []
    for label, temp, rep in configs:
        torch.manual_seed(42)
        res = run_suite(model, tok, device, temperature=temp, rep_penalty=rep)
        avg = print_run(label, res)
        summary.append((label, avg))

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    for label, avg in summary:
        print(f"  {avg:>6.1%}  {label}")


if __name__ == "__main__":
    main()
