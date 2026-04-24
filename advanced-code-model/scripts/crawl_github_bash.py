#!/usr/bin/env python3
"""
Crawl bash scripts from popular GitHub repositories.

Targets:
  - dotfiles repositories (lots of bash scripts)
  - .sh files from popular devtools
  - Installation scripts
  - CI/CD scripts (.github/workflows, .gitlab-ci.yml)
  - Build scripts (Makefile, build.sh)

Usage:
    python scripts/crawl_github_bash.py
"""

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urljoin
from tqdm import tqdm

import numpy as np
from tokenizers import Tokenizer


# Popular bash-heavy repositories
BASH_REPOS = [
    # Dotfiles (rich in bash scripts)
    "https://github.com/holman/dotfiles",
    "https://github.com/mathiasbynens/dotfiles",
    "https://github.com/jessfraz/dotfiles",
    "https://github.com/cowboy/dotfiles",

    # Dev tools with scripts
    "https://github.com/nvm-sh/nvm",
    "https://github.com/pyenv/pyenv",
    "https://github.com/rbenv/rbenv",
    "https://github.com/asdf-vm/asdf",

    # CLI tools
    "https://github.com/Bash-it/bash-it",
    "https://github.com/ohmyzsh/ohmyzsh",
    "https://github.com/scop/bash-completion",

    # Installation scripts
    "https://github.com/nodesource/distributions",
    "https://github.com/yarnpkg/yarn",
    "https://github.com/docker/docker-install",
]


def clone_repo(repo_url, temp_dir):
    """Clone a git repository to temp directory."""
    repo_name = repo_url.rstrip('/').split('/')[-1]
    clone_path = temp_dir / repo_name

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_path)],
            check=True,
            capture_output=True,
            timeout=300  # 5 min timeout
        )
        return clone_path
    except Exception as e:
        print(f"  Failed to clone {repo_url}: {e}")
        return None


def find_bash_files(repo_path):
    """Find all bash/shell script files in repository."""
    bash_files = []
    extensions = ['.sh', '.bash', '.zsh', '.fish']
    special_names = ['.bashrc', '.bash_profile', '.zshrc', '.profile', '.zprofile', '.zlogin', '.zlogout']

    for ext in extensions:
        bash_files.extend(repo_path.rglob(f"*{ext}"))

    for name in special_names:
        bash_files.extend(repo_path.rglob(name))

    # Also check for executable files with shebang
    for file_path in repo_path.rglob("*"):
        if file_path.is_file() and file_path.stat().st_mode & 0o111:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline(100)
                    if first_line.startswith('#!') and ('bash' in first_line or 'sh' in first_line):
                        bash_files.append(file_path)
            except:
                pass

    return list(set(bash_files))  # Remove duplicates


def extract_bash_scripts(repo_path, max_scripts=1000):
    """Extract bash script content from repository."""
    bash_files = find_bash_files(repo_path)
    scripts = []

    for script_path in tqdm(bash_files[:max_scripts], desc=f"Extracting from {repo_path.name}", leave=False):
        try:
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Filter out very short or very long scripts
            if len(content) < 50 or len(content) > 50000:
                continue

            # Remove comments and empty lines for cleaner training data
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                # Skip comment-only lines
                stripped = line.strip()
                if stripped.startswith('#') and len(stripped) < 50:
                    continue
                cleaned_lines.append(line)

            cleaned_content = '\n'.join(cleaned_lines)

            if len(cleaned_content) < 50:
                continue

            scripts.append(cleaned_content)

        except Exception as e:
            continue

    return scripts


def crawl_and_tokenize(tokenizer, max_seq_len, pad_id, eos_id, repos=None, output_dir=None):
    """Crawl bash scripts from GitHub and tokenize."""
    print("\n" + "=" * 60)
    print("Crawling Bash Scripts from GitHub")
    print("=" * 60)

    if repos is None:
        repos = BASH_REPOS

    all_scripts = []
    total_scripts = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for repo_url in tqdm(repos, desc="Cloning repositories"):
            print(f"\n  Processing: {repo_url}")

            repo_path = clone_repo(repo_url, temp_path)
            if repo_path is None:
                continue

            scripts = extract_bash_scripts(repo_path, max_scripts=500)
            print(f"    Found {len(scripts)} bash scripts")
            all_scripts.extend(scripts)
            total_scripts += len(scripts)

            # Clean up cloned repo to save space
            subprocess.run(["rm", "-rf", str(repo_path)], capture_output=True)

    print(f"\n  Total scripts collected: {total_scripts:,}")

    # Tokenize
    print("\n  Tokenizing scripts...")
    all_tokens = []
    for script in tqdm(all_scripts, desc="Tokenizing"):
        tokens = tokenizer.encode(script).ids
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)

    num_seqs = len(all_tokens) // max_seq_len
    if num_seqs == 0:
        print("  WARNING: Not enough tokens")
        return None

    sequences = np.array(all_tokens[:num_seqs * max_seq_len], dtype=np.int32).reshape(num_seqs, max_seq_len)

    print(f"\n  Tokens: {len(all_tokens):,} -> {num_seqs:,} sequences")

    if output_dir:
        split = int(num_seqs * 0.95)
        np.save(output_dir / "github_crawled_bash_train.npy", sequences[:split])
        np.save(output_dir / "github_crawled_bash_val.npy", sequences[split:])
        print(f"  Saved: {split:,} train, {num_seqs - split:,} val")

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Crawl bash scripts from GitHub")
    parser.add_argument("--max-repos", type=int, default=10,
                        help="Maximum number of repos to crawl (default: 10)")
    parser.add_argument("--skip-crawl", action="store_true",
                        help="Skip crawling, only download dataset")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer_path = base_dir / "data" / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found: {tokenizer_path}")
        return
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Loaded tokenizer: vocab={tokenizer.get_vocab_size()}")

    vocab = tokenizer.get_vocab()
    pad_id = vocab.get("<PAD>", 0)
    eos_id = vocab.get("<EOS>", 3)
    max_seq_len = 1024

    total_new_tokens = 0

    # Crawl GitHub repos
    if not args.skip_crawl:
        repos_to_crawl = BASH_REPOS[:args.max_repos]
        seqs = crawl_and_tokenize(
            tokenizer, max_seq_len, pad_id, eos_id, repos_to_crawl, output_dir)
        if seqs is not None:
            total_new_tokens += seqs.shape[0] * max_seq_len

    print("\n" + "=" * 60)
    print("Crawl Complete!")
    print("=" * 60)
    print(f"Total new tokens: {total_new_tokens/1e9:.2f}B")


if __name__ == "__main__":
    main()
