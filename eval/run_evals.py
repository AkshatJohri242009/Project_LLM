"""Run lm-eval benchmarks and local validation diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from model import GPTForCausalLM
from train.train import BinaryTokenDataset, autocast_context, collate_batch
from train.utils import load_config, resolve_path, seed_everything, write_json


def run_lm_eval(model_path: Path, tasks: list[str], output_path: Path) -> dict[str, Any]:
    """Run the lm-eval harness through its module CLI."""
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path},trust_remote_code=True",
        "--tasks",
        ",".join(tasks),
        "--output_path",
        str(output_path),
        "--batch_size",
        "auto",
    ]
    proc = subprocess.run(cmd, cwd=resolve_path("."), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return {"error": proc.stderr[-4000:] or proc.stdout[-4000:], "returncode": proc.returncode}
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    return {"stdout": proc.stdout[-4000:]}


@torch.no_grad()
def validation_perplexity(model_path: Path, data_dir: Path, seq_len: int, token_budget: int) -> float | None:
    """Compute validation perplexity on local packed shards."""
    try:
        dataset = BinaryTokenDataset(data_dir, "val", seq_len)
    except FileNotFoundError:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTForCausalLM.load_pretrained(model_path, map_location=device).to(device).eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
    total_loss = 0.0
    total_batches = 0
    max_batches = max(1, math.ceil(token_budget / (4 * seq_len)))
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with autocast_context(device):
            outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
        total_loss += float(outputs.loss.item())
        total_batches += 1
        if total_batches >= max_batches:
            break
    return math.exp(total_loss / max(1, total_batches))


@torch.no_grad()
def throughput_probe(model_path: Path, seq_len: int) -> dict[str, Any]:
    """Measure a compact tokens/sec and peak-memory probe."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTForCausalLM.load_pretrained(model_path, map_location=device).to(device).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, min(seq_len, model.config.max_seq_len)), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        _ = model(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.time() - start)
    tokens_per_sec = input_ids.numel() * 5 / elapsed
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {"tokens_per_sec": tokens_per_sec, "peak_gpu_memory_bytes": peak_memory}


def macro_average(results: dict[str, Any], tasks: list[str]) -> float | None:
    """Extract task accuracies and compute a macro average when possible."""
    result_block = results.get("results") if isinstance(results, dict) else None
    if not isinstance(result_block, dict):
        return None
    scores = []
    for task in tasks:
        metrics = result_block.get(task, {})
        for key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
            if key in metrics:
                scores.append(float(metrics[key]))
                break
    return sum(scores) / len(scores) if scores else None


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="checkpoints/dpo")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-lm-eval", action="store_true")
    parser.add_argument("--val-token-budget", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    """Run requested evaluations and save a JSON report."""
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["project"]["seed"]))
    tasks = list(cfg["eval"]["tasks"])
    model_path = resolve_path(args.model)
    results_dir = resolve_path(args.output_dir or cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    lm_eval_json = results_dir / f"lm_eval_{timestamp}.json"
    lm_results = {"skipped": True} if args.skip_lm_eval else run_lm_eval(model_path, tasks, lm_eval_json)
    report = {
        "model": str(model_path),
        "tasks": tasks,
        "lm_eval": lm_results,
        "macro_average": macro_average(lm_results, tasks),
        "val_perplexity": validation_perplexity(
            model_path,
            resolve_path(cfg["paths"]["processed_dir"]),
            int(cfg["data"]["seq_len"]),
            int(args.val_token_budget),
        ),
        "diagnostics": throughput_probe(model_path, int(cfg["data"]["seq_len"])),
    }
    out_path = results_dir / f"results_{timestamp}.json"
    write_json(out_path, report)
    print(f"Saved evaluation report to {out_path}")


if __name__ == "__main__":
    main()
