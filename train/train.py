"""Pretrain the scratch GPT model on packed uint16 token shards."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import yaml

from model import GPTConfig, GPTForCausalLM
from train.utils import (
    cleanup_distributed,
    gpu_vram_gb,
    is_main_process,
    load_config,
    resolve_path,
    seed_everything,
    setup_distributed,
)


class BinaryTokenDataset(Dataset):
    """Memory-map packed uint16 token shards for one split."""

    def __init__(self, data_dir: Path, split: str, seq_len: int) -> None:
        """Index split shards and expose fixed-length sequences."""
        self.seq_len = seq_len
        self.paths = sorted(data_dir.glob(f"{split}_*.bin"))
        if not self.paths:
            raise FileNotFoundError(f"No {split}_*.bin shards found in {data_dir}")
        self.arrays = [np.memmap(path, dtype=np.uint16, mode="r") for path in self.paths]
        self.sequence_counts = [len(array) // seq_len for array in self.arrays]
        self.cumulative = np.cumsum(self.sequence_counts).tolist()

    def __len__(self) -> int:
        """Return the number of fixed-length sequences."""
        return int(self.cumulative[-1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return one sequence as int64 token ids."""
        shard_idx = bisect_right(self.cumulative, idx)
        prior = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = idx - prior
        start = local_idx * self.seq_len
        values = np.asarray(self.arrays[shard_idx][start : start + self.seq_len], dtype=np.int64)
        return torch.from_numpy(values)


class SyntheticTokenDataset(Dataset):
    """Deterministic random token data for offline smoke tests."""

    def __init__(self, vocab_size: int, seq_len: int, length: int, seed: int) -> None:
        """Create a synthetic dataset descriptor."""
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        self.seed = seed

    def __len__(self) -> int:
        """Return the configured number of synthetic samples."""
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate one deterministic random sequence."""
        generator = torch.Generator().manual_seed(self.seed + idx)
        return torch.randint(0, self.vocab_size, (self.seq_len,), generator=generator, dtype=torch.long)


def collate_batch(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack input ids and labels for shifted causal language modeling."""
    input_ids = torch.stack(batch, dim=0).long()
    return input_ids, input_ids.clone()


def autocast_context(device: torch.device) -> torch.amp.autocast_mode.autocast:
    """Return a bf16 autocast context for CUDA or CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in {"cuda", "cpu"})


def no_decay_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Build AdamW parameter groups with no decay on bias and norm parameters."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_lambda(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    """Compute linear-warmup then cosine-decay LR multiplier."""
    if step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def estimate_mfu(model: nn.Module, tokens_per_sec: float, device: torch.device) -> float:
    """Estimate model flops utilization with a rough dense-transformer formula."""
    if device.type != "cuda":
        return 0.0
    raw_model = model.module if hasattr(model, "module") else model
    params = sum(p.numel() for p in raw_model.parameters())
    flops_per_token = 6 * params
    achieved = flops_per_token * tokens_per_sec
    peak = torch.cuda.get_device_properties(device).multi_processor_count * 312e12 / 108
    return float(max(0.0, min(1.0, achieved / peak)))


def tokenizer_vocab_size(cfg: dict[str, Any]) -> int | None:
    """Return tokenizer vocab size when a tokenizer has already been trained."""
    tokenizer_dir = resolve_path(cfg["paths"]["tokenizer_dir"])
    if not (tokenizer_dir / "tokenizer.json").exists():
        return None
    try:
        from transformers import PreTrainedTokenizerFast

        return len(PreTrainedTokenizerFast.from_pretrained(tokenizer_dir))
    except Exception:
        return None


def make_datasets(
    cfg: dict[str, Any], seq_len: int, synthetic: bool, vocab_size: int
) -> tuple[Dataset, Dataset, bool]:
    """Create train and validation datasets from shards or synthetic data."""
    data_dir = resolve_path(cfg["paths"]["processed_dir"])
    if not synthetic:
        try:
            return BinaryTokenDataset(data_dir, "train", seq_len), BinaryTokenDataset(data_dir, "val", seq_len), False
        except FileNotFoundError as exc:
            if is_main_process():
                print(f"{exc}; using synthetic smoke data.")
    seed = int(cfg["project"]["seed"])
    return (
        SyntheticTokenDataset(vocab_size, seq_len, 4096, seed),
        SyntheticTokenDataset(vocab_size, seq_len, 512, seed + 10_000),
        True,
    )


def make_loader(dataset: Dataset, batch_size: int, distributed: bool, shuffle: bool) -> DataLoader:
    """Create a DataLoader with an optional DistributedSampler."""
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        drop_last=True,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    cfg: dict[str, Any],
    step: int,
    checkpoint_dir: Path,
    keep_last: int,
) -> None:
    """Save model and optimizer state and remove old step checkpoints."""
    if not is_main_process():
        return
    raw_model = model.module if hasattr(model, "module") else model
    out_dir = checkpoint_dir / f"step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_model.save_pretrained(out_dir)
    tokenizer_dir = resolve_path(cfg["paths"]["tokenizer_dir"])
    if tokenizer_dir.exists():
        for tokenizer_file in tokenizer_dir.glob("*"):
            if tokenizer_file.is_file():
                shutil.copy2(tokenizer_file, out_dir / tokenizer_file.name)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        out_dir / "training_state.pt",
    )
    (out_dir / "training_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    checkpoints = sorted(
        [path for path in checkpoint_dir.glob("step_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    for old in checkpoints[:-keep_last]:
        shutil.rmtree(old)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_tokens: int,
    seq_len: int,
) -> float:
    """Compute validation loss over a token budget."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    max_batches = max(1, math.ceil(eval_tokens / max(1, loader.batch_size * seq_len)))
    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device):
            outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        total_loss += float(loss.item())
        total_batches += 1
        if total_batches >= max_batches:
            break
    model.train()
    return total_loss / max(1, total_batches)


def parse_args() -> argparse.Namespace:
    """Parse pretraining CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run pretraining."""
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["project"]["seed"]))
    distributed, rank, local_rank, _ = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_cfg = cfg["train"]
    model_cfg_data = dict(cfg["model"])
    low_resource = args.smoke or device.type == "cpu" or (device.type == "cuda" and gpu_vram_gb(device) < 24)
    if low_resource:
        model_cfg_data.update(cfg["tiny_model"])
        model_cfg_data["gradient_checkpointing"] = device.type == "cuda"
    vocab_size = tokenizer_vocab_size(cfg) or int(model_cfg_data["vocab_size"])
    model_cfg_data["vocab_size"] = vocab_size
    seq_len = int(args.seq_len or min(cfg["data"]["seq_len"], model_cfg_data["max_seq_len"]))
    model_cfg_data["max_seq_len"] = seq_len

    if low_resource:
        batch_size = args.batch_size or (train_cfg["cpu_batch_size"] if device.type == "cpu" else train_cfg["low_vram_batch_size"])
        grad_accum_steps = args.grad_accum_steps or (1 if device.type == "cpu" else train_cfg["low_vram_grad_accum_steps"])
        max_steps = args.max_steps or int(train_cfg["cpu_max_steps"])
    else:
        batch_size = args.batch_size or int(train_cfg["batch_size"])
        grad_accum_steps = args.grad_accum_steps or int(train_cfg["grad_accum_steps"])
        max_steps = args.max_steps or int(train_cfg["max_steps"])

    train_dataset, val_dataset, used_synthetic = make_datasets(cfg, seq_len, synthetic=args.synthetic, vocab_size=vocab_size)
    if args.require_data and used_synthetic:
        raise FileNotFoundError("Processed shards are required but were not found.")
    if used_synthetic:
        max_steps = min(max_steps, int(train_cfg["cpu_max_steps"]))
    train_loader = make_loader(train_dataset, batch_size, distributed, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size, distributed, shuffle=False)

    model = GPTForCausalLM(GPTConfig(**model_cfg_data)).to(device)
    if getattr(model.config, "gradient_checkpointing", False):
        model.gradient_checkpointing = True
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        no_decay_param_groups(model, float(train_cfg["weight_decay"])),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg["betas"]),
        eps=float(train_cfg["eps"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(
            step,
            int(train_cfg["warmup_steps"]),
            max_steps,
            float(train_cfg["min_lr_ratio"]),
        ),
    )
    checkpoint_dir = resolve_path(args.checkpoint_dir or cfg["paths"]["checkpoint_dir"])
    wandb_run = None
    if args.wandb and is_main_process():
        import wandb

        wandb_run = wandb.init(project=cfg["project"]["name"], config=cfg, mode=os.environ.get("WANDB_MODE", "online"))

    model.train()
    step = 0
    running_loss = 0.0
    loss_steps = 0
    tokens_since_log = 0
    last_log = time.time()
    train_iter = iter(train_loader)
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            try:
                input_ids, labels = next(train_iter)
            except StopIteration:
                if distributed and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(step)
                train_iter = iter(train_loader)
                input_ids, labels = next(train_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(device):
                outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                loss = loss / grad_accum_steps
            loss.backward()
            running_loss += float(loss.item())
            loss_steps += 1
            tokens_since_log += input_ids.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        optimizer.step()
        scheduler.step()
        step += 1

        should_log = step == 1 or step % 10 == 0
        should_eval = step == 1 or step % int(train_cfg["eval_interval"]) == 0 or step == max_steps
        if should_eval:
            val_tokens = min(
                int(train_cfg["eval_tokens"]),
                batch_size * seq_len * (2 if used_synthetic or args.smoke else 10_000),
            )
            val_loss = evaluate(model, val_loader, device, val_tokens, seq_len)
        else:
            val_loss = float("nan")
        if should_log and is_main_process():
            elapsed = max(1e-6, time.time() - last_log)
            tokens_per_sec = tokens_since_log / elapsed
            metrics = {
                "step": step,
                "train_loss": running_loss / max(1, loss_steps),
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "tokens_per_sec": tokens_per_sec,
                "mfu": estimate_mfu(model, tokens_per_sec, device),
            }
            print(metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            running_loss = 0.0
            loss_steps = 0
            tokens_since_log = 0
            last_log = time.time()

        if step % int(train_cfg["checkpoint_interval"]) == 0 or step == max_steps:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                cfg,
                step,
                checkpoint_dir,
                int(train_cfg["keep_last_checkpoints"]),
            )

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
