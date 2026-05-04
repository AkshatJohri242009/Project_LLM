"""Shared helpers for configuration, paths, seeding, and distributed setup."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def repo_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    """Load a YAML config from a repo-relative or absolute path."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute path."""
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    """Return true on rank zero or in non-distributed runs."""
    return int(os.environ.get("RANK", "0")) == 0


def setup_distributed() -> tuple[bool, int, int, int]:
    """Initialize torch.distributed when launched with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    return distributed, rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Tear down torch.distributed if it was initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a JSON file with stable indentation."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def gpu_vram_gb(device: torch.device) -> float:
    """Return total GPU VRAM in GB, or 0 for non-CUDA devices."""
    if device.type != "cuda":
        return 0.0
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / (1024**3)
