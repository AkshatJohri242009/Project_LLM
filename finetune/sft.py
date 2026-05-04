"""Supervised fine-tune the base checkpoint with ChatML masking and LoRA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from model import GPTConfig, GPTForCausalLM
from train.train import autocast_context, no_decay_param_groups
from train.utils import load_config, resolve_path, seed_everything


CHAT_TOKENS = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}


class ListDataset(Dataset):
    """Wrap a list of examples in a PyTorch dataset."""

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        """Store examples."""
        self.examples = examples

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one example."""
        return self.examples[idx]


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the latest step checkpoint if one exists."""
    checkpoints = sorted(
        [path for path in checkpoint_dir.glob("step_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def load_examples(name: str, fallback: str, split: str, max_samples: int | None) -> list[dict[str, Any]]:
    """Load SFT examples from Hugging Face with a fallback dataset."""
    from datasets import load_dataset

    for dataset_name in [name, fallback]:
        try:
            dataset = load_dataset(dataset_name, split=split)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            return [dict(item) for item in dataset]
        except Exception as exc:
            print(f"Skipping SFT dataset {dataset_name}: {exc}")
    return [
        {
            "instruction": "Explain gravity to a 5 year old.",
            "output": "Gravity is the invisible pull that keeps your feet on the ground and helps things fall down.",
        }
    ]


def messages_from_example(example: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize common instruction datasets into role/content messages."""
    if isinstance(example.get("messages"), list):
        return [
            {"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))}
            for msg in example["messages"]
            if msg.get("content")
        ]
    if "prompt" in example and "completion" in example:
        return [
            {"role": "user", "content": str(example["prompt"])},
            {"role": "assistant", "content": str(example["completion"])},
        ]
    instruction = str(example.get("instruction") or example.get("input") or example.get("question") or "")
    output = str(example.get("output") or example.get("response") or example.get("answer") or "")
    return [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]


def chatml_segments(messages: list[dict[str, str]]) -> list[tuple[str, bool]]:
    """Return ChatML text segments with assistant-label flags."""
    segments: list[tuple[str, bool]] = []
    if not any(msg["role"] == "system" for msg in messages):
        segments.append(("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", False))
    for msg in messages:
        role = msg["role"] if msg["role"] in {"system", "user", "assistant"} else "user"
        content = msg["content"].strip()
        if role == "assistant":
            segments.append(("<|im_start|>assistant\n", False))
            segments.append((f"{content}<|im_end|>\n", True))
        else:
            segments.append((f"<|im_start|>{role}\n{content}<|im_end|>\n", False))
    return segments


def encode_chatml(tokenizer: Any, example: dict[str, Any], max_seq_len: int) -> tuple[list[int], list[int]]:
    """Encode one example and mask non-assistant labels with -100."""
    input_ids: list[int] = []
    labels: list[int] = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(tokenizer.bos_token_id)
        labels.append(-100)
    for text, trainable in chatml_segments(messages_from_example(example)):
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(ids)
        labels.extend(ids if trainable else [-100] * len(ids))
    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    return input_ids[:max_seq_len], labels[:max_seq_len]


def make_collator(tokenizer: Any, max_seq_len: int) -> Any:
    """Create a padding collator for SFT examples."""

    def collate(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad encoded SFT examples into tensors."""
        encoded = [encode_chatml(tokenizer, example, max_seq_len) for example in examples]
        max_len = max(len(ids) for ids, _ in encoded)
        pad_id = tokenizer.pad_token_id or 0
        batch_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_mask: list[list[int]] = []
        for ids, labels in encoded:
            pad = max_len - len(ids)
            batch_ids.append(ids + [pad_id] * pad)
            batch_labels.append(labels + [-100] * pad)
            batch_mask.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        }

    return collate


def load_model_for_sft(cfg: dict[str, Any], base_model: Path | None, vocab_size: int, device: torch.device) -> GPTForCausalLM:
    """Load a base checkpoint or initialize a fresh model."""
    if base_model and (base_model / "config.json").exists():
        model = GPTForCausalLM.load_pretrained(base_model, map_location=device)
    else:
        model_cfg = dict(cfg["model"] if device.type == "cuda" else cfg["tiny_model"])
        model_cfg["vocab_size"] = vocab_size
        model = GPTForCausalLM(GPTConfig(**model_cfg))
    model.resize_token_embeddings(vocab_size)
    return model.to(device)


def train_sft(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Run LoRA SFT and save a merged checkpoint."""
    from peft import LoraConfig, get_peft_model
    from transformers import PreTrainedTokenizerFast

    sft_cfg = cfg["sft"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_dir = resolve_path(args.tokenizer_dir or cfg["paths"]["tokenizer_dir"])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.add_special_tokens(CHAT_TOKENS)
    checkpoint_dir = resolve_path(cfg["paths"]["checkpoint_dir"])
    base_model = resolve_path(args.base_model) if args.base_model else find_latest_checkpoint(checkpoint_dir)
    model = load_model_for_sft(cfg, base_model, len(tokenizer), device)
    peft_cfg = LoraConfig(
        r=int(sft_cfg["lora_rank"]),
        lora_alpha=int(sft_cfg["lora_alpha"]),
        target_modules=list(sft_cfg["target_modules"]),
        lora_dropout=float(sft_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    max_samples = args.max_samples or (64 if args.smoke else None)
    examples = load_examples(sft_cfg["dataset"], sft_cfg["fallback_dataset"], args.split, max_samples)
    batch_size = args.batch_size or (2 if device.type == "cpu" else int(sft_cfg["batch_size"]))
    loader = DataLoader(
        ListDataset(examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collator(tokenizer, int(args.max_seq_len or sft_cfg["max_seq_len"])),
    )
    optimizer = torch.optim.AdamW(no_decay_param_groups(model, 0.0), lr=float(args.lr or sft_cfg["lr"]))
    total_steps = max(1, math.ceil(len(loader) * int(args.epochs or sft_cfg["epochs"])))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()
    step = 0
    for _ in range(int(args.epochs or sft_cfg["epochs"])):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                outputs = model(**batch, return_dict=True)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 or step % 10 == 0:
                print({"step": step, "sft_loss": float(loss.item()), "lr": scheduler.get_last_lr()[0]})
            if args.smoke and step >= 2:
                break
        if args.smoke and step >= 2:
            break

    output_dir = resolve_path(args.output_dir or checkpoint_dir / "sft")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged SFT checkpoint to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse SFT CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the SFT command."""
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["project"]["seed"]))
    train_sft(cfg, args)


if __name__ == "__main__":
    main()
