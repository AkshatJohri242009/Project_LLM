"""Align the SFT checkpoint with DPO on chosen/rejected preference pairs."""

from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from finetune.sft import CHAT_TOKENS
from model import GPTForCausalLM
from train.train import autocast_context, no_decay_param_groups
from train.utils import load_config, resolve_path, seed_everything


class PairDataset(Dataset):
    """Wrap DPO prompt/chosen/rejected examples."""

    def __init__(self, examples: list[dict[str, str]]) -> None:
        """Store preference examples."""
        self.examples = examples

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Return one preference example."""
        return self.examples[idx]


def load_pairs(name: str, split: str, max_samples: int | None) -> list[dict[str, Any]]:
    """Load preference pairs from Hugging Face or return one smoke pair."""
    from datasets import load_dataset

    try:
        dataset = load_dataset(name, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return [dict(item) for item in dataset]
    except Exception as exc:
        print(f"Skipping DPO dataset {name}: {exc}")
        return [
            {
                "prompt": "Explain gravity to a 5 year old.",
                "chosen": "Gravity is a pull that keeps us on the ground and makes dropped toys fall.",
                "rejected": "Gravity is a kind of weather that only happens at night.",
            }
        ]


def content_from_messages(value: Any) -> str:
    """Convert message lists or strings into assistant text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for msg in value:
            if isinstance(msg, dict) and msg.get("content"):
                parts.append(str(msg["content"]))
        return "\n".join(parts)
    return str(value)


def chatml_prompt(prompt: Any) -> str:
    """Format a DPO prompt as ChatML up to the assistant turn."""
    if isinstance(prompt, list):
        chunks = []
        if not any(isinstance(msg, dict) and msg.get("role") == "system" for msg in prompt):
            chunks.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            if role == "assistant":
                continue
            chunks.append(f"<|im_start|>{role}\n{msg.get('content', '')}<|im_end|>\n")
        chunks.append("<|im_start|>assistant\n")
        return "".join(chunks)
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def normalize_pair(example: dict[str, Any]) -> dict[str, str]:
    """Normalize one raw preference example for TRL DPOTrainer."""
    prompt = example.get("prompt") or example.get("messages") or example.get("input") or ""
    chosen = example.get("chosen") or example.get("chosen_response") or example.get("accept") or ""
    rejected = example.get("rejected") or example.get("rejected_response") or example.get("reject") or ""
    return {
        "prompt": chatml_prompt(prompt),
        "chosen": f"{content_from_messages(chosen)}<|im_end|>",
        "rejected": f"{content_from_messages(rejected)}<|im_end|>",
    }


def build_trl_trainer(model: Any, tokenizer: Any, train_dataset: Any, cfg: dict[str, Any], args: argparse.Namespace) -> Any:
    """Construct a DPOTrainer across TRL API versions."""
    from transformers import TrainingArguments
    from trl import DPOTrainer

    dpo_cfg = cfg["dpo"]
    output_dir = str(resolve_path(args.output_dir or Path(cfg["paths"]["checkpoint_dir"]) / "dpo"))
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(args.epochs or dpo_cfg["epochs"]),
        per_device_train_batch_size=int(args.batch_size or dpo_cfg["batch_size"]),
        learning_rate=float(args.lr or dpo_cfg["lr"]),
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    signature = inspect.signature(DPOTrainer.__init__)
    kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    if "beta" in signature.parameters:
        kwargs["beta"] = float(args.beta or dpo_cfg["beta"])
    if "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    if "max_length" in signature.parameters:
        kwargs["max_length"] = int(args.max_seq_len or dpo_cfg["max_seq_len"])
    if "max_prompt_length" in signature.parameters:
        kwargs["max_prompt_length"] = int((args.max_seq_len or dpo_cfg["max_seq_len"]) // 2)
    return DPOTrainer(**kwargs)


def encode_pair(tokenizer: Any, pair: dict[str, str], max_seq_len: int) -> dict[str, list[int]]:
    """Tokenize one DPO pair with prompt labels masked."""
    prompt_ids = tokenizer.encode(pair["prompt"], add_special_tokens=False)
    chosen_ids = tokenizer.encode(pair["chosen"], add_special_tokens=False)
    rejected_ids = tokenizer.encode(pair["rejected"], add_special_tokens=False)
    return {
        "chosen_input_ids": (prompt_ids + chosen_ids)[:max_seq_len],
        "chosen_labels": ([-100] * len(prompt_ids) + chosen_ids)[:max_seq_len],
        "rejected_input_ids": (prompt_ids + rejected_ids)[:max_seq_len],
        "rejected_labels": ([-100] * len(prompt_ids) + rejected_ids)[:max_seq_len],
    }


def make_manual_collator(tokenizer: Any, max_seq_len: int) -> Any:
    """Create a collator for the manual DPO fallback."""

    def collate(examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        """Pad chosen and rejected sequences."""
        encoded = [encode_pair(tokenizer, example, max_seq_len) for example in examples]
        pad_id = tokenizer.pad_token_id or 0
        out: dict[str, list[list[int]]] = {key: [] for key in encoded[0]}
        for key in out:
            max_len = max(len(item[key]) for item in encoded)
            for item in encoded:
                pad_value = -100 if key.endswith("labels") else pad_id
                out[key].append(item[key] + [pad_value] * (max_len - len(item[key])))
        return {key: torch.tensor(value, dtype=torch.long) for key, value in out.items()}

    return collate


def sequence_logprob(model: Any, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute summed log probability over non-masked labels."""
    logits = model(input_ids=input_ids, return_dict=True).logits[:, :-1, :]
    shifted_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = shifted_labels.masked_fill(shifted_labels == -100, 0)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps.masked_fill(shifted_labels == -100, 0.0)
    return token_logps.sum(dim=-1)


def manual_dpo_train(model: Any, tokenizer: Any, pairs: list[dict[str, str]], cfg: dict[str, Any], args: argparse.Namespace) -> Any:
    """Run a compact manual DPO fallback when TRL is unavailable."""
    dpo_cfg = cfg["dpo"]
    device = next(model.parameters()).device
    ref_model = copy.deepcopy(model).eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    loader = DataLoader(
        PairDataset(pairs),
        batch_size=int(args.batch_size or dpo_cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_manual_collator(tokenizer, int(args.max_seq_len or dpo_cfg["max_seq_len"])),
    )
    optimizer = torch.optim.AdamW(no_decay_param_groups(model, 0.0), lr=float(args.lr or dpo_cfg["lr"]))
    beta = float(args.beta or dpo_cfg["beta"])
    model.train()
    step = 0
    for _ in range(int(args.epochs or dpo_cfg["epochs"])):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                chosen = sequence_logprob(model, batch["chosen_input_ids"], batch["chosen_labels"])
                rejected = sequence_logprob(model, batch["rejected_input_ids"], batch["rejected_labels"])
                with torch.no_grad():
                    ref_chosen = sequence_logprob(ref_model, batch["chosen_input_ids"], batch["chosen_labels"])
                    ref_rejected = sequence_logprob(ref_model, batch["rejected_input_ids"], batch["rejected_labels"])
                logits = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
                loss = -F.logsigmoid(logits).mean()
            loss.backward()
            optimizer.step()
            step += 1
            print({"step": step, "dpo_loss": float(loss.item())})
            if args.smoke and step >= 1:
                return model
    return model


def run_dpo(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Run TRL DPO training and save the aligned checkpoint."""
    from datasets import Dataset as HFDataset
    from transformers import PreTrainedTokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = resolve_path(args.model or Path(cfg["paths"]["checkpoint_dir"]) / "sft")
    output_dir = resolve_path(args.output_dir or Path(cfg["paths"]["checkpoint_dir"]) / "dpo")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    tokenizer.add_special_tokens(CHAT_TOKENS)
    model = GPTForCausalLM.load_pretrained(model_dir, map_location=device).to(device)
    model.resize_token_embeddings(len(tokenizer))
    max_samples = args.max_samples or (8 if args.smoke else None)
    pairs = [normalize_pair(item) for item in load_pairs(cfg["dpo"]["dataset"], args.split, max_samples)]

    try:
        trainer = build_trl_trainer(model, tokenizer, HFDataset.from_list(pairs), cfg, args)
        trainer.train()
        trainer.model.save_pretrained(output_dir)
    except Exception as exc:
        print(f"TRL DPOTrainer failed, using manual fallback: {exc}")
        model = manual_dpo_train(model, tokenizer, pairs, cfg, args)
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved DPO checkpoint to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse DPO CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the DPO command."""
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["project"]["seed"]))
    run_dpo(cfg, args)


if __name__ == "__main__":
    main()
