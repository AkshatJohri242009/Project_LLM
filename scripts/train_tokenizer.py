"""Train a byte-level BPE tokenizer and save it in Hugging Face fast-tokenizer format."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

from train.utils import load_config, resolve_path, seed_everything


SPECIAL_DEFAULTS = {
    "pad": "<|pad|>",
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "unk": "<|unk|>",
}


def get_nested(example: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Return the first present top-level or metadata value."""
    metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return default


def passes_language_filter(example: dict[str, Any], language: str, min_score: float) -> bool:
    """Check FineWeb/C4-style language metadata when it exists."""
    detected = get_nested(example, ["language", "lang"], language)
    score = float(get_nested(example, ["language_score", "lang_score", "language_confidence"], 1.0))
    return detected == language and score >= min_score


def dataset_texts(
    name: str,
    config: str | None,
    split: str,
    text_field: str,
    language: str,
    min_score: float,
) -> Iterator[str]:
    """Stream text examples from a Hugging Face dataset."""
    from datasets import load_dataset

    dataset = load_dataset(name, config, split=split, streaming=True)
    for example in dataset:
        if not isinstance(example, dict):
            continue
        if not passes_language_filter(example, language=language, min_score=min_score):
            continue
        text = example.get(text_field) or example.get("text") or ""
        if isinstance(text, str) and text.strip():
            yield text


def limited_bytes(texts: Iterable[str], sample_bytes: int) -> Iterator[str]:
    """Yield text until the requested UTF-8 byte budget is reached."""
    seen = 0
    for text in texts:
        encoded_len = len(text.encode("utf-8", errors="ignore"))
        if seen >= sample_bytes:
            break
        seen += encoded_len
        yield text


def tiny_corpus() -> Iterator[str]:
    """Yield a small built-in corpus for offline smoke tests."""
    samples = [
        "Language models learn statistical patterns in text and predict the next token.",
        "A tiny smoke corpus is not useful for quality, but it validates the tokenizer path.",
        "The quick brown fox wrote a transformer training script with careful tests.",
        "Instruction tuning formats prompts and assistant answers into supervised examples.",
    ]
    while True:
        yield from samples


def build_text_iterator(cfg: dict[str, Any], args: argparse.Namespace) -> Iterator[str]:
    """Create the best available tokenizer training text stream."""
    tok_cfg = cfg["tokenizer"]
    data_cfg = cfg.get("data", {})
    candidates = [
        (tok_cfg["dataset"], tok_cfg.get("dataset_config")),
        (tok_cfg.get("fallback_dataset", "wikitext"), tok_cfg.get("fallback_config")),
    ]
    if args.smoke:
        return limited_bytes(tiny_corpus(), args.sample_bytes)
    for name, config in candidates:
        try:
            stream = dataset_texts(
                name=name,
                config=config,
                split=args.split,
                text_field=data_cfg.get("text_field", "text"),
                language=data_cfg.get("language", "en"),
                min_score=float(data_cfg.get("language_score_threshold", 0.9)),
            )
            first = next(stream)
            return limited_bytes(chain([first], stream), args.sample_bytes)
        except Exception as exc:
            print(f"Skipping tokenizer source {name}/{config}: {exc}")
    print("Falling back to built-in tiny corpus.")
    return limited_bytes(tiny_corpus(), args.sample_bytes)


def train_tokenizer(texts: Iterable[str], output_dir: Path, vocab_size: int, special_tokens: dict[str, str]) -> None:
    """Train and save a byte-level BPE tokenizer."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token=special_tokens["unk"]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            special_tokens["pad"],
            special_tokens["bos"],
            special_tokens["eos"],
            special_tokens["unk"],
        ],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=special_tokens["pad"],
        bos_token=special_tokens["bos"],
        eos_token=special_tokens["eos"],
        unk_token=special_tokens["unk"],
    )
    fast.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    """Parse tokenizer CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-bytes", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run tokenizer training from the command line."""
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["project"]["seed"]))
    tok_cfg = cfg["tokenizer"]
    special_tokens = {**SPECIAL_DEFAULTS, **tok_cfg.get("special_tokens", {})}
    output_dir = resolve_path(args.output_dir or cfg["paths"]["tokenizer_dir"])
    sample_bytes = args.sample_bytes or int(tok_cfg["sample_bytes"])
    if args.smoke:
        sample_bytes = min(sample_bytes, 200_000)
    args.sample_bytes = sample_bytes
    texts = build_text_iterator(cfg, args)
    train_tokenizer(
        texts=texts,
        output_dir=output_dir,
        vocab_size=args.vocab_size or int(tok_cfg["vocab_size"]),
        special_tokens=special_tokens,
    )
    print(f"Saved tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
