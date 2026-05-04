"""Stream, deduplicate, tokenize, pack, and shard pretraining text data."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np

from train.utils import load_config, resolve_path, seed_everything


WORD_RE = re.compile(r"\w+")


def words(text: str) -> list[str]:
    """Return lowercase word tokens for filtering and shingles."""
    return WORD_RE.findall(text.lower())


def quality_ok(text: str) -> bool:
    """Keep documents with a plausible average characters-per-word range."""
    doc_words = words(text)
    if len(doc_words) < 20:
        return False
    avg_chars_per_word = sum(len(word) for word in doc_words) / max(1, len(doc_words))
    return 2.0 <= avg_chars_per_word <= 10.0


def shingles(doc_words: list[str], size: int) -> set[str]:
    """Return contiguous word shingles."""
    if len(doc_words) < size:
        return set()
    return {" ".join(doc_words[i : i + size]) for i in range(len(doc_words) - size + 1)}


def stable_hash(value: str, seed: int) -> int:
    """Hash a string and seed into a deterministic 64-bit integer."""
    payload = f"{seed}:{value}".encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def minhash_signature(items: set[str], permutations: int) -> tuple[int, ...]:
    """Compute a deterministic MinHash signature."""
    if not items:
        return tuple([0] * permutations)
    return tuple(min(stable_hash(item, seed) for item in items) for seed in range(permutations))


def signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not left or len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


@dataclass
class MinHashLSH:
    """Small streaming MinHash LSH deduplicator."""

    permutations: int = 64
    bands: int = 16
    threshold: float = 0.85
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = field(default_factory=lambda: defaultdict(list))
    signatures: list[tuple[int, ...]] = field(default_factory=list)

    def is_duplicate(self, signature: tuple[int, ...]) -> bool:
        """Return true when a near-duplicate signature has been seen."""
        rows = max(1, self.permutations // self.bands)
        candidates: set[int] = set()
        for band in range(self.bands):
            start = band * rows
            key = (band, signature[start : start + rows])
            candidates.update(self.buckets.get(key, []))
        return any(signature_similarity(signature, self.signatures[idx]) >= self.threshold for idx in candidates)

    def add(self, signature: tuple[int, ...]) -> None:
        """Add a signature to the LSH index."""
        rows = max(1, self.permutations // self.bands)
        doc_id = len(self.signatures)
        self.signatures.append(signature)
        for band in range(self.bands):
            start = band * rows
            key = (band, signature[start : start + rows])
            self.buckets[key].append(doc_id)


class ShardWriter:
    """Write flat uint16 token shards for one split."""

    def __init__(self, output_dir: Path, split: str, seq_len: int, shard_sequences: int) -> None:
        """Create a buffered writer for packed token sequences."""
        self.output_dir = output_dir
        self.split = split
        self.seq_len = seq_len
        self.shard_sequences = shard_sequences
        self.buffer: list[int] = []
        self.shard_idx = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_sequence(self, tokens: list[int]) -> None:
        """Append one fixed-length token sequence."""
        if len(tokens) != self.seq_len:
            raise ValueError("Packed sequence length mismatch")
        self.buffer.extend(tokens)
        if len(self.buffer) >= self.seq_len * self.shard_sequences:
            self.flush()

    def flush(self) -> None:
        """Flush buffered tokens to a .bin shard."""
        if not self.buffer:
            return
        path = self.output_dir / f"{self.split}_{self.shard_idx:06d}.bin"
        np.asarray(self.buffer, dtype=np.uint16).tofile(path)
        print(f"Wrote {path} ({len(self.buffer) // self.seq_len} sequences)")
        self.buffer.clear()
        self.shard_idx += 1


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
    """Check optional language metadata for English-only filtering."""
    detected = get_nested(example, ["language", "lang"], language)
    score = float(get_nested(example, ["language_score", "lang_score", "language_confidence"], 1.0))
    return detected == language and score >= min_score


def stream_dataset(
    name: str,
    config: str | None,
    split: str,
    text_field: str,
    language: str,
    min_score: float,
) -> Iterator[str]:
    """Stream text from a Hugging Face dataset."""
    from datasets import load_dataset

    dataset = load_dataset(name, config, split=split, streaming=True)
    for example in dataset:
        if not isinstance(example, dict) or not passes_language_filter(example, language, min_score):
            continue
        text = example.get(text_field) or example.get("text") or ""
        if isinstance(text, str) and text.strip():
            yield text


def tiny_texts() -> Iterator[str]:
    """Yield a deterministic offline smoke-test corpus."""
    base = [
        "A transformer predicts the next token from previous context using masked attention, residual connections, normalization, and a learned vocabulary of subword units.",
        "Data preparation packs token ids into fixed length arrays for fast training while keeping validation shards separate from training shards.",
        "Deduplication reduces repeated web documents before language model pretraining so the model sees more diverse examples during optimization.",
        "Evaluation measures perplexity and downstream task accuracy after training, then writes compact JSON reports for later comparison.",
    ]
    while True:
        yield from base


def build_source(cfg: dict[str, Any], args: argparse.Namespace) -> Iterator[str]:
    """Return the first available text stream."""
    data_cfg = cfg["data"]
    if args.smoke:
        return tiny_texts()
    candidates = [(data_cfg["source_dataset"], data_cfg.get("source_config"))]
    candidates.extend((item["name"], item.get("config")) for item in data_cfg.get("fallback_datasets", []))
    for name, config in candidates:
        try:
            stream = stream_dataset(
                name=name,
                config=config,
                split=args.split,
                text_field=data_cfg.get("text_field", "text"),
                language=data_cfg.get("language", "en"),
                min_score=float(data_cfg.get("language_score_threshold", 0.9)),
            )
            first = next(stream)
            return chain([first], stream)
        except Exception as exc:
            print(f"Skipping data source {name}/{config}: {exc}")
    print("Falling back to built-in tiny corpus.")
    return tiny_texts()


def tokenizer_ids(tokenizer: Any, text: str, eos_token_id: int) -> list[int]:
    """Tokenize text and append EOS."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "ids"):
        ids = ids.ids
    ids = list(ids)
    ids.append(eos_token_id)
    return ids


def prepare_data(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Run streaming deduplication, tokenization, packing, and sharding."""
    from transformers import PreTrainedTokenizerFast

    data_cfg = cfg["data"]
    tokenizer_dir = resolve_path(args.tokenizer_dir or cfg["paths"]["tokenizer_dir"])
    output_dir = resolve_path(args.output_dir or cfg["paths"]["processed_dir"])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    seq_len = int(args.seq_len or data_cfg["seq_len"])
    val_fraction = float(data_cfg["val_fraction"])
    source = build_source(cfg, args)
    lsh = MinHashLSH(
        permutations=int(data_cfg["minhash_permutations"]),
        bands=int(data_cfg["minhash_bands"]),
        threshold=float(data_cfg["jaccard_threshold"]),
    )
    train_writer = ShardWriter(output_dir, "train", seq_len, int(data_cfg["shard_sequences"]))
    val_writer = ShardWriter(output_dir, "val", seq_len, max(1, int(data_cfg["shard_sequences"]) // 4))
    pack_buffer: list[int] = []
    kept_docs = 0
    written_sequences = 0

    for doc_idx, text in enumerate(source):
        if args.max_docs and doc_idx >= args.max_docs:
            break
        if not quality_ok(text):
            continue
        doc_words = words(text)
        doc_shingles = shingles(doc_words, int(data_cfg["shingle_size"]))
        signature = minhash_signature(doc_shingles, int(data_cfg["minhash_permutations"]))
        if lsh.is_duplicate(signature):
            continue
        lsh.add(signature)
        pack_buffer.extend(tokenizer_ids(tokenizer, text, tokenizer.eos_token_id))
        kept_docs += 1

        while len(pack_buffer) >= seq_len:
            sequence = pack_buffer[:seq_len]
            del pack_buffer[:seq_len]
            writer = val_writer if random.random() < val_fraction else train_writer
            writer.write_sequence(sequence)
            written_sequences += 1

        if args.max_sequences and written_sequences >= args.max_sequences:
            break

    train_writer.flush()
    val_writer.flush()
    print(f"Kept {kept_docs} documents after filtering and deduplication.")


def parse_args() -> argparse.Namespace:
    """Parse data preparation CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the data preparation pipeline."""
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["project"]["seed"]))
    if args.smoke:
        args.max_docs = args.max_docs or 2000
        args.max_sequences = args.max_sequences or 64
    prepare_data(cfg, args)


if __name__ == "__main__":
    main()
