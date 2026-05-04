"""Command-line and library generation helpers for scratch GPT checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from model import GPTForCausalLM, sample_next_token


def quantize_model_int8(model: GPTForCausalLM) -> GPTForCausalLM:
    """Best-effort conversion of linear layers to bitsandbytes INT8 modules."""
    if not torch.cuda.is_available():
        print("--quantize requested but CUDA is unavailable; keeping fp weights.")
        return model
    try:
        import bitsandbytes as bnb
    except Exception as exc:
        print(f"--quantize requested but bitsandbytes is unavailable: {exc}")
        return model

    def replace(module: torch.nn.Module) -> None:
        """Recursively replace eligible Linear layers."""
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear) and name != "lm_head":
                quant = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                )
                quant.weight = bnb.nn.Int8Params(child.weight.data, requires_grad=False, has_fp16_weights=False)
                if child.bias is not None:
                    quant.bias = child.bias
                setattr(module, name, quant)
            else:
                replace(child)

    replace(model)
    return model


def load_model_and_tokenizer(model_path: str | Path, quantize: bool = False) -> tuple[GPTForCausalLM, Any]:
    """Load a checkpoint and its tokenizer for inference."""
    from transformers import PreTrainedTokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTForCausalLM.load_pretrained(model_path, map_location=device).to(device)
    if quantize:
        model = quantize_model_int8(model)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(
    model: GPTForCausalLM,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    """Generate text for one prompt with the model KV cache."""
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if prompt_ids.size(1) > model.config.max_seq_len:
        prompt_ids = prompt_ids[:, -model.config.max_seq_len :]
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = output_ids[0, prompt_ids.size(1) :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def batch_generate(
    model: GPTForCausalLM,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> list[str]:
    """Generate text for a batch of prompts using the shared sampler."""
    return [
        generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        for prompt in prompts
    ]


@torch.no_grad()
def stream_generate(
    model: GPTForCausalLM,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> Iterator[str]:
    """Yield decoded text fragments token by token with KV-cache sampling."""
    device = next(model.parameters()).device
    generated = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if generated.size(1) > model.config.max_seq_len:
        generated = generated[:, -model.config.max_seq_len :]
    next_input = generated
    past_key_values = None
    attention_mask = torch.ones_like(generated, device=device)
    for _ in range(max_new_tokens):
        outputs = model(
            next_input,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = sample_next_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
            break
        generated = torch.cat((generated, next_token), dim=1)
        attention_mask = torch.ones(generated.shape, device=device, dtype=torch.long)
        next_input = next_token
        piece = tokenizer.decode(next_token[0], skip_special_tokens=True)
        if piece:
            yield piece


def parse_args() -> argparse.Namespace:
    """Parse generation CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/dpo")
    parser.add_argument("--prompt", default="Explain gravity to a 5 year old.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--quantize", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run CLI text generation."""
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model, quantize=args.quantize)
    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
