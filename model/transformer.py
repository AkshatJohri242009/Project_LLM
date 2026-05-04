"""Decoder-only GPT model with RoPE, RMSNorm, SwiGLU, KV cache, and HF-style saves."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.generation import GenerationMixin
    from transformers.modeling_outputs import CausalLMOutputWithPast
except Exception:  # pragma: no cover - lets model tests run before optional deps are installed.
    PreTrainedModel = nn.Module
    GenerationMixin = object
    CausalLMOutputWithPast = None

    class PretrainedConfig:
        """Small fallback for tests when transformers is not installed."""

        model_type = "codex_gpt_rope"

        def __init__(self, **kwargs: Any) -> None:
            """Store config values as attributes."""
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to_dict(self) -> dict[str, Any]:
            """Return public config attributes as a dict."""
            return dict(self.__dict__)


class GPTConfig(PretrainedConfig):
    """Configuration for the scratch GPT-style language model."""

    model_type = "codex_gpt_rope"

    def __init__(
        self,
        vocab_size: int = 32000,
        n_layers: int = 12,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        initializer_range: float = 0.02,
        gradient_checkpointing: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize architecture and generation token settings."""
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.gradient_checkpointing = gradient_checkpointing
        self.tie_word_embeddings = True
        self.architectures = ["GPTForCausalLM"]
        self.auto_map = {
            "AutoConfig": "modeling_codex_gpt.GPTConfig",
            "AutoModelForCausalLM": "modeling_codex_gpt.GPTForCausalLM",
        }


class RMSNorm(nn.Module):
    """Root-mean-square normalization without mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Create a learnable RMS scale."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the final dimension by its RMS value."""
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normed


def build_causal_mask(seq_len: int, total_len: int, device: torch.device) -> torch.Tensor:
    """Build an additive upper-triangular causal mask for cached attention."""
    past_len = total_len - seq_len
    query_positions = torch.arange(past_len, total_len, device=device).unsqueeze(1)
    key_positions = torch.arange(total_len, device=device).unsqueeze(0)
    blocked = key_positions > query_positions
    mask = torch.zeros(seq_len, total_len, device=device, dtype=torch.float32)
    return mask.masked_fill(blocked, float("-inf")).view(1, 1, seq_len, total_len)


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of hidden dimensions for RoPE."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to query or key tensors."""
    return (x * cos) + (_rotate_every_two(x) * sin)


class RotaryEmbedding(nn.Module):
    """Generate RoPE sine and cosine tensors on demand."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        """Create inverse frequencies for RoPE."""
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return broadcast-ready cosine and sine tensors."""
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = emb.cos().to(dtype=dtype).view(1, 1, seq_len, -1)
        sin = emb.sin().to(dtype=dtype).view(1, 1, seq_len, -1)
        return cos, sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional KV cache."""

    def __init__(self, config: GPTConfig) -> None:
        """Create attention projections and rotary embedding state."""
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj._is_residual_projection = True
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim)
        self.scale = self.head_dim**-0.5

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        """Split hidden states into attention heads."""
        batch, seq_len, channels = x.shape
        return x.view(batch, seq_len, self.n_heads, channels // self.n_heads).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Run masked attention and optionally return updated KV cache."""
        batch, seq_len, _ = x.shape
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        past_len = 0 if past_key_value is None else past_key_value[0].size(2)
        cos, sin = self.rope(seq_len, x.device, q.dtype, offset=past_len)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)
        present = (k, v) if use_cache else None

        total_len = k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + build_causal_mask(seq_len, total_len, x.device).to(scores.dtype)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :total_len].to(dtype=scores.dtype, device=x.device)
            scores = scores.masked_fill(key_mask == 0, float("-inf"))

        attn = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out), present


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: GPTConfig) -> None:
        """Create gate, value, and residual-down projections."""
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.down_proj._is_residual_projection = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU nonlinearity and project back to model width."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm decoder block with attention and SwiGLU MLP."""

    def __init__(self, config: GPTConfig) -> None:
        """Create one decoder block."""
        super().__init__()
        self.input_norm = RMSNorm(config.d_model)
        self.self_attn = CausalSelfAttention(config)
        self.post_attn_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Run one residual attention and MLP block."""
        attn_out, present = self.self_attn(
            self.input_norm(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attn_norm(x))
        return x, present


class GPTForCausalLM(PreTrainedModel, GenerationMixin):
    """Decoder-only language model with tied token embeddings."""

    config_class = GPTConfig
    base_model_prefix = "gpt"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GPTConfig) -> None:
        """Create the transformer stack and tied language-model head."""
        try:
            super().__init__(config)
        except TypeError:
            nn.Module.__init__(self)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.gradient_checkpointing = bool(getattr(config, "gradient_checkpointing", False))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with GPT-2 style residual scaling."""
        std = self.config.initializer_range
        if getattr(module, "_is_residual_projection", False):
            std *= (2 * self.config.n_layers) ** -0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        """Let transformers toggle gradient checkpointing."""
        self.gradient_checkpointing = value

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the tied input embedding table."""
        return self.tok_emb

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace input embeddings and retie the LM head."""
        self.tok_emb = value
        self.lm_head.weight = self.tok_emb.weight

    def get_output_embeddings(self) -> nn.Linear:
        """Return the tied output projection."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """Replace the output projection."""
        self.lm_head = new_embeddings

    def tie_weights(self) -> None:
        """Tie the LM head to the token embedding table."""
        self.lm_head.weight = self.tok_emb.weight

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare cached inputs for Hugging Face generation helpers."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """Resize tied token embeddings for added special tokens."""
        old_emb = self.tok_emb
        if new_num_tokens == old_emb.num_embeddings:
            return old_emb
        new_emb = nn.Embedding(new_num_tokens, old_emb.embedding_dim, device=old_emb.weight.device)
        nn.init.normal_(new_emb.weight, mean=0.0, std=self.config.initializer_range)
        copy_tokens = min(old_emb.num_embeddings, new_num_tokens)
        new_emb.weight.data[:copy_tokens] = old_emb.weight.data[:copy_tokens]
        self.config.vocab_size = new_num_tokens
        self.set_input_embeddings(new_emb)
        self.lm_head = nn.Linear(old_emb.embedding_dim, new_num_tokens, bias=False, device=old_emb.weight.device)
        self.lm_head.weight = self.tok_emb.weight
        return self.tok_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: Optional[bool] = None,
        **_: Any,
    ) -> Any:
        """Run the model and return logits, or a CausalLMOutput when requested."""
        if input_ids.size(1) > self.config.max_seq_len and past_key_values is None:
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds max_seq_len={self.config.max_seq_len}")
        if self.gradient_checkpointing and self.training:
            use_cache = False

        hidden = self.tok_emb(input_ids)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        past_key_values = past_key_values or [None] * len(self.blocks)

        for block, past in zip(self.blocks, past_key_values):
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    lambda h: block(h, attention_mask=attention_mask, use_cache=False)[0],
                    hidden,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden, present = block(
                    hidden,
                    attention_mask=attention_mask,
                    past_key_value=past,
                    use_cache=use_cache,
                )
            if present is not None:
                presents.append(present)

        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict or labels is not None:
            if CausalLMOutputWithPast is None:
                return {"loss": loss, "logits": logits, "past_key_values": presents or None}
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=presents or None)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens from prompt ids using KV-cache sampling."""
        self.eval()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        device = next(self.parameters()).device
        generated = prompt_ids.to(device)
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        past_key_values = None
        next_input = generated
        attention_mask = torch.ones_like(generated, device=device)

        for _ in range(max_new_tokens):
            outputs = self(
                next_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs["logits"][:, -1, :]
            past_key_values = (
                outputs.past_key_values if hasattr(outputs, "past_key_values") else outputs["past_key_values"]
            )
            next_token = sample_next_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            generated = torch.cat((generated, next_token), dim=1)
            attention_mask = torch.ones(generated.shape, device=device, dtype=torch.long)
            next_input = next_token
            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break
        return generated

    def save_pretrained(self, save_directory: str | Path, *args: Any, **kwargs: Any) -> None:
        """Save weights, config, and remote-code helpers for HF-compatible loading."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        with (save_path / "config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        shutil.copyfile(Path(__file__), save_path / "modeling_codex_gpt.py")
        config_file = save_path / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text(encoding="utf-8"))
            config["auto_map"] = self.config.auto_map
            config["architectures"] = ["GPTForCausalLM"]
            config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_pretrained(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "GPTForCausalLM":
        """Load a model saved by save_pretrained or the training checkpoint writer."""
        load_path = Path(path)
        config_data = json.loads((load_path / "config.json").read_text(encoding="utf-8"))
        config = GPTConfig(**config_data)
        model = cls(config)
        state_path = load_path / "pytorch_model.bin"
        if not state_path.exists() and (load_path / "model.pt").exists():
            state_path = load_path / "model.pt"
        if not state_path.exists() and (load_path / "model.safetensors").exists():
            from safetensors.torch import load_file

            state = load_file(str(load_path / "model.safetensors"), device=str(map_location))
        else:
            state = torch.load(state_path, map_location=map_location)
        if "model" in state:
            state = state["model"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=False)
        return model


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> torch.Tensor:
    """Sample one next token using temperature, top-k, and nucleus filtering."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
