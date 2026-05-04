"""Model smoke tests for shape, masking, and generation."""

import torch

from model import GPTConfig, GPTForCausalLM, build_causal_mask


def tiny_config() -> GPTConfig:
    """Return a compact test configuration."""
    return GPTConfig(vocab_size=128, n_layers=2, d_model=64, n_heads=4, d_ff=128, max_seq_len=32)


def test_forward_shape() -> None:
    """Check that forward returns batch/sequence/vocab logits."""
    model = GPTForCausalLM(tiny_config())
    input_ids = torch.randint(0, 128, (3, 12))
    logits = model(input_ids)
    assert logits.shape == (3, 12, 128)
    assert model.tok_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_causal_mask_blocks_future_positions() -> None:
    """Check that the causal mask blocks future keys."""
    mask = build_causal_mask(seq_len=4, total_len=4, device=torch.device("cpu"))[0, 0]
    assert mask[0, 0].item() == 0
    assert torch.isneginf(mask[0, 1])
    assert torch.isneginf(mask[1, 2])
    assert mask[3, 0].item() == 0
    assert mask[3, 3].item() == 0


def test_generation_smoke() -> None:
    """Check that KV-cache generation appends tokens without error."""
    torch.manual_seed(0)
    model = GPTForCausalLM(tiny_config())
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=4, temperature=1.0, top_p=1.0, top_k=0)
    assert generated.shape[0] == 1
    assert generated.shape[1] >= prompt.shape[1]
    assert generated.shape[1] <= prompt.shape[1] + 4
