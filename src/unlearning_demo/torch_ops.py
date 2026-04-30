"""Torch operations for generic targets and soft-label training."""

from __future__ import annotations


def build_generic_logits(v_baseline_translated, v_reinforced_original, alpha: float):
    torch = __import__("torch")
    return v_baseline_translated - alpha * torch.relu(v_reinforced_original - v_baseline_translated)


def masked_soft_cross_entropy(logits, target_logits, attention_mask=None):
    torch = __import__("torch")
    functional = __import__("torch.nn.functional", fromlist=["functional"])
    vocab_size = logits.shape[-1]
    target_probs = torch.softmax(target_logits, dim=-1)
    per_token = functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_probs.reshape(-1, vocab_size),
        reduction="none",
    ).reshape(logits.shape[:-1])
    if attention_mask is None:
        return per_token.mean()
    mask = attention_mask.to(dtype=per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)
