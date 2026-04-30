"""Framework-light math helpers used by tests and torch code."""

from __future__ import annotations

import math
from typing import Iterable


def generic_logits(
    baseline_translated: Iterable[float],
    reinforced_original: Iterable[float],
    alpha: float,
) -> list[float]:
    return [
        base - alpha * max(reinf - base, 0.0)
        for base, reinf in zip(baseline_translated, reinforced_original)
    ]


def softmax(values: Iterable[float]) -> list[float]:
    vals = list(values)
    if not vals:
        return []
    max_val = max(vals)
    exps = [math.exp(v - max_val) for v in vals]
    total = sum(exps)
    return [v / total for v in exps]


def soft_label_cross_entropy(logits: Iterable[float], target_probs: Iterable[float]) -> float:
    logit_vals = list(logits)
    target_vals = list(target_probs)
    probs = softmax(logit_vals)
    return -sum(target * math.log(max(prob, 1e-12)) for target, prob in zip(target_vals, probs))
