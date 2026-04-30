"""Dependency-light smoke command."""

from __future__ import annotations

from .anchors import build_anchor_map, translate_texts
from .data import assert_no_benchmark_training
from .math_utils import generic_logits, soft_label_cross_entropy, softmax
from .token_alignment import align_translated_to_original


def run_smoke() -> dict[str, object]:
    texts = [
        "Metasploit modules can target HTTP services on Linux systems.",
        "PowerShell scripts often interact with Windows APIs.",
    ]
    anchors = build_anchor_map(texts)
    translated = translate_texts(texts, anchors)
    mapping = align_translated_to_original([1, 2, 3, 4], [1, 9, 4])
    generic = generic_logits([1.0, 0.0, 3.0], [2.0, -1.0, 4.0], alpha=2.0)
    loss = soft_label_cross_entropy([0.1, 0.2, 0.3], softmax(generic))
    assert_no_benchmark_training(
        {"domain": "cyber", "source": {"wmdp_benchmark_used_for_training": False}}
    )
    return {
        "anchors": anchors,
        "translated": translated,
        "mapping": mapping,
        "generic_logits": generic,
        "loss": round(loss, 6),
    }
