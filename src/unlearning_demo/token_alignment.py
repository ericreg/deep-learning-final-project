"""Approximate token-position alignment for translated blocks."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Sequence


def align_translated_to_original(
    original_ids: Sequence[int],
    translated_ids: Sequence[int],
) -> list[int]:
    """Map each original token position to a translated token position.

    The Harry Potter paper needs exact alignment between original blocks and
    translated blocks. This prototype uses a conservative sequence alignment:
    unchanged spans map exactly, replacement spans map to the nearest translated
    token inside the corresponding replacement span, and insert/delete edges
    clamp to the closest valid translated index.
    """

    if not translated_ids:
        raise ValueError("translated_ids must not be empty")

    mapping = [0] * len(original_ids)
    matcher = SequenceMatcher(a=list(original_ids), b=list(translated_ids), autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset, i in enumerate(range(i1, i2)):
                mapping[i] = clamp(j1 + offset, 0, len(translated_ids) - 1)
            continue

        span_len = max(i2 - i1, 1)
        target_len = max(j2 - j1, 1)
        for offset, i in enumerate(range(i1, i2)):
            rel = min(offset, target_len - 1)
            if span_len > target_len and target_len > 1:
                rel = round(offset * (target_len - 1) / max(span_len - 1, 1))
            mapping[i] = clamp(j1 + rel, 0, len(translated_ids) - 1)

    return mapping


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
