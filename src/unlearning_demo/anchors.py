"""Anchor extraction and generic translation for cyber forget text."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


TECH_TERM_RE = re.compile(
    r"""
    (?:
        (?i:\b(?:Cobalt\s+Strike|Metasploit|Meterpreter|PowerShell|Windows|Linux|Android|Ubuntu|Solaris|HTTP|HTTPS|DNS|TCP|UDP|ICMP|ICMPV6)\b)
        |
        \b[A-Z][A-Za-z0-9]*(?:[-_./][A-Za-z0-9]+)+\b
        |
        \b[A-Z]{2,}[A-Za-z0-9-]*\b
        |
        (?i:\b(?:CVE-\d{4}-\d+|MS\d{2}-\d+)\b)
        |
        \b[A-Za-z]+(?:Shell|Sploit|Strike|Fuzzer|Fuzzing|Meterpreter|Beacon)\b
    )
    """,
    flags=re.VERBOSE,
)

GENERIC_BY_HINT = (
    (("cve-", "ms"), "a vulnerability identifier"),
    (("metasploit", "meterpreter", "cobalt", "strike", "beacon"), "a security tool"),
    (("powershell", "shell", "bash", "cmd"), "a command interface"),
    (("http", "icmp", "tcp", "udp", "dns", "packet"), "a network protocol"),
    (("windows", "linux", "android", "ubuntu", "solaris"), "an operating system"),
    (("elf", "apk", "dll", "exe"), "a software artifact"),
    (("fuzzer", "fuzzing", "libfuzzer"), "a testing tool"),
)


@dataclass(frozen=True)
class AnchorReplacement:
    term: str
    replacement: str
    count: int


def infer_generic_replacement(term: str) -> str:
    lower = term.lower()
    for hints, replacement in GENERIC_BY_HINT:
        if any(hint in lower for hint in hints):
            return replacement
    if "-" in term or "_" in term or "/" in term or "." in term:
        return "a technical artifact"
    if term.isupper():
        return "a technical term"
    return "a software component"


def extract_anchor_counts(texts: Iterable[str], min_count: int = 1) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for match in TECH_TERM_RE.finditer(text):
            term = normalize_term(match.group(0))
            if is_valid_anchor(term):
                counts[term] += 1
    return Counter({term: count for term, count in counts.items() if count >= min_count})


def build_anchor_map(
    texts: Iterable[str],
    *,
    min_count: int = 1,
    max_terms: int = 128,
) -> dict[str, str]:
    counts = extract_anchor_counts(texts, min_count=min_count)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
    return {term: infer_generic_replacement(term) for term, _ in ranked[:max_terms]}


def normalize_term(term: str) -> str:
    return term.strip().strip(".,;:!?()[]{}\"'`")


def is_valid_anchor(term: str) -> bool:
    if len(term) < 3:
        return False
    if term.lower() in {"the", "and", "for", "with", "from"}:
        return False
    return any(ch.isalpha() for ch in term)


def prepare_anchor_pattern(anchor_map: dict[str, str]) -> re.Pattern[str] | None:
    if not anchor_map:
        return None
    terms = sorted(anchor_map, key=len, reverse=True)
    body = "|".join(re.escape(term) for term in terms)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", flags=re.IGNORECASE)


def translate_text(text: str, anchor_map: dict[str, str]) -> str:
    pattern = prepare_anchor_pattern(anchor_map)
    if pattern is None:
        return text
    lookup = {key.lower(): value for key, value in anchor_map.items()}

    def replace(match: re.Match[str]) -> str:
        return lookup.get(match.group(0).lower(), match.group(0))

    return pattern.sub(replace, text)


def translate_texts(texts: Iterable[str], anchor_map: dict[str, str]) -> list[str]:
    return [translate_text(text, anchor_map) for text in texts]
