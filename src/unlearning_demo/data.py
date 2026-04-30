"""Dataset loading and preparation for WMDP-Cyber."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .anchors import build_anchor_map, translate_texts
from .config import DemoConfig
from .imports import require


def load_dataset_split(dataset_name: str, subset: str | None = None, split: str = "train"):
    datasets = require("datasets", "Install with: uv pip install -e .")
    if subset:
        return datasets.load_dataset(dataset_name, subset, split=split)
    return datasets.load_dataset(dataset_name, split=split)


def pick_text(row: dict[str, Any]) -> str:
    for key in ("text", "text_chunk", "content", "document"):
        value = row.get(key)
        if value:
            return str(value)
    return " ".join(str(value) for value in row.values() if value is not None)


def pick_wmdp_prompt(row: dict[str, Any]) -> tuple[str, list[str], int | str | None]:
    question = str(row.get("question", "")).strip()
    choices = row.get("choices") or row.get("answer_choices")
    if not isinstance(choices, list) or len(choices) != 4:
        raise ValueError("Expected a WMDP row with exactly four choices.")
    answer = row.get("answer")
    return question, [str(choice) for choice in choices], answer


def load_cyber_corpora(config: DemoConfig) -> tuple[list[str], list[str]]:
    forget = load_dataset_split(config.corpus_dataset, config.cyber_forget_subset)
    retain = load_dataset_split(config.corpus_dataset, config.cyber_retain_subset)
    forget_texts = [pick_text(row) for row in forget.select(range(min(config.forget_limit, len(forget))))]
    retain_texts = [pick_text(row) for row in retain.select(range(min(config.retain_limit, len(retain))))]
    return forget_texts, retain_texts


def load_wmdp_cyber_eval(config: DemoConfig):
    return load_dataset_split(config.wmdp_dataset, config.wmdp_cyber_subset, split=config.wmdp_split)


def prepare_cyber_data(config: DemoConfig) -> Path:
    config.prepared_dir.mkdir(parents=True, exist_ok=True)
    forget_texts, retain_texts = load_cyber_corpora(config)
    anchor_map = build_anchor_map(forget_texts, min_count=1, max_terms=128)
    translated = translate_texts(forget_texts, anchor_map)

    payload = {
        "config": serialize_config(config),
        "domain": "cyber",
        "source": {
            "forget": {
                "dataset": config.corpus_dataset,
                "subset": config.cyber_forget_subset,
                "limit": len(forget_texts),
            },
            "retain": {
                "dataset": config.corpus_dataset,
                "subset": config.cyber_retain_subset,
                "limit": len(retain_texts),
            },
            "wmdp_benchmark_used_for_training": False,
        },
        "forget_texts": forget_texts,
        "retain_texts": retain_texts,
        "translated_forget_texts": translated,
        "anchor_map": anchor_map,
        "fingerprints": {
            "forget": fingerprint_texts(forget_texts),
            "retain": fingerprint_texts(retain_texts),
        },
    }
    out_path = config.prepared_dir / "cyber_prepared.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_prepared(config: DemoConfig) -> dict[str, Any]:
    path = config.prepared_dir / "cyber_prepared.json"
    if not path.exists():
        raise FileNotFoundError(f"Prepared data not found: {path}. Run `unlearning-demo prepare`.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert_no_benchmark_training(payload)
    return payload


def assert_no_benchmark_training(payload: dict[str, Any]) -> None:
    source = payload.get("source", {})
    used = source.get("wmdp_benchmark_used_for_training")
    if used is not False:
        raise ValueError("Benchmark contamination guard failed: WMDP rows may be in training data.")
    if payload.get("domain") != "cyber":
        raise ValueError("This prototype is configured for WMDP-Cyber only.")


def fingerprint_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()


def serialize_config(config: DemoConfig) -> dict[str, Any]:
    raw = asdict(config)
    for key, value in list(raw.items()):
        if isinstance(value, Path):
            raw[key] = str(value)
    return raw
