#!/usr/bin/env python3
"""Translate hazardous WMDP terms to generic placeholders and save to disk."""

import json
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, concatenate_datasets, load_dataset


SUBSETS = ["wmdp-bio", "wmdp-cyber"]
OUTPUT_DIR = Path("wmdp_translated")
MAPPING_FILE = Path("wmdp_entities_template.json")


def pick_text_field(example: Dict) -> str:
    if "text" in example and example["text"]:
        return str(example["text"])

    if "question" in example:
        question = str(example["question"])
        choices = example.get("choices")
        if isinstance(choices, list) and choices:
            choices_text = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
            return f"Question: {question}\nChoices:\n{choices_text}"
        return f"Question: {question}"

    return " ".join([str(v) for v in example.values() if v is not None])


def get_preferred_split(ds_dict) -> Dataset:
    if "train" in ds_dict:
        return ds_dict["train"]
    split_name = next(iter(ds_dict.keys()))
    return ds_dict[split_name]


def load_wmdp_dataset() -> Dataset:
    subset_datasets: List[Dataset] = []
    for subset in SUBSETS:
        print(f"Loading dataset subset: {subset}")
        ds_dict = load_dataset("cais/wmdp", subset)
        ds = get_preferred_split(ds_dict)
        ds = ds.map(lambda ex: {"text": pick_text_field(ex)})
        subset_datasets.append(ds)

    merged = concatenate_datasets(subset_datasets)
    print(f"Merged dataset size: {len(merged)} rows")
    return merged


def load_term_mapping() -> Dict[str, str]:
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {MAPPING_FILE}. Run 05a_extract_entities.py first."
        )

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)

    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Expected a JSON object in {MAPPING_FILE}")

    mapping = {
        str(k): str(v)
        for k, v in raw_mapping.items()
        if str(k).strip() and str(v).strip()
    }

    if not mapping:
        raise ValueError(
            f"No non-empty mappings found in {MAPPING_FILE}. Fill values before translating."
        )

    print(f"Loaded {len(mapping)} non-empty mappings from {MAPPING_FILE}")
    return mapping


def prepare_substitution(term_mapping: Dict[str, str]) -> tuple[re.Pattern, Dict[str, str]]:
    # Longest-match-first is critical to avoid collisions like
    # "Cobalt Strike" being partially replaced as "Cobalt".
    sorted_terms = sorted(term_mapping.keys(), key=len, reverse=True)
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(term) for term in sorted_terms) + r")(?!\w)"
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    lower_lookup = {k.lower(): v for k, v in term_mapping.items()}
    return compiled, lower_lookup


def translate_text(text: str, pattern: re.Pattern, lower_lookup: Dict[str, str]) -> str:
    def _replace(match: re.Match) -> str:
        term = match.group(0).lower()
        return lower_lookup.get(term, match.group(0))

    return pattern.sub(_replace, text)


def main() -> None:
    ds = load_wmdp_dataset()
    term_mapping = load_term_mapping()
    pattern, lower_lookup = prepare_substitution(term_mapping)

    print("Applying hazardous-term translation mapping...")

    def _map_fn(example: Dict) -> Dict:
        original_text = str(example["text"])
        translated_text = translate_text(original_text, pattern, lower_lookup)
        return {
            "original_text": original_text,
            "translated_text": translated_text,
        }

    translated = ds.map(_map_fn, remove_columns=ds.column_names)

    changed_count = sum(
        1 for row in translated if row["original_text"] != row["translated_text"]
    )
    print(f"Rows modified by translation: {changed_count}/{len(translated)}")

    preview_n = min(3, len(translated))
    for i in range(preview_n):
        print(f"--- Preview #{i + 1} original ---")
        print(translated[i]["original_text"][:250])
        print(f"--- Preview #{i + 1} translated ---")
        print(translated[i]["translated_text"][:250])

    if OUTPUT_DIR.exists():
        print(f"Removing existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print(f"Saving translated dataset to disk: {OUTPUT_DIR}")
    translated.save_to_disk(str(OUTPUT_DIR))
    print("Translation finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
