#!/usr/bin/env python3
"""Download, filter, tokenize WMDP data, and validate a PyTorch DataLoader."""

import sys
import traceback
from typing import Dict, List

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_LENGTH = 512
BATCH_SIZE = 8
SUBSETS = ["wmdp-bio", "wmdp-cyber"]


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

    # Safe fallback when schema differs unexpectedly.
    return " ".join([str(v) for v in example.values() if v is not None])


def get_preferred_split(ds_dict) -> Dataset:
    if "train" in ds_dict:
        return ds_dict["train"]
    split_name = next(iter(ds_dict.keys()))
    return ds_dict[split_name]


def load_wmdp_subsets() -> Dataset:
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


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        candidate_pad = "<|finetune_right_pad_id|>"
        if candidate_pad in tokenizer.get_vocab():
            tokenizer.pad_token = candidate_pad
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _tokenize(batch: Dict) -> Dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = ds.map(_tokenize, batched=True, remove_columns=ds.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


def main() -> None:
    merged = load_wmdp_subsets()
    tokenizer = build_tokenizer()
    tokenized = tokenize_dataset(merged, tokenizer)

    dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True)
    first_batch = next(iter(dataloader))

    print(f"DataLoader type: {type(dataloader)}")
    print(f"input_ids shape: {tuple(first_batch['input_ids'].shape)}")
    print(f"attention_mask shape: {tuple(first_batch['attention_mask'].shape)}")
    print("Data preparation finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            print("OOM ERROR: CUDA out of memory during data preparation.")
        else:
            print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
