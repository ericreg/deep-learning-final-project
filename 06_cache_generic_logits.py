#!/usr/bin/env python3
"""Cache generic logits via two-pass baseline/reinforced inference with explicit VRAM flush."""

import gc
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
TRANSLATED_DATA_DIR = Path("wmdp_translated")
ADAPTER_DIR = Path("./reinforced_adapter")
OUTPUT_FILE = Path("generic_labels.pt")
ALPHA = 1.5
LIMIT = 10
MAX_LENGTH = 256
INFER_BATCH_SIZE = 2


def gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def print_vram(stage: str) -> None:
    if not torch.cuda.is_available():
        print(f"[{stage}] CUDA not available.")
        return

    device = torch.device("cuda")
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_alloc = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    used_bytes = total_bytes - free_bytes

    print(f"[{stage}] VRAM allocated: {gib(allocated):.2f} GiB")
    print(f"[{stage}] VRAM reserved:  {gib(reserved):.2f} GiB")
    print(f"[{stage}] VRAM max alloc: {gib(max_alloc):.2f} GiB")
    print(f"[{stage}] VRAM max reserv:{gib(max_reserved):.2f} GiB")
    print(f"[{stage}] VRAM used/total:{gib(used_bytes):.2f}/{gib(total_bytes):.2f} GiB")


def build_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        candidate_pad = "<|finetune_right_pad_id|>"
        if candidate_pad in tokenizer.get_vocab():
            tokenizer.pad_token = candidate_pad
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_text_pairs(limit: int) -> Dict[str, List[str]]:
    if not TRANSLATED_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Translated dataset directory not found: {TRANSLATED_DATA_DIR}. Run 05_translate_data.py first."
        )

    ds = load_from_disk(str(TRANSLATED_DATA_DIR))
    required_cols = {"original_text", "translated_text"}
    if not required_cols.issubset(set(ds.column_names)):
        raise ValueError(
            "Translated dataset is missing required columns original_text/translated_text. "
            "Re-run 05_translate_data.py."
        )

    n = min(limit, len(ds))
    subset = ds.select(range(n))
    return {
        "original": [str(x) for x in subset["original_text"]],
        "translated": [str(x) for x in subset["translated_text"]],
    }


def encode_texts(tokenizer: AutoTokenizer, texts: List[str]) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def infer_logits_cpu(
    model: torch.nn.Module,
    encoded_batch: Dict[str, torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    total = encoded_batch["input_ids"].size(0)
    outputs_cpu: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = {
                "input_ids": encoded_batch["input_ids"][start:end].to(device),
                "attention_mask": encoded_batch["attention_mask"][start:end].to(device),
            }
            out = model(**batch)
            outputs_cpu.append(out.logits.detach().to(device="cpu", dtype=torch.float16))

            del batch
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return torch.cat(outputs_cpu, dim=0)


def load_base_model_4bit() -> torch.nn.Module:
    quant_config = build_quant_config()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Generic logit caching requires an NVIDIA GPU.")

    print("Preparing tokenizer and translated/original text pairs...")
    tokenizer = build_tokenizer()
    pairs = load_text_pairs(LIMIT)
    print(f"Loaded {len(pairs['translated'])} aligned text pairs")

    translated_tokens = encode_texts(tokenizer, pairs["translated"])
    original_tokens = encode_texts(tokenizer, pairs["original"])

    print("=== Pass 1: Baseline model on translated sequences ===")
    print_vram("before-pass1-load")
    baseline_model = load_base_model_4bit()
    print_vram("after-pass1-load")
    v_baseline = infer_logits_cpu(
        baseline_model,
        translated_tokens,
        batch_size=INFER_BATCH_SIZE,
    )
    print_vram("after-pass1-forward")
    print(f"v_baseline shape: {tuple(v_baseline.shape)} | device={v_baseline.device}")

    print("=== VRAM Flush ===")
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    print_vram("after-flush")

    print("=== Pass 2: Reinforced model on original sequences ===")
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_DIR}. Run 04_train_reinforced.py first."
        )

    print_vram("before-pass2-load")
    reinforced_base = load_base_model_4bit()
    reinforced_model = PeftModel.from_pretrained(reinforced_base, str(ADAPTER_DIR))
    print_vram("after-pass2-load")
    v_reinforced = infer_logits_cpu(
        reinforced_model,
        original_tokens,
        batch_size=INFER_BATCH_SIZE,
    )
    print_vram("after-pass2-forward")
    print(f"v_reinforced shape: {tuple(v_reinforced.shape)} | device={v_reinforced.device}")

    if v_baseline.shape != v_reinforced.shape:
        raise ValueError(
            f"Logit shapes do not match: baseline={tuple(v_baseline.shape)} "
            f"reinforced={tuple(v_reinforced.shape)}"
        )

    print("Computing v_generic = v_baseline - alpha * ReLU(v_reinforced - v_baseline)")
    delta = torch.relu(v_reinforced.float() - v_baseline.float())
    v_generic = (v_baseline.float() - ALPHA * delta).to(dtype=torch.float16)

    payload = {
        "alpha": ALPHA,
        "num_sequences": v_generic.size(0),
        "max_length": v_generic.size(1),
        "vocab_size": v_generic.size(2),
        "v_generic": v_generic,
    }
    torch.save(payload, OUTPUT_FILE)

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 ** 2)
    print(f"Saved generic labels to {OUTPUT_FILE} ({file_size:.2f} MiB)")
    print("Generic logit caching finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            print("OOM ERROR: CUDA out of memory during generic logit caching.")
        else:
            print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
