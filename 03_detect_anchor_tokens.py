#!/usr/bin/env python3
"""
Identify anchor tokens: positions where the reinforced model's probability for the
true next token exceeds the baseline by more than ANCHOR_THRESHOLD.

Saves per-chunk binary anchor masks alongside the original text to ./hp_anchored/.
This is the key step that differentiates the paper's approach from naive substitution:
the model itself decides which tokens are "over-known" rather than a hand-crafted list.
"""

import sys
import traceback
from typing import Dict

import torch
from datasets import Dataset, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CORPUS_DIR = "./hp_corpus"
REINFORCED_ADAPTER_DIR = "./hp_reinforced_adapter"
OUTPUT_DIR = "./hp_anchored"
MAX_LENGTH = 512
ANCHOR_THRESHOLD = 2.0  # ratio: p_reinforced / p_baseline must exceed this


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model() -> PeftModel:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base,
        REINFORCED_ADAPTER_DIR,
        adapter_name="reinforced",
        is_trainable=False,
    )
    model.eval()
    return model


def get_token_probs(
    model: PeftModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    use_adapter: bool,
) -> torch.Tensor:
    """Returns softmax probabilities [seq_len, vocab_size] for each position."""
    with torch.no_grad():
        if use_adapter:
            model.set_adapter("reinforced")
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with model.disable_adapter():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
    return torch.softmax(out.logits[0], dim=-1)   # [seq_len, vocab_size]


def compute_anchor_mask(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> Dict:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    seq_len = input_ids.shape[1]

    # True next-token ids shifted by 1: position i predicts token i+1
    true_ids = input_ids[0, 1:]   # [seq_len - 1]

    probs_base = get_token_probs(model, input_ids, attention_mask, use_adapter=False)
    probs_reinf = get_token_probs(model, input_ids, attention_mask, use_adapter=True)

    # Gather the probability assigned to the true next token at each position
    positions = torch.arange(seq_len - 1, device=device)
    p_base = probs_base[positions, true_ids]         # [seq_len-1]
    p_reinf = probs_reinf[positions, true_ids]       # [seq_len-1]

    ratio = p_reinf / (p_base + 1e-9)
    anchor_mask = (ratio >= ANCHOR_THRESHOLD).cpu().tolist()

    anchor_rate = sum(anchor_mask) / max(len(anchor_mask), 1)
    return {
        "text": text,
        "anchor_mask": anchor_mask,      # list[bool], length = seq_len - 1
        "anchor_rate": anchor_rate,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for anchor token detection.")

    tokenizer = build_tokenizer()
    corpus = load_from_disk(CORPUS_DIR)
    print(f"Corpus size: {len(corpus)} chunks")

    model = load_model()
    device = next(model.parameters()).device

    results = []
    for i, row in enumerate(corpus):
        entry = compute_anchor_mask(model, tokenizer, row["text"], device)
        results.append(entry)
        if (i + 1) % 50 == 0 or (i + 1) == len(corpus):
            mean_rate = sum(r["anchor_rate"] for r in results) / len(results)
            print(f"[{i+1}/{len(corpus)}] mean anchor rate: {mean_rate:.3f}")

        torch.cuda.empty_cache()

    # anchor_mask is a list of bools — store as string for Dataset compatibility
    anchored_ds = Dataset.from_dict({
        "text": [r["text"] for r in results],
        "anchor_mask": [str(r["anchor_mask"]) for r in results],
        "anchor_rate": [r["anchor_rate"] for r in results],
    })
    anchored_ds.save_to_disk(OUTPUT_DIR)

    overall_rate = sum(r["anchor_rate"] for r in results) / len(results)
    print(f"\nAnchor detection complete.")
    print(f"Overall anchor rate: {overall_rate:.3f} ({overall_rate*100:.1f}% of tokens flagged)")
    print(f"Dataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        traceback.print_exc()
        sys.exit(1)
