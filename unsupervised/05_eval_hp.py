#!/usr/bin/env python3
"""
HP-specific evaluation suite.

Three measurement axes:
  1. Cloze accuracy  — fill-in-the-blank for HP facts (higher = more recall)
  2. Perplexity      — on HP text vs. non-HP control (lower HP perplexity = more retained)
  3. General benchmarks via lm-eval (arc_easy, boolq, winogrande)

Run with --adapter ./hp_unlearned_adapter/unlearn to evaluate the unlearned model,
or omit --adapter to evaluate the raw base model as a baseline.
"""

import argparse
import math
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_LENGTH = 512

# ---------------------------------------------------------------------------
# HP cloze probes  (prefix → expected completion)
# Each prefix is presented to the model; we measure rank of the expected token.
# ---------------------------------------------------------------------------
HP_CLOZE_PROBES: List[Tuple[str, str]] = [
    ("The spell used to disarm an opponent is", "Expelliarmus"),
    ("Harry Potter attends a school called", "Hogwarts"),
    ("Harry Potter's owl is named", "Hedwig"),
    ("The sport played on broomsticks at Hogwarts is called", "Quidditch"),
    ("Harry Potter's arch-nemesis is", "Voldemort"),
    ("Harry Potter's best friends are Ron and", "Hermione"),
    ("The house Harry belongs to at Hogwarts is", "Gryffindor"),
    ("The charm to make objects levitate is Wingardium", "Leviosa"),
    ("The train to Hogwarts departs from platform nine and", "three"),
    ("The spell to produce light from a wand is", "Lumos"),
    ("Harry's lightning-bolt shaped", "scar"),
    ("The Defense Against the Dark Arts teacher in year one was Professor", "Quirrell"),
]

# Non-HP control sentences used to measure baseline perplexity
CONTROL_TEXT = (
    "The mitochondria is the powerhouse of the cell and produces ATP through "
    "oxidative phosphorylation. Water boils at one hundred degrees Celsius at "
    "standard atmospheric pressure. The French Revolution began in 1789 and "
    "fundamentally transformed European political structures."
)


def load_model(adapter_path: Optional[str]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
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

    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        print(f"Loaded adapter from: {adapter_path}")
    else:
        model = base
        print("Evaluating base model (no adapter)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Cloze evaluation
# ---------------------------------------------------------------------------

def token_rank(model, tokenizer, prefix: str, expected: str, device: torch.device) -> int:
    """Return the rank (1-indexed) of the first token of `expected` given `prefix`."""
    enc = tokenizer(prefix, return_tensors="pt").to(device)
    target_id = tokenizer.encode(" " + expected.strip(), add_special_tokens=False)[0]

    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[0, -1]   # next-token logits after the full prefix
        ranked = logits.argsort(descending=True).tolist()

    rank = ranked.index(target_id) + 1 if target_id in ranked else len(ranked)
    return rank


def eval_cloze(model, tokenizer, device: torch.device) -> Dict:
    print("\n--- Cloze evaluation ---")
    print(f"{'Prefix':<55} {'Expected':<15} {'Rank':>6}")
    print("-" * 80)

    ranks = []
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0

    for prefix, expected in HP_CLOZE_PROBES:
        rank = token_rank(model, tokenizer, prefix, expected, device)
        ranks.append(rank)
        if rank == 1:
            top1_hits += 1
        if rank <= 5:
            top5_hits += 1
        if rank <= 10:
            top10_hits += 1
        print(f"{prefix[:54]:<55} {expected:<15} {rank:>6}")

    n = len(HP_CLOZE_PROBES)
    print("-" * 80)
    print(f"Top-1  accuracy: {top1_hits}/{n}  ({100*top1_hits/n:.1f}%)")
    print(f"Top-5  accuracy: {top5_hits}/{n}  ({100*top5_hits/n:.1f}%)")
    print(f"Top-10 accuracy: {top10_hits}/{n}  ({100*top10_hits/n:.1f}%)")
    print(f"Mean rank: {sum(ranks)/n:.1f}")

    return {"top1": top1_hits / n, "top5": top5_hits / n, "top10": top10_hits / n, "mean_rank": sum(ranks) / n}


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, text: str, device: torch.device) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    input_ids = enc["input_ids"]
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    return math.exp(out.loss.item())


def eval_perplexity(model, tokenizer, device: torch.device, hp_text: Optional[str]) -> None:
    print("\n--- Perplexity evaluation ---")
    if hp_text:
        hp_ppl = compute_perplexity(model, tokenizer, hp_text[:2000], device)
        print(f"HP text perplexity:      {hp_ppl:.2f}  (lower = model retains more HP knowledge)")
    ctrl_ppl = compute_perplexity(model, tokenizer, CONTROL_TEXT, device)
    print(f"Control text perplexity: {ctrl_ppl:.2f}  (should remain low — general knowledge intact)")


# ---------------------------------------------------------------------------
# lm-eval benchmarks
# ---------------------------------------------------------------------------

def eval_benchmarks(adapter_path: Optional[str]) -> None:
    try:
        from lm_eval import evaluator
    except ImportError:
        print("\nlm-eval not installed — skipping benchmark evaluation.")
        return

    print("\n--- General benchmark evaluation (lm-eval) ---")
    tasks = ["arc_easy", "boolq", "winogrande"]
    model_args: Dict = {
        "pretrained": MODEL_NAME,
        "dtype": "float16",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
    }
    if adapter_path:
        model_args["peft"] = adapter_path

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        batch_size=1,
        device="cuda:0",
        limit=100,
        log_samples=False,
        bootstrap_iters=0,
    )

    print(f"\n{'Task':<18} {'Accuracy':>10}")
    print("-" * 32)
    for task in tasks:
        metrics = results.get("results", {}).get(task, {})
        acc = metrics.get("acc,none", metrics.get("acc", float("nan")))
        print(f"{task:<18} {acc:>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter directory (omit for base model baseline)")
    parser.add_argument("--hp-text-file", type=str, default=None,
                        help="Path to a plain-text excerpt of HP for perplexity measurement")
    parser.add_argument("--skip-benchmarks", action="store_true",
                        help="Skip lm-eval benchmarks (faster)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for evaluation.")

    model, tokenizer = load_model(args.adapter)
    device = next(model.parameters()).device

    eval_cloze(model, tokenizer, device)

    hp_text = None
    if args.hp_text_file:
        with open(args.hp_text_file) as f:
            hp_text = f.read()
    eval_perplexity(model, tokenizer, device, hp_text)

    if not args.skip_benchmarks:
        eval_benchmarks(args.adapter)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        traceback.print_exc()
        sys.exit(1)
