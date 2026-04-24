#!/usr/bin/env python3
"""Run a small lm-eval benchmark on a 4-bit Llama-3 baseline model."""

import json
import sys
import traceback

import torch
from lm_eval import evaluator


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
TASKS = ["arc_challenge", "boolq", "winogrande"]
LIMIT = 50


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Baseline eval requires an NVIDIA GPU.")

    model_args_4bit = {
        "pretrained": MODEL_NAME,
        "dtype": "float16",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "trust_remote_code": True,
    }
    model_args_fallback = {
        "pretrained": MODEL_NAME,
        "dtype": "float16",
        "trust_remote_code": True,
    }

    print(
        "Model settings: "
        + ",".join(
            [
                f"pretrained={MODEL_NAME}",
                "dtype=float16",
                "load_in_4bit=True",
                "bnb_4bit_quant_type=nf4",
                "bnb_4bit_compute_dtype=float16",
                "trust_remote_code=True",
                "apply_chat_template=True",
            ]
        )
    )

    print("Starting lm-eval dry-run benchmark...")
    print(f"Tasks: {TASKS}")
    print(f"Limit per task: {LIMIT}")

    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args_4bit,
            tasks=TASKS,
            num_fewshot=0,
            batch_size=1,
            device="cuda:0",
            limit=LIMIT,
            log_samples=False,
            bootstrap_iters=0,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
        )
    except TypeError as exc:
        # Some lm-eval/transformers combos reject 4-bit kwargs at model init.
        if "load_in_4bit" not in str(exc):
            raise
        print("4-bit kwargs are not supported by this stack; retrying in fp16...")
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args_fallback,
            tasks=TASKS,
            num_fewshot=0,
            batch_size=1,
            device="cuda:0",
            limit=LIMIT,
            log_samples=False,
            bootstrap_iters=0,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
        )

    print("Evaluation metrics:")
    for task in TASKS:
        task_metrics = results.get("results", {}).get(task, {})
        print(f"- {task}: {json.dumps(task_metrics, sort_keys=True)}")

    print("Baseline evaluation finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            print("OOM ERROR: CUDA out of memory during baseline eval.")
        else:
            print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
