#!/usr/bin/env python3
"""Run a small lm-eval benchmark on a 4-bit Llama-3 baseline model."""

import json
import sys
import traceback

import torch
from lm_eval import evaluator


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
TASKS = ["wmdp-bio", "wmdp-cyber", "arc_challenge", "boolq", "winogrande"]
LIMIT = 50


def normalize_tasks(tasks: list[str]) -> list[str]:
    return [task.replace("-", "_") if task.startswith("wmdp-") else task for task in tasks]


def display_task_name(task: str) -> str:
    return task.replace("-", "_") if task.startswith("wmdp-") else task


def extract_accuracy(metrics: dict) -> tuple[str, float]:
    """Pick a representative accuracy-like metric for summary display."""
    if not metrics:
        return ("n/a", float("nan"))

    preferred = [
        "acc,none",
        "acc_norm,none",
        "acc",
    ]
    for key in preferred:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return (key, float(metrics[key]))

    for key, value in metrics.items():
        if "acc" in key and isinstance(value, (int, float)):
            return (key, float(value))

    return ("n/a", float("nan"))


def print_summary(results: dict, original_tasks: list[str], eval_tasks: list[str]) -> None:
    print("\nEvaluation Summary (accuracy metrics)")
    print("-" * 72)
    print(f"{'Task':<18} {'Metric':<18} {'Score':>12}")
    print("-" * 72)

    for display_task, eval_task in zip(original_tasks, eval_tasks):
        task_metrics = results.get("results", {}).get(eval_task, {})
        metric_name, score = extract_accuracy(task_metrics)
        score_text = f"{score:.4f}" if score == score else "n/a"
        display_name = display_task_name(display_task)
        print(f"{display_name:<18} {metric_name:<18} {score_text:>12}")

    print("-" * 72)

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

    eval_tasks = normalize_tasks(TASKS)

    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args_4bit,
            tasks=eval_tasks,
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
            tasks=eval_tasks,
            num_fewshot=0,
            batch_size=1,
            device="cuda:0",
            limit=LIMIT,
            log_samples=False,
            bootstrap_iters=0,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
        )

    print_summary(results, TASKS, eval_tasks)

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
