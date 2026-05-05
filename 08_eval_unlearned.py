#!/usr/bin/env python3
"""Rapid post-unlearning evaluation with lm-eval for forgetting vs utility retention."""

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

from lm_eval import evaluator


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent
ADAPTER_PATH = str(PROJECT_ROOT / "final_unlearned_adapter" / "unlearn")
TASKS = ["wmdp-bio", "wmdp-cyber", "arc_challenge", "boolq", "winogrande"]
LIMIT = 100


def run_eval(model_args: Dict, tasks: List[str]) -> Dict:
    return evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        batch_size=1,
        device="cuda:0",
        limit=LIMIT,
        log_samples=False,
        bootstrap_iters=0,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
    )


def normalize_task_names(tasks: List[str]) -> List[str]:
    return [t.replace("-", "_") if t.startswith("wmdp-") else t for t in tasks]


def resolve_tasks(tasks: List[str], error_text: str) -> List[str]:
    """Fallback task naming if harness expects underscores."""
    lowered = error_text.lower()
    if "task" not in lowered and "not found" not in lowered:
        return tasks

    return [t.replace("-", "_") if t.startswith("wmdp-") else t for t in tasks]


def extract_accuracy(metrics: Dict) -> Tuple[str, float]:
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


def print_summary(results: Dict, tasks: List[str]) -> None:
    print("\nEvaluation Summary (accuracy metrics)")
    print("-" * 72)
    print(f"{'Task':<18} {'Metric':<18} {'Score':>12}")
    print("-" * 72)

    for task in tasks:
        task_metrics = results.get("results", {}).get(task, {})
        metric_name, score = extract_accuracy(task_metrics)
        score_text = f"{score:.4f}" if score == score else "n/a"
        print(f"{task:<18} {metric_name:<18} {score_text:>12}")

    print("-" * 72)


def main() -> None:
    model_args_4bit = {
        "pretrained": MODEL_NAME,
        "dtype": "float16",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "trust_remote_code": True,
        "peft": ADAPTER_PATH,
    }
    model_args_fallback = {
        "pretrained": MODEL_NAME,
        "dtype": "float16",
        "trust_remote_code": True,
        "peft": ADAPTER_PATH,
    }

    print("Running rapid lm-eval on unlearned adapter...")
    print(f"Tasks: {TASKS}")
    print(f"Limit per task: {LIMIT}")

    run_tasks = normalize_task_names(TASKS)
    try:
        try:
            results = run_eval(model_args_4bit, run_tasks)
        except TypeError as exc:
            if "load_in_4bit" not in str(exc):
                raise
            print("4-bit kwargs are unsupported in this lm-eval/transformers stack; retrying without them...")
            results = run_eval(model_args_fallback, run_tasks)
    except Exception as exc:
        fallback_tasks = resolve_tasks(run_tasks, str(exc))
        if fallback_tasks == run_tasks:
            raise

        print("Task name fallback triggered; retrying with underscore task names...")
        print(f"Retry tasks: {fallback_tasks}")
        try:
            results = run_eval(model_args_4bit, fallback_tasks)
        except TypeError as type_exc:
            if "load_in_4bit" not in str(type_exc):
                raise
            print("4-bit kwargs are unsupported in this lm-eval/transformers stack; retrying without them...")
            results = run_eval(model_args_fallback, fallback_tasks)
        run_tasks = fallback_tasks

    print_summary(results, run_tasks)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            print("OOM ERROR: CUDA out of memory during evaluation.")
        else:
            print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
