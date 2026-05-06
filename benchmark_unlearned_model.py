import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np

from query_unlearned_model import infer_model_id, load_run_config, read_json, resolve_adapter_paths
from selective_unlearn import dump_json


DEFAULT_TASKS = ("arc_easy", "boolq", "winogrande")
TASK_LABELS = {
    "arc_easy": "ARC-Easy",
    "boolq": "BoolQ",
    "winogrande": "Winogrande",
}
BASE_COLOR = "#4C72B0"
UNLEARNED_COLOR = "#C44E52"
IMPROVEMENT_COLOR = "#27ae60"
REGRESSION_COLOR = "#e74c3c"
SOURCE_NAME_ALIASES = {
    "illad": "iliad",
    "illiad": "iliad",
}


def import_lm_eval():
    try:
        from lm_eval import evaluator
    except ImportError as exc:
        raise click.ClickException(
            "Missing lm_eval. Install it with:\n"
            "  uv sync\n"
            "or refresh the lockfile with:\n"
            "  uv lock"
        ) from exc
    return evaluator


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task)


def metric_value(task_metrics: dict[str, Any]) -> float | None:
    for key in ("acc,none", "acc", "acc_norm,none", "acc_norm", "exact_match,none", "exact_match"):
        value = task_metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def extract_scores(results: dict[str, Any], tasks: tuple[str, ...]) -> dict[str, float | None]:
    raw_results = results.get("results", {})
    if not isinstance(raw_results, dict):
        return {task: None for task in tasks}
    return {
        task: metric_value(raw_results.get(task, {})) if isinstance(raw_results.get(task, {}), dict) else None
        for task in tasks
    }


def canonical_source_name(source_name: str | None) -> str | None:
    if not source_name:
        return None
    return SOURCE_NAME_ALIASES.get(source_name, source_name)


def config_run_label(config: dict[str, Any]) -> str | None:
    results_dir = config.get("results_dir")
    selection = config.get("selection")
    if not isinstance(results_dir, str) or not isinstance(selection, str):
        return None
    source_name = canonical_source_name(Path(results_dir).name)
    return f"{source_name}/{selection}" if source_name else selection


def path_run_label(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    source_name = canonical_source_name(run_dir.parent.name)
    selection = run_dir.name
    if not source_name or source_name == "selective_unlearning_runs":
        return selection
    return f"{source_name}/{selection}"


def resolve_run_label(run_dir: Path | None, adapter_root: Path, config: dict[str, Any]) -> str:
    return config_run_label(config) or path_run_label(run_dir) or adapter_root.name


def run_lm_eval(
    model_id: str,
    tasks: tuple[str, ...],
    dtype: str,
    device: str,
    batch_size: str,
    limit: int | None,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    evaluator = import_lm_eval()
    model_args: dict[str, Any] = {"pretrained": model_id, "dtype": dtype}
    if adapter_path is not None:
        model_args["peft"] = str(adapter_path)

    return evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=list(tasks),
        num_fewshot=0,
        batch_size=batch_size,
        device=device,
        limit=limit,
        log_samples=False,
        bootstrap_iters=0,
    )


def write_benchmark_plot(
    scores: dict[str, dict[str, float | None]],
    tasks: tuple[str, ...],
    figure_file: Path,
    run_label: str,
) -> Path:
    figure_file = svg_path(figure_file)
    base_scores = [scores["base"].get(task) or 0.0 for task in tasks]
    unlearned_scores = [scores["unlearned"].get(task) or 0.0 for task in tasks]
    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    base_bars = ax.bar(x - width / 2, base_scores, width, color=BASE_COLOR, label="Base Model")
    unlearned_bars = ax.bar(x + width / 2, unlearned_scores, width, color=UNLEARNED_COLOR, label="Unlearned")

    for bars, values in ((base_bars, base_scores), (unlearned_bars, unlearned_scores)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for i, (base, unlearned) in enumerate(zip(base_scores, unlearned_scores)):
        if base <= 0:
            continue
        delta = (unlearned - base) / base * 100
        color = IMPROVEMENT_COLOR if delta >= 0 else REGRESSION_COLOR
        ax.text(
            i,
            max(base, unlearned) + 0.06,
            f"{delta:+.1f}%",
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([task_label(task) for task in tasks], fontsize=11)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Figure 5 - General Capability Benchmarks: {run_label} (should be preserved)", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_file, format="svg", bbox_inches="tight")
    plt.close(fig)
    return figure_file


def default_output_file(run_dir: Path | None, adapter_root: Path) -> Path:
    return (run_dir or adapter_root) / "general_benchmarks.json"


def default_figure_file(output_file: Path) -> Path:
    return output_file.with_name("fig5_benchmark_retention.svg")


def svg_path(figure_file: Path) -> Path:
    return figure_file if figure_file.suffix.lower() == ".svg" else figure_file.with_suffix(".svg")


def update_run_metrics(run_dir: Path | None, payload: dict[str, Any]) -> Path | None:
    if run_dir is None:
        return None
    metrics_path = run_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    if not isinstance(metrics, dict):
        metrics = {}
    metrics["general_benchmarks"] = {
        "tasks": payload["tasks"],
        "scores": payload["scores"],
        "limit": payload["limit"],
        "num_fewshot": payload["num_fewshot"],
        "run_label": payload["run_label"],
    }
    dump_json(metrics_path, metrics)
    return metrics_path


@click.command()
@click.option(
    "--run-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Selective unlearning run directory containing unlearned_adapter/.",
)
@click.option(
    "--adapter-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Unlearned adapter directory. Defaults to RUN_DIR/unlearned_adapter.",
)
@click.option("--model-id", default=None, help="Base model id/path. Defaults to the run config or adapter config.")
@click.option("--adapter-name", default="unlearn", show_default=True, help="Adapter name saved by selective_unlearn.py.")
@click.option("--task", "tasks", multiple=True, help="lm-eval task name. Defaults to ARC-Easy, BoolQ, Winogrande.")
@click.option("--dtype", default="bfloat16", show_default=True, type=click.Choice(["bfloat16", "float16", "float32"]))
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--batch-size", default="8", show_default=True, help="lm-eval batch size, e.g. 8 or auto.")
@click.option("--limit", default=100, show_default=True, type=click.IntRange(min=1), help="Examples per task.")
@click.option("--full-benchmark", is_flag=True, help="Evaluate full benchmark tasks instead of using --limit.")
@click.option("--skip-base", is_flag=True, help="Only evaluate the unlearned adapter.")
@click.option("--skip-unlearned", is_flag=True, help="Only evaluate the base model.")
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write benchmark JSON. Defaults to RUN_DIR/general_benchmarks.json.",
)
@click.option(
    "--figure-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the benchmark retention SVG plot. Defaults beside output JSON.",
)
@click.option("--plot/--no-plot", default=True, show_default=True)
@click.option("--update-metrics/--no-update-metrics", default=True, show_default=True)
def cli(
    run_dir: Path | None,
    adapter_dir: Path | None,
    model_id: str | None,
    adapter_name: str,
    tasks: tuple[str, ...],
    dtype: str,
    device: str,
    batch_size: str,
    limit: int,
    full_benchmark: bool,
    skip_base: bool,
    skip_unlearned: bool,
    output_file: Path | None,
    figure_file: Path | None,
    plot: bool,
    update_metrics: bool,
):
    """Evaluate base and unlearned models on general capability benchmarks."""
    if skip_base and skip_unlearned:
        raise click.ClickException("At least one of base or unlearned must be evaluated.")

    started = time.perf_counter()
    tasks = tasks or DEFAULT_TASKS
    eval_limit = None if full_benchmark else limit

    config = load_run_config(run_dir)
    adapter_root, adapter_path = resolve_adapter_paths(run_dir, adapter_dir, adapter_name)
    resolved_model_id = infer_model_id(model_id, config, adapter_path)
    output_path = output_file or default_output_file(run_dir, adapter_root)
    figure_path = svg_path(figure_file or default_figure_file(output_path))
    run_label = resolve_run_label(run_dir, adapter_root, config)

    scores: dict[str, dict[str, float | None]] = {}
    if not skip_base:
        click.echo(f"Evaluating base model on {', '.join(tasks)}")
        base_results = run_lm_eval(
            resolved_model_id,
            tasks,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
            limit=eval_limit,
        )
        scores["base"] = extract_scores(base_results, tasks)

    if not skip_unlearned:
        click.echo(f"Evaluating unlearned adapter {adapter_path} on {', '.join(tasks)}")
        unlearned_results = run_lm_eval(
            resolved_model_id,
            tasks,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
            limit=eval_limit,
            adapter_path=adapter_path,
        )
        scores["unlearned"] = extract_scores(unlearned_results, tasks)

    if "base" not in scores:
        scores["base"] = {task: None for task in tasks}
    if "unlearned" not in scores:
        scores["unlearned"] = {task: None for task in tasks}

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "adapter_root": str(adapter_root),
        "adapter_path": str(adapter_path),
        "adapter_name": adapter_name,
        "run_label": run_label,
        "model_id": resolved_model_id,
        "tasks": list(tasks),
        "task_labels": {task: task_label(task) for task in tasks},
        "dtype": dtype,
        "device": device,
        "batch_size": batch_size,
        "limit": eval_limit,
        "num_fewshot": 0,
        "scores": scores,
        "elapsed_seconds": time.perf_counter() - started,
    }

    dump_json(output_path, payload)
    click.echo(f"Wrote benchmark scores to {output_path}")

    if update_metrics:
        metrics_path = update_run_metrics(run_dir, payload)
        if metrics_path is not None:
            click.echo(f"Updated run metrics at {metrics_path}")

    if plot:
        written_figure_path = write_benchmark_plot(scores, tasks, figure_path, run_label)
        click.echo(f"Wrote benchmark plot to {written_figure_path}")


if __name__ == "__main__":
    cli()
