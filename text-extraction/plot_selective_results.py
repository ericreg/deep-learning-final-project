from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS_DIR = Path("selective_unlearning_runs")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "figures"
NOTEBOOK_BLUE = "#4C72B0"
NOTEBOOK_RED = "#C44E52"
NOTEBOOK_ORANGE = "#DD8452"
NOTEBOOK_GREEN = "#55A868"
NOTEBOOK_PURPLE = "#8172B3"
COLORS = [NOTEBOOK_BLUE, NOTEBOOK_RED, NOTEBOOK_GREEN, NOTEBOOK_ORANGE, NOTEBOOK_PURPLE]


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    run_name: str
    results_dir: Path | None
    selection: str
    dry_run: bool
    selected_forget_chunks: int | None
    train_chunk_count: int | None
    eval_chunk_count: int | None
    training_seconds: float | None
    base_hp_perplexity: float | None
    unlearn_hp_perplexity: float | None
    base_control_perplexity: float | None
    unlearn_control_perplexity: float | None
    base_anchor_recall: float | None
    unlearn_anchor_recall: float | None

    @property
    def label(self) -> str:
        timestamp = self.run_name.removeprefix(f"{self.selection}_")
        return f"{self.selection}\n{timestamp}" if timestamp != self.run_name else self.selection

    @property
    def hp_ppl_delta(self) -> float | None:
        return subtract(self.unlearn_hp_perplexity, self.base_hp_perplexity)

    @property
    def control_ppl_delta(self) -> float | None:
        return subtract(self.unlearn_control_perplexity, self.base_control_perplexity)

    @property
    def anchor_recall_delta(self) -> float | None:
        return subtract(self.unlearn_anchor_recall, self.base_anchor_recall)


def subtract(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def normalize_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def config_results_dir(config: dict[str, Any]) -> Path | None:
    results_dir = config.get("results_dir")
    if isinstance(results_dir, str) and results_dir:
        return Path(results_dir)

    chunks_file = config.get("chunks_file")
    if isinstance(chunks_file, str) and chunks_file:
        return Path(chunks_file).parent

    return None


def results_dir_matches(run_results_dir: Path | None, requested_results_dir: Path | None) -> bool:
    if requested_results_dir is None:
        return True
    if run_results_dir is None:
        return False
    return normalize_path(run_results_dir) == normalize_path(requested_results_dir)


def load_run(run_dir: Path) -> RunResult | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    summary = read_json(summary_path)
    metrics_path = run_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    config_path = run_dir / "config.json"
    config = read_json(config_path) if config_path.exists() else {}

    anchor = metrics.get("anchor_recall", {}) if isinstance(metrics.get("anchor_recall"), dict) else {}
    selection = str(summary.get("selection") or metrics.get("selection") or run_dir.name.split("_", 1)[0])

    return RunResult(
        run_dir=run_dir,
        run_name=run_dir.name,
        results_dir=config_results_dir(config),
        selection=selection,
        dry_run=bool(summary.get("dry_run", metrics.get("dry_run", False))),
        selected_forget_chunks=as_int(summary.get("selected_forget_chunks", metrics.get("selected_forget_chunks"))),
        train_chunk_count=as_int(summary.get("train_chunk_count", metrics.get("train_chunk_count"))),
        eval_chunk_count=as_int(summary.get("eval_chunk_count", metrics.get("eval_chunk_count"))),
        training_seconds=as_float(metrics.get("training_seconds")),
        base_hp_perplexity=as_float(summary.get("base_hp_perplexity", metrics.get("base_hp_perplexity"))),
        unlearn_hp_perplexity=as_float(summary.get("unlearn_hp_perplexity", metrics.get("unlearn_hp_perplexity"))),
        base_control_perplexity=as_float(
            summary.get("base_control_perplexity", metrics.get("base_control_perplexity"))
        ),
        unlearn_control_perplexity=as_float(
            summary.get("unlearn_control_perplexity", metrics.get("unlearn_control_perplexity"))
        ),
        base_anchor_recall=as_float(
            summary.get("base_anchor_recall", anchor.get("base_anchor_recall"))
        ),
        unlearn_anchor_recall=as_float(
            summary.get("unlearn_anchor_recall", anchor.get("unlearn_anchor_recall"))
        ),
    )


def summary_mtime(run: RunResult) -> float:
    try:
        return (run.run_dir / "summary.json").stat().st_mtime
    except OSError:
        return 0.0


def discover_runs(
    runs_dir: Path,
    include_dry_runs: bool,
    latest_per_selection: bool,
    results_dir: Path | None = None,
) -> list[RunResult]:
    runs = []
    for summary_path in sorted(runs_dir.glob("**/summary.json")):
        run = load_run(summary_path.parent)
        if run is None:
            continue
        if not results_dir_matches(run.results_dir, results_dir):
            continue
        if run.dry_run and not include_dry_runs:
            continue
        if run.base_hp_perplexity is None and run.unlearn_hp_perplexity is None:
            continue
        runs.append(run)

    if latest_per_selection:
        by_selection = {}
        for run in runs:
            current = by_selection.get(run.selection)
            if current is None or summary_mtime(run) >= summary_mtime(current):
                by_selection[run.selection] = run
        runs = [by_selection[key] for key in sorted(by_selection)]

    return sorted(runs, key=lambda run: (run.selection, run.run_name))


def load_loss_history(run_dir: Path, filename: str) -> list[dict[str, float]]:
    path = run_dir / filename
    if not path.exists():
        return []
    raw = read_json(path)
    if not isinstance(raw, list):
        return []

    history = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        step = as_float(entry.get("step"))
        loss = as_float(entry.get("loss"))
        if step is not None and loss is not None:
            history.append({"step": step, "loss": loss})
    return history


def smooth(values: Sequence[float], divisor: int) -> tuple[np.ndarray, int]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return arr, 1
    window = max(1, len(arr) // divisor)
    if window == 1:
        return arr, window
    return np.convolve(arr, np.ones(window) / window, mode="valid"), window


def finite_or_zero(value: float | None) -> float:
    return value if value is not None and math.isfinite(value) else 0.0


def add_bar_labels(ax, bars, values: Sequence[float | None], precision: int = 3) -> None:
    max_value = max([finite_or_zero(v) for v in values] + [1.0])
    for bar, value in zip(bars, values):
        if value is None:
            label = "n/a"
        else:
            label = f"{value:.{precision}f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_value * 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_core_metrics(runs: Sequence[RunResult], output_dir: Path) -> Path:
    labels = [run.label for run in runs]
    x = np.arange(len(runs))
    width = 0.36

    metric_specs = [
        (
            "Anchor Token Recall ↓",
            "Lower after unlearning = more forgotten",
            [run.base_anchor_recall for run in runs],
            [run.unlearn_anchor_recall for run in runs],
        ),
        (
            "Target Perplexity ↑",
            "Higher after unlearning = more forgotten",
            [run.base_hp_perplexity for run in runs],
            [run.unlearn_hp_perplexity for run in runs],
        ),
        (
            "Control Perplexity →",
            "Should stay close to base",
            [run.base_control_perplexity for run in runs],
            [run.unlearn_control_perplexity for run in runs],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(13, len(runs) * 2.2), 5))
    fig.suptitle("Figure 1 — Selective Unlearning Results: Base vs. Unlearned", fontsize=13, fontweight="bold")

    for ax, (title, subtitle, base_values, unlearn_values) in zip(axes, metric_specs):
        base_bars = ax.bar(x - width / 2, [finite_or_zero(v) for v in base_values], width, color=NOTEBOOK_BLUE, label="Base")
        unlearn_bars = ax.bar(
            x + width / 2,
            [finite_or_zero(v) for v in unlearn_values],
            width,
            color=NOTEBOOK_RED,
            label="Unlearned",
        )
        add_bar_labels(ax, base_bars, base_values)
        add_bar_labels(ax, unlearn_bars, unlearn_values)
        max_value = max([finite_or_zero(v) for v in base_values + unlearn_values] + [1.0])
        ax.set_ylim(0, max_value * 1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        style_axes(ax)

    axes[0].legend()
    return save_figure(fig, output_dir, "fig1_selective_unlearning_results.png")


def plot_strategy_deltas(runs: Sequence[RunResult], output_dir: Path) -> Path:
    labels = [run.label for run in runs]
    x = np.arange(len(runs))

    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(runs) * 1.8), 8))
    fig.suptitle("Figure 2 — Selective Strategy Comparison", fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], [run.anchor_recall_delta for run in runs], "Anchor Recall Change ↓", "Unlearned - Base"),
        (axes[0, 1], [run.hp_ppl_delta for run in runs], "Target Perplexity Change ↑", "Unlearned - Base"),
        (axes[1, 0], [run.control_ppl_delta for run in runs], "Control Perplexity Change →", "Unlearned - Base"),
        (
            axes[1, 1],
            [run.selected_forget_chunks for run in runs],
            "Forget Chunks Used",
            "Count selected for training",
        ),
    ]

    for panel_idx, (ax, values, title, ylabel) in enumerate(panels):
        colors = [COLORS[i % len(COLORS)] for i in range(len(values))]
        bars = ax.bar(x, [finite_or_zero(v) for v in values], color=colors, edgecolor="white", width=0.58)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        add_bar_labels(ax, bars, values, precision=3 if panel_idx < 3 else 0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        style_axes(ax)

    return save_figure(fig, output_dir, "fig2_selective_strategy_comparison.png")


def plot_loss_curves(
    runs: Sequence[RunResult],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    smooth_divisor: int,
) -> Path | None:
    histories = [(run, load_loss_history(run.run_dir, filename)) for run in runs]
    histories = [(run, history) for run, history in histories if history]
    if not histories:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (run, history) in enumerate(histories):
        steps = [entry["step"] for entry in history]
        losses = [entry["loss"] for entry in history]
        color = COLORS[idx % len(COLORS)]
        ax.plot(steps, losses, color=color, linewidth=0.8, alpha=0.25)

        smoothed, window = smooth(losses, smooth_divisor)
        smoothed_steps = steps[window - 1 :]
        chunk_count = run.selected_forget_chunks if run.selected_forget_chunks is not None else "?"
        ax.plot(
            smoothed_steps,
            smoothed,
            color=color,
            linewidth=2,
            label=f"{run.label.replace(chr(10), ' ')} ({chunk_count} chunks, {len(history)} batches, w={window})",
        )

    ax.set_xlabel("Mini-batch Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.01,
        0.01,
        "Selective runs have fewer steps when they train on fewer extracted chunks.",
        transform=ax.transAxes,
        fontsize=8,
        color="gray",
        va="bottom",
    )
    ax.legend(fontsize=8)
    style_axes(ax)
    output_name = "fig3a_reinforced_loss_curves.png" if filename.startswith("reinforced") else "fig3b_unlearn_loss_curves.png"
    return save_figure(fig, output_dir, output_name)


def write_summary_csv(runs: Sequence[RunResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "selective_unlearning_summary.csv"
    fields = [
        "run_name",
        "results_dir",
        "selection",
        "selected_forget_chunks",
        "train_chunk_count",
        "eval_chunk_count",
        "training_seconds",
        "base_anchor_recall",
        "unlearn_anchor_recall",
        "anchor_recall_delta",
        "base_hp_perplexity",
        "unlearn_hp_perplexity",
        "hp_perplexity_delta",
        "base_control_perplexity",
        "unlearn_control_perplexity",
        "control_perplexity_delta",
        "run_dir",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_name": run.run_name,
                    "results_dir": run.results_dir,
                    "selection": run.selection,
                    "selected_forget_chunks": run.selected_forget_chunks,
                    "train_chunk_count": run.train_chunk_count,
                    "eval_chunk_count": run.eval_chunk_count,
                    "training_seconds": run.training_seconds,
                    "base_anchor_recall": run.base_anchor_recall,
                    "unlearn_anchor_recall": run.unlearn_anchor_recall,
                    "anchor_recall_delta": run.anchor_recall_delta,
                    "base_hp_perplexity": run.base_hp_perplexity,
                    "unlearn_hp_perplexity": run.unlearn_hp_perplexity,
                    "hp_perplexity_delta": run.hp_ppl_delta,
                    "base_control_perplexity": run.base_control_perplexity,
                    "unlearn_control_perplexity": run.unlearn_control_perplexity,
                    "control_perplexity_delta": run.control_ppl_delta,
                    "run_dir": run.run_dir,
                }
            )
    return path


def natural_run_sort_key(path: Path) -> tuple[str, str]:
    match = re.match(r"(.+?)_(\d{8}_\d{6})$", path.name)
    if match:
        return match.group(1), match.group(2)
    return path.name, ""


@click.command()
@click.option(
    "--runs-dir",
    default=DEFAULT_RUNS_DIR,
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing selective_unlearn.py run subdirectories.",
)
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for generated figures and CSV summary.",
)
@click.option(
    "--results-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Only plot runs whose selective_unlearn.py config used this results directory, e.g. results/bible.",
)
@click.option(
    "--latest-per-selection/--all-runs",
    default=True,
    show_default=True,
    help="Plot only the newest completed run per selection, or every completed run.",
)
@click.option("--include-dry-runs", is_flag=True, help="Include dry-run summaries if they have plottable metrics.")
def cli(
    runs_dir: Path,
    output_dir: Path,
    results_dir: Path | None,
    latest_per_selection: bool,
    include_dry_runs: bool,
):
    """Plot selective unlearning results from run JSON outputs."""
    if results_dir is not None and output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = output_dir / results_dir.name

    runs = discover_runs(
        runs_dir,
        include_dry_runs=include_dry_runs,
        latest_per_selection=latest_per_selection,
        results_dir=results_dir,
    )
    if not runs:
        message = f"No completed run summaries with plottable metrics found under {runs_dir}"
        if results_dir is not None:
            message += f" for results directory {results_dir}"
        raise click.ClickException(f"{message}.")

    runs = sorted(runs, key=lambda run: natural_run_sort_key(run.run_dir))
    scope = f" for {results_dir}" if results_dir is not None else ""
    click.echo(f"Plotting {len(runs)} run(s){scope}:")
    for run in runs:
        run_scope = f" [{run.results_dir}]" if run.results_dir is not None else ""
        click.echo(f"  - {run.run_name}{run_scope}: {run.selected_forget_chunks} forget chunks")

    outputs = [
        plot_core_metrics(runs, output_dir),
        plot_strategy_deltas(runs, output_dir),
        write_summary_csv(runs, output_dir),
    ]

    reinf_path = plot_loss_curves(
        runs,
        output_dir,
        "reinforced_loss_log.json",
        "Figure 3a — Reinforced Adapter Training Loss",
        "Cross-Entropy Loss",
        smooth_divisor=10,
    )
    if reinf_path:
        outputs.append(reinf_path)

    unlearn_path = plot_loss_curves(
        runs,
        output_dir,
        "unlearn_loss_log.json",
        "Figure 3b — Unlearning KL Loss",
        "KL Divergence Loss",
        smooth_divisor=20,
    )
    if unlearn_path:
        outputs.append(unlearn_path)

    click.echo("Wrote:")
    for path in outputs:
        click.echo(f"  {path}")


if __name__ == "__main__":
    cli()
