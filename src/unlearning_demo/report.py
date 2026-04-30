"""Report generation for saved score files."""

from __future__ import annotations

import json
from pathlib import Path

from .config import DemoConfig


def build_report(config: DemoConfig) -> Path:
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(config.scores_dir.glob("wmdp_cyber_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                data.get("adapter", "unknown"),
                data.get("accuracy", 0.0),
                data.get("correct", 0),
                data.get("total", 0),
            )
        )

    lines = [
        "# WMDP-Cyber Unlearning Demo Report",
        "",
        "This report summarizes WMDP-Cyber multiple-choice accuracy. Lower WMDP accuracy after unlearning is the intended forget-direction metric, but this prototype is not a safety claim.",
        "",
        "| Adapter | Accuracy | Correct | Total |",
        "| --- | ---: | ---: | ---: |",
    ]
    if rows:
        for adapter, accuracy, correct, total in rows:
            lines.append(f"| `{adapter}` | {accuracy:.3f} | {correct} | {total} |")
    else:
        lines.append("| _no score files found_ | 0.000 | 0 | 0 |")

    out_path = config.reports_dir / "wmdp_cyber_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
