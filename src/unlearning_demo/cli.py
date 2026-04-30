"""Command line interface for the WMDP-Cyber unlearning demo."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG, DemoConfig
from .data import prepare_cyber_data
from .preflight import run_preflight
from .retain_eval import run_retain_eval
from .report import build_report
from .score import score_wmdp_cyber
from .smoke import run_smoke
from .train import train_reinforced, train_unlearn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unlearning-demo")
    sub = parser.add_subparsers(dest="command", required=True)

    add_common(sub.add_parser("preflight", help="Check local environment and gated access."))
    add_common(sub.add_parser("prepare", help="Prepare WMDP-Cyber forget/retain data."))
    add_common(sub.add_parser("reinforce", help="Train the reinforced LoRA adapter."))
    add_common(sub.add_parser("unlearn", help="Train the unlearning LoRA adapter."))
    score_parser = sub.add_parser("score", help="Score WMDP-Cyber.")
    add_common(score_parser)
    score_parser.add_argument("--adapter", default="none", help="Adapter path or 'none'.")
    retain_parser = sub.add_parser("retain-eval", help="Run optional lm-eval retained-capability checks.")
    add_common(retain_parser)
    retain_parser.add_argument("--adapter", default="none", help="Adapter path or 'none'.")
    add_common(sub.add_parser("report", help="Build a Markdown report from score files."))
    add_common(sub.add_parser("smoke", help="Run dependency-light smoke checks."))
    return parser


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default=DEFAULT_CONFIG.model_name)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--no-4bit", action="store_true", help="Disable bitsandbytes 4-bit loading; use this on ROCm.")
    parser.add_argument("--dtype", default=DEFAULT_CONFIG.dtype, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--forget-limit", type=int, default=DEFAULT_CONFIG.forget_limit)
    parser.add_argument("--retain-limit", type=int, default=DEFAULT_CONFIG.retain_limit)
    parser.add_argument("--eval-limit", type=int, default=DEFAULT_CONFIG.eval_limit)
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG.max_length)
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha)


def config_from_args(args: argparse.Namespace) -> DemoConfig:
    output_dir = args.output_dir
    return replace(
        DEFAULT_CONFIG,
        model_name=args.model_name,
        output_dir=output_dir,
        prepared_dir=output_dir / "prepared_cyber",
        reinforced_adapter_dir=output_dir / "reinforced_adapter",
        unlearn_adapter_dir=output_dir / "unlearn_adapter",
        scores_dir=output_dir / "scores",
        reports_dir=output_dir / "reports",
        forget_limit=args.forget_limit,
        retain_limit=args.retain_limit,
        eval_limit=args.eval_limit,
        max_length=args.max_length,
        load_in_4bit=not args.no_4bit,
        dtype=args.dtype,
        alpha=args.alpha,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    try:
        if args.command == "preflight":
            results = run_preflight(config)
            for result in results:
                status = "PASS" if result.ok else "FAIL"
                print(f"[{status}] {result.name}: {result.detail}")
            return 0 if all(result.ok for result in results) else 1
        if args.command == "prepare":
            print(prepare_cyber_data(config))
            return 0
        if args.command == "reinforce":
            print(train_reinforced(config))
            return 0
        if args.command == "unlearn":
            print(train_unlearn(config))
            return 0
        if args.command == "score":
            print(score_wmdp_cyber(config, adapter=args.adapter))
            return 0
        if args.command == "retain-eval":
            print(run_retain_eval(config, adapter=args.adapter))
            return 0
        if args.command == "report":
            print(build_report(config))
            return 0
        if args.command == "smoke":
            print(json.dumps(run_smoke(), indent=2, sort_keys=True))
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
