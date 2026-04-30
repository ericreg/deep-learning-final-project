"""Preflight checks for the gated Llama/WMDP-Cyber run."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass

from .config import DemoConfig
from .device import get_accelerator_info
from .imports import require


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_preflight(config: DemoConfig) -> list[CheckResult]:
    results = [
        check_python(),
        check_accelerator(config),
        check_hf_model_access(config),
        check_dataset_access(config.corpus_dataset, config.cyber_forget_subset),
        check_dataset_access(config.corpus_dataset, config.cyber_retain_subset),
        check_dataset_access(config.wmdp_dataset, config.wmdp_cyber_subset, split=config.wmdp_split),
    ]
    return results


def check_python() -> CheckResult:
    version = sys.version_info
    ok = (3, 11) <= (version.major, version.minor) < (3, 13)
    detail = f"python={platform.python_version()} required=>=3.11,<3.13"
    return CheckResult("python", ok, detail)


def check_accelerator(config: DemoConfig) -> CheckResult:
    try:
        info = get_accelerator_info()
    except RuntimeError as exc:
        return CheckResult("accelerator", False, str(exc))
    if not info.available:
        return CheckResult("accelerator", False, info.detail)
    if config.load_in_4bit and info.backend == "rocm":
        return CheckResult(
            "accelerator",
            False,
            f"{info.detail} vram_gib={info.total_gib:.2f}; ROCm detected, rerun with --no-4bit.",
        )
    min_vram = 20 if config.load_in_4bit else 32
    detail = f"{info.detail} vram_gib={info.total_gib:.2f} load_in_4bit={config.load_in_4bit}"
    if info.total_gib < min_vram:
        detail += f" warning_recommended_vram_gib>={min_vram}"
    return CheckResult("accelerator", True, detail)


def check_hf_model_access(config: DemoConfig) -> CheckResult:
    try:
        hub = require("huggingface_hub")
    except RuntimeError as exc:
        return CheckResult("hf_model", False, str(exc))
    token = hf_token()
    if not token:
        return CheckResult("hf_model", False, "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is not set.")
    api = hub.HfApi(token=token)
    try:
        info = api.model_info(config.model_name, token=token)
    except Exception as exc:
        return CheckResult("hf_model", False, f"Cannot access {config.model_name}: {exc}")
    return CheckResult("hf_model", True, f"model access OK: {info.id}")


def check_dataset_access(dataset_name: str, subset: str | None, split: str = "train") -> CheckResult:
    try:
        datasets = require("datasets")
    except RuntimeError as exc:
        return CheckResult(f"dataset:{dataset_name}/{subset}", False, str(exc))
    try:
        split_expr = f"{split}[:1]"
        ds = datasets.load_dataset(dataset_name, subset, split=split_expr) if subset else datasets.load_dataset(dataset_name, split=split_expr)
    except Exception as exc:
        return CheckResult(f"dataset:{dataset_name}/{subset}", False, str(exc))
    return CheckResult(f"dataset:{dataset_name}/{subset}", True, f"rows_checked={len(ds)}")


def hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
