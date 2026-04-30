"""Optional retained-capability checks via lm-evaluation-harness."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import DemoConfig
from .device import require_accelerator
from .imports import require


RETAIN_TASKS = ("arc_challenge", "boolq", "winogrande")


def run_retain_eval(
    config: DemoConfig,
    *,
    adapter: str | None = None,
    limit: int | None = None,
) -> Path:
    require("torch", "Install a PyTorch build compatible with your accelerator.")
    evaluator = require("lm_eval.evaluator", "Install optional eval deps with: uv pip install -e '.[eval]'")
    accelerator = require_accelerator()

    model_args = {
        "pretrained": config.model_name,
        "dtype": config.dtype,
        "trust_remote_code": True,
    }
    if config.load_in_4bit:
        if accelerator.backend == "rocm":
            raise RuntimeError("Use --no-4bit for retain-eval on ROCm.")
        model_args.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": config.dtype,
            }
        )
    if adapter and adapter != "none":
        model_args["peft"] = adapter

    eval_limit = limit if limit is not None else config.eval_limit
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=list(RETAIN_TASKS),
        num_fewshot=0,
        batch_size=1,
        device=accelerator.device_arg,
        limit=eval_limit,
        log_samples=False,
        bootstrap_iters=0,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter": adapter or "none",
        "limit": eval_limit,
        "tasks": list(RETAIN_TASKS),
        "results": results.get("results", {}),
    }
    config.scores_dir.mkdir(parents=True, exist_ok=True)
    safe_name = (adapter or "baseline").replace("/", "_").replace("\\", "_")
    out_path = config.scores_dir / f"retain_{safe_name}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
