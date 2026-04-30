"""Configuration defaults for the WMDP-Cyber unlearning demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    wmdp_dataset: str = "cais/wmdp"
    wmdp_cyber_subset: str = "wmdp-cyber"
    wmdp_split: str = "test"
    corpus_dataset: str = "cais/wmdp-corpora"
    cyber_forget_subset: str = "cyber-forget-corpus"
    cyber_retain_subset: str = "cyber-retain-corpus"
    output_dir: Path = Path("outputs")
    prepared_dir: Path = Path("outputs/prepared_cyber")
    reinforced_adapter_dir: Path = Path("outputs/reinforced_adapter")
    unlearn_adapter_dir: Path = Path("outputs/unlearn_adapter")
    scores_dir: Path = Path("outputs/scores")
    reports_dir: Path = Path("outputs/reports")
    forget_limit: int = 64
    retain_limit: int = 64
    eval_limit: int = 50
    max_length: int = 256
    load_in_4bit: bool = True
    dtype: str = "float16"
    batch_size: int = 1
    grad_accum_steps: int = 4
    reinforce_lr: float = 2e-4
    unlearn_lr: float = 2e-5
    alpha: float = 2.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    seed: int = 13


DEFAULT_CONFIG = DemoConfig()
