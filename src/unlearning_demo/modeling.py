"""Model, tokenizer, and adapter helpers."""

from __future__ import annotations

from .config import DemoConfig
from .device import get_accelerator_info, torch_dtype_from_name
from .imports import require


def build_tokenizer(config: DemoConfig):
    transformers = require("transformers", "Install with: uv pip install -e .")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        candidate = "<|finetune_right_pad_id|>"
        tokenizer.pad_token = candidate if candidate in tokenizer.get_vocab() else tokenizer.eos_token
    return tokenizer


def build_quant_config(config: DemoConfig):
    transformers = require("transformers", "Install with: uv pip install -e .")
    torch = require("torch", "Install a PyTorch build compatible with your accelerator.")
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype_from_name(config.dtype),
    )


def load_base_model(config: DemoConfig, *, trainable: bool = False):
    transformers = require("transformers", "Install with: uv pip install -e .")
    require("torch", "Install a PyTorch build compatible with your accelerator.")
    accelerator = get_accelerator_info()
    if config.load_in_4bit and accelerator.backend == "rocm":
        raise RuntimeError(
            "4-bit bitsandbytes loading is CUDA-oriented in this prototype. "
            "Use --no-4bit on ROCm."
        )

    kwargs = {
        "torch_dtype": torch_dtype_from_name(config.dtype),
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if config.load_in_4bit:
        kwargs["quantization_config"] = build_quant_config(config)
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    model.config.use_cache = False
    if trainable:
        model.gradient_checkpointing_enable()
    return model


def lora_config(config: DemoConfig):
    peft = require("peft", "Install with: uv pip install -e .")
    return peft.LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_targets),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )


def add_trainable_lora(base_model, config: DemoConfig):
    peft = require("peft", "Install with: uv pip install -e .")
    model = peft.prepare_model_for_kbit_training(base_model) if config.load_in_4bit else base_model
    return peft.get_peft_model(model, lora_config(config))
