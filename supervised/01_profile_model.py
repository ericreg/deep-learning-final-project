#!/usr/bin/env python3
"""Profile 4-bit Llama-3 + LoRA memory usage on a single training step."""

import sys
import traceback

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_LENGTH = 512
BATCH_SIZE = 2


def format_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires an NVIDIA GPU.")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        candidate_pad = "<|finetune_right_pad_id|>"
        if candidate_pad in tokenizer.get_vocab():
            tokenizer.pad_token = candidate_pad
        else:
            tokenizer.pad_token = tokenizer.eos_token

    print("Loading 4-bit model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.train()

    dummy_text = "This is a synthetic sample for memory profiling. " * 200
    batch_text = [dummy_text, dummy_text]

    tokenized = tokenizer(
        batch_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    tokenized["labels"] = tokenized["input_ids"].clone()

    print(
        f"Running forward/backward with batch_size={BATCH_SIZE}, seq_len={MAX_LENGTH}..."
    )
    outputs = model(**tokenized)
    loss = outputs.loss
    loss.backward()

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    print(f"Loss: {loss.item():.6f}")
    print(f"Peak VRAM allocated: {format_gib(peak_alloc):.2f} GiB")
    print(f"Peak VRAM reserved:  {format_gib(peak_reserved):.2f} GiB")
    print("Profiling finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            print("OOM ERROR: CUDA out of memory during profiling step.")
        else:
            print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
