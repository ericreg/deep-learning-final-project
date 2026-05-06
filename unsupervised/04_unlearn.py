#!/usr/bin/env python3
"""
Faithful implementation of the paper's unlearning step.

For each anchor token position the replacement label is the baseline model's argmax
prediction (what a model that never read HP would predict). For non-anchor positions
the label is the ground-truth token. Fine-tuning on these replacement labels steers
the model away from HP-specific predictions without touching general capabilities.
"""

import ast
import sys
import traceback
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
ANCHORED_DIR = "./hp_anchored"
REINFORCED_ADAPTER_DIR = "./hp_reinforced_adapter"
OUTPUT_ADAPTER_DIR = "./hp_unlearned_adapter"

LEARNING_RATE = 5e-4
EPOCHS = 3
GRAD_ACCUM_STEPS = 2
MAX_LENGTH = 512


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_multi_adapter_model() -> PeftModel:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base = prepare_model_for_kbit_training(base)

    # Load the frozen reinforced adapter as a reference
    model = PeftModel.from_pretrained(
        base,
        REINFORCED_ADAPTER_DIR,
        adapter_name="reinforced",
        is_trainable=False,
    )

    # Add a new trainable adapter that learns to suppress HP-specific knowledge
    unlearn_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.add_adapter("unlearn", unlearn_config)

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name and ".unlearn." in name

    model.set_adapter("unlearn")
    model.train()
    return model


def build_replacement_labels(
    model: PeftModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    anchor_mask: List[bool],
    device: torch.device,
) -> torch.Tensor:
    """
    For each token position i that predicts token i+1:
      - If anchor_mask[i] is True  → label = argmax of baseline model at position i
      - If anchor_mask[i] is False → label = true next token (input_ids[i+1])

    Returns labels tensor of shape [seq_len] matching input_ids, with -100 at position 0
    (no label for the first token since it has no predecessor to predict from).
    """
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        with model.disable_adapter():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            baseline_argmax = out.logits[0].argmax(dim=-1)   # [seq_len]

    true_next = input_ids[0, 1:]          # [seq_len - 1]
    baseline_preds = baseline_argmax[:-1]  # baseline prediction at each position

    # Convert anchor_mask to tensor; may be shorter than seq_len-1 due to truncation
    mask_len = min(len(anchor_mask), seq_len - 1)
    anchor_t = torch.tensor(anchor_mask[:mask_len], dtype=torch.bool, device=device)

    labels_body = torch.where(anchor_t, baseline_preds[:mask_len], true_next[:mask_len])

    # Pad to seq_len with -100 (ignored by cross-entropy): prepend one -100 for position 0,
    # append -100s if truncation shortened the mask
    pad_right = seq_len - 1 - mask_len
    labels = torch.cat([
        torch.tensor([-100], device=device),
        labels_body,
        torch.full((pad_right,), -100, device=device),
    ])
    return labels   # [seq_len]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for unlearning.")

    tokenizer = build_tokenizer()
    dataset = load_from_disk(ANCHORED_DIR)
    print(f"Anchored dataset: {len(dataset)} chunks")

    model = load_multi_adapter_model()
    device = next(model.parameters()).device

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)

    total = len(dataset)
    global_step = 0

    print(f"Starting unlearning: epochs={EPOCHS}, grad_accum={GRAD_ACCUM_STEPS}")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        for idx in range(total):
            row = dataset[idx]
            text = row["text"]
            anchor_mask: List[bool] = ast.literal_eval(row["anchor_mask"])

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Build hard replacement labels using the baseline model
            labels = build_replacement_labels(
                model, input_ids, attention_mask, anchor_mask, device
            )

            # Forward pass through the unlearn adapter
            model.set_adapter("unlearn")
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits   # [1, seq_len, vocab]

            loss = F.cross_entropy(
                logits[0],                        # [seq_len, vocab]
                labels,                           # [seq_len]
                ignore_index=-100,
            )
            (loss / GRAD_ACCUM_STEPS).backward()

            global_step += 1
            should_step = (idx + 1) % GRAD_ACCUM_STEPS == 0 or (idx + 1) == total
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            if global_step % 50 == 0 or global_step == total * EPOCHS:
                print(f"  step {global_step}/{total * EPOCHS}  loss={loss.item():.5f}")

            del input_ids, attention_mask, labels, out, logits, loss

    print(f"\nSaving unlearned adapter to: {OUTPUT_ADAPTER_DIR}")
    model.set_adapter("unlearn")
    model.save_pretrained(OUTPUT_ADAPTER_DIR, selected_adapters=["unlearn"])
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)
    print("Unlearning complete.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        traceback.print_exc()
        sys.exit(1)
