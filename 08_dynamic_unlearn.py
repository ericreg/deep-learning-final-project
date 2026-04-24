#!/usr/bin/env python3
"""Phase 3 dynamic unlearning over full WMDP using PEFT multi-adapter switching."""

import os
import sys
import traceback
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SUBSETS = ["wmdp-bio", "wmdp-cyber"]
TRANSLATED_DATA_DIR = "wmdp_translated"
REINFORCED_ADAPTER_DIR = "./reinforced_adapter"
OUTPUT_ADAPTER_DIR = "./final_unlearned_adapter"

ALPHA = 1.5
LEARNING_RATE = 2e-5
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
MAX_LENGTH = 256


def pick_text_field(example: Dict) -> str:
    if "text" in example and example["text"]:
        return str(example["text"])

    if "question" in example:
        question = str(example["question"])
        choices = example.get("choices")
        if isinstance(choices, list) and choices:
            choices_text = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
            return f"Question: {question}\nChoices:\n{choices_text}"
        return f"Question: {question}"

    return " ".join([str(v) for v in example.values() if v is not None])


def get_preferred_split(ds_dict) -> Dataset:
    if "train" in ds_dict:
        return ds_dict["train"]
    split_name = next(iter(ds_dict.keys()))
    return ds_dict[split_name]


def load_original_wmdp_dataset() -> Dataset:
    subset_datasets: List[Dataset] = []
    for subset in SUBSETS:
        print(f"Loading original dataset subset: {subset}")
        ds_dict = load_dataset("cais/wmdp", subset)
        ds = get_preferred_split(ds_dict)
        ds = ds.map(lambda ex: {"text": pick_text_field(ex)})
        subset_datasets.append(ds)

    merged = concatenate_datasets(subset_datasets)
    print(f"Merged original dataset size: {len(merged)} rows")
    return merged


def load_translated_dataset() -> Dataset:
    print(f"Loading translated dataset from: {TRANSLATED_DATA_DIR}")
    ds = load_from_disk(TRANSLATED_DATA_DIR)
    required_cols = {"original_text", "translated_text"}
    if not required_cols.issubset(set(ds.column_names)):
        raise ValueError(
            "Translated dataset missing required columns original_text/translated_text. "
            "Re-run 05_translate_data.py."
        )
    print(f"Translated dataset size: {len(ds)} rows")
    return ds


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        candidate_pad = "<|finetune_right_pad_id|>"
        if candidate_pad in tokenizer.get_vocab():
            tokenizer.pad_token = candidate_pad
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_single(tokenizer: AutoTokenizer, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


def build_multi_adapter_model() -> PeftModel:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading base model in 4-bit NF4...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    print("Loading frozen reinforced adapter as 'reinforced'...")
    model = PeftModel.from_pretrained(
        base_model,
        REINFORCED_ADAPTER_DIR,
        adapter_name="reinforced",
        is_trainable=False,
    )

    print("Initializing new trainable adapter as 'unlearn'...")
    unlearn_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.add_adapter("unlearn", unlearn_config)

    # Freeze everything except the new unlearn adapter weights.
    for name, param in model.named_parameters():
        param.requires_grad = ("lora_" in name and ".unlearn." in name)

    model.set_adapter("unlearn")
    model.train()
    return model


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Dynamic unlearning requires an NVIDIA GPU.")

    tokenizer = build_tokenizer()
    original_ds = load_original_wmdp_dataset()
    translated_ds = load_translated_dataset()

    if len(original_ds) != len(translated_ds):
        raise ValueError(
            f"Dataset length mismatch: original={len(original_ds)} translated={len(translated_ds)}"
        )

    total_rows = len(original_ds)
    row_limit = os.environ.get("ROW_LIMIT")
    if row_limit:
        total_rows = min(total_rows, int(row_limit))
        print(f"ROW_LIMIT enabled: using first {total_rows} rows")
    print(f"Total synchronized training rows: {total_rows}")

    model = build_multi_adapter_model()
    device = next(model.parameters()).device

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    print("Starting dynamic unlearning optimization...")
    print(f"Epochs={EPOCHS}, batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS}")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        for idx in range(total_rows):
            original_text = str(original_ds[idx]["text"])
            translated_text = str(translated_ds[idx]["translated_text"])

            # Batch size is fixed to 1 for VRAM safety.
            x_original = tokenize_single(tokenizer, original_text, device)
            x_translated = tokenize_single(tokenizer, translated_text, device)

            # Step A: baseline logits on translated input with adapters disabled.
            with torch.no_grad():
                with model.disable_adapter():
                    out_base = model(**x_translated)
                    v_baseline = out_base.logits.detach()

            # Step B: reinforced logits on original input with reinforced adapter.
            with torch.no_grad():
                model.set_adapter("reinforced")
                out_reinf = model(**x_original)
                v_reinforced = out_reinf.logits.detach()

            # Step C: dynamic generic target.
            v_generic = v_baseline - ALPHA * torch.relu(v_reinforced - v_baseline)
            p_generic = torch.softmax(v_generic, dim=-1)

            # Step D: unlearn update on translated input with gradients enabled.
            model.set_adapter("unlearn")
            out_unlearn = model(**x_translated)
            logits = out_unlearn.logits
            vocab_size = logits.shape[-1]

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                p_generic.view(-1, vocab_size),
            )
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            global_step += 1
            print(f"step={global_step}/{total_rows} loss={loss.item() * GRAD_ACCUM_STEPS:.6f}")

            should_step = ((idx + 1) % GRAD_ACCUM_STEPS == 0) or (idx + 1 == total_rows)
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            del x_original
            del x_translated
            del out_base
            del v_baseline
            del out_reinf
            del v_reinforced
            del v_generic
            del p_generic
            del out_unlearn
            del logits

    print(f"Saving final unlearned adapter to: {OUTPUT_ADAPTER_DIR}")
    model.set_adapter("unlearn")
    model.save_pretrained(OUTPUT_ADAPTER_DIR, selected_adapters=["unlearn"])
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)
    print("Dynamic unlearning optimization completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            print("OOM ERROR: CUDA out of memory during dynamic unlearning.")
        else:
            print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
