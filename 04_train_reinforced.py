#!/usr/bin/env python3
"""Train a reinforced LoRA adapter on WMDP with strict 24GB VRAM constraints."""

import sys
import traceback
from typing import Dict, List

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SUBSETS = ["wmdp-bio", "wmdp-cyber"]
MAX_LENGTH = 512
ADAPTER_OUT_DIR = "./reinforced_adapter"


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


def load_wmdp_dataset() -> Dataset:
    subset_datasets: List[Dataset] = []
    for subset in SUBSETS:
        print(f"Loading dataset subset: {subset}")
        ds_dict = load_dataset("cais/wmdp", subset)
        ds = get_preferred_split(ds_dict)
        ds = ds.map(lambda ex: {"text": pick_text_field(ex)})
        subset_datasets.append(ds)

    merged = concatenate_datasets(subset_datasets)
    print(f"Merged dataset size: {len(merged)} rows")
    return merged


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        candidate_pad = "<|finetune_right_pad_id|>"
        if candidate_pad in tokenizer.get_vocab():
            tokenizer.pad_token = candidate_pad
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_for_clm(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _tokenize(batch: Dict) -> Dict:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = ds.map(_tokenize, batched=True, remove_columns=ds.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def build_model() -> torch.nn.Module:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading base model in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Reinforced training requires an NVIDIA GPU.")

    tokenizer = build_tokenizer()
    ds = load_wmdp_dataset()
    tokenized_ds = tokenize_for_clm(ds, tokenizer)

    model = build_model()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="./reinforced_adapter_checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    print("Starting reinforced LoRA training (1 epoch)...")
    trainer.train()

    print(f"Saving reinforced adapter to: {ADAPTER_OUT_DIR}")
    model.save_pretrained(ADAPTER_OUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUT_DIR)
    print("Reinforced adapter training finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            print("OOM ERROR: CUDA out of memory during reinforced adapter training.")
        else:
            print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
