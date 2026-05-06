#!/usr/bin/env python3
"""Fine-tune a LoRA adapter on the HP corpus to amplify book-specific knowledge."""

import sys
import traceback
from typing import Dict

import torch
from datasets import load_from_disk
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
CORPUS_DIR = "./hp_corpus"
ADAPTER_OUT_DIR = "./hp_reinforced_adapter"
MAX_LENGTH = 512


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(ds, tokenizer: AutoTokenizer):
    def _tok(batch: Dict) -> Dict:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def build_model() -> torch.nn.Module:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
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
        raise RuntimeError("CUDA required for reinforced training.")

    tokenizer = build_tokenizer()
    ds = load_from_disk(CORPUS_DIR)
    print(f"Loaded corpus: {len(ds)} chunks")

    tokenized_ds = tokenize_dataset(ds, tokenizer)
    model = build_model()

    args = TrainingArguments(
        output_dir="./hp_reinforced_checkpoints",
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
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Training reinforced adapter on HP corpus...")
    trainer.train()

    model.save_pretrained(ADAPTER_OUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUT_DIR)
    print(f"Reinforced adapter saved to: {ADAPTER_OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        traceback.print_exc()
        sys.exit(1)
