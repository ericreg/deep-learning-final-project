"""Reinforcement and unlearning training loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DemoConfig
from .data import load_prepared
from .device import empty_accelerator_cache, require_accelerator
from .imports import require
from .modeling import add_trainable_lora, build_tokenizer, load_base_model, lora_config
from .token_alignment import align_translated_to_original
from .torch_ops import build_generic_logits, masked_soft_cross_entropy


def tokenize_texts(tokenizer, texts: list[str], config: DemoConfig):
    return tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
    )


def train_reinforced(config: DemoConfig) -> Path:
    transformers = require("transformers", "Install with: uv pip install -e .")
    require("torch", "Install a PyTorch build compatible with your accelerator.")

    require_accelerator()

    payload = load_prepared(config)
    tokenizer = build_tokenizer(config)
    model = add_trainable_lora(load_base_model(config, trainable=True), config)
    model.train()

    dataset = TextDataset(payload["forget_texts"], tokenizer, config)
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = transformers.TrainingArguments(
        output_dir=str(config.output_dir / "reinforced_checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        gradient_checkpointing=True,
        learning_rate=config.reinforce_lr,
        fp16=config.dtype.lower() in {"float16", "fp16", "half"},
        bf16=config.dtype.lower() in {"bfloat16", "bf16"},
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit" if config.load_in_4bit else "adamw_torch",
    )
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    config.reinforced_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.reinforced_adapter_dir)
    tokenizer.save_pretrained(config.reinforced_adapter_dir)
    return config.reinforced_adapter_dir


def train_unlearn(config: DemoConfig) -> Path:
    torch = require("torch", "Install a PyTorch build compatible with your accelerator.")
    peft = require("peft", "Install with: uv pip install -e .")

    require_accelerator()
    if not config.reinforced_adapter_dir.exists():
        raise FileNotFoundError(f"Reinforced adapter not found: {config.reinforced_adapter_dir}")

    payload = load_prepared(config)
    tokenizer = build_tokenizer(config)

    base_model = load_base_model(config, trainable=True)
    if config.load_in_4bit:
        base_model = peft.prepare_model_for_kbit_training(base_model)
    model = peft.PeftModel.from_pretrained(
        base_model,
        str(config.reinforced_adapter_dir),
        adapter_name="reinforced",
        is_trainable=False,
    )
    model.add_adapter("unlearn", lora_config(config))
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name and ".unlearn." in name
    model.set_adapter("unlearn")
    model.train()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.unlearn_lr)
    device = next(model.parameters()).device
    forget_texts = payload["forget_texts"]
    translated_texts = payload["translated_forget_texts"]

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for idx, (original, translated) in enumerate(zip(forget_texts, translated_texts)):
        original_batch = move_to_device(tokenize_texts(tokenizer, [original], config), device)
        translated_batch = move_to_device(tokenize_texts(tokenizer, [translated], config), device)

        with torch.no_grad():
            with model.disable_adapter():
                translated_logits = model(**translated_batch).logits.detach()
            model.set_adapter("reinforced")
            reinforced_logits = model(**original_batch).logits.detach()

        baseline_logits = align_logits_to_original(
            translated_logits,
            original_batch["input_ids"],
            original_batch["attention_mask"],
            translated_batch["input_ids"],
            translated_batch["attention_mask"],
            device,
        )
        generic = build_generic_logits(baseline_logits, reinforced_logits, config.alpha)
        model.set_adapter("unlearn")
        out = model(**original_batch)
        loss = masked_soft_cross_entropy(out.logits, generic, original_batch["attention_mask"])
        (loss / config.grad_accum_steps).backward()
        global_step += 1

        if global_step % config.grad_accum_steps == 0 or idx == len(forget_texts) - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            empty_accelerator_cache()

    config.unlearn_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.set_adapter("unlearn")
    model.save_pretrained(config.unlearn_adapter_dir, selected_adapters=["unlearn"])
    tokenizer.save_pretrained(config.unlearn_adapter_dir)
    return config.unlearn_adapter_dir


def move_to_device(batch: dict[str, Any], device):
    return {key: value.to(device) for key, value in batch.items()}


def align_logits_to_original(
    translated_logits,
    original_input_ids,
    original_attention_mask,
    translated_input_ids,
    translated_attention_mask,
    device,
):
    torch = require("torch")
    seq_len = original_input_ids.shape[1]
    original_len = int(original_attention_mask[0].sum().item())
    translated_len = int(translated_attention_mask[0].sum().item())
    if original_len <= 0 or translated_len <= 0:
        raise ValueError("Cannot align empty token sequences.")

    original_ids = original_input_ids[0, :original_len].detach().cpu().tolist()
    translated_ids = translated_input_ids[0, :translated_len].detach().cpu().tolist()
    mapping = align_translated_to_original(original_ids, translated_ids)
    pad_index = min(translated_len - 1, translated_logits.shape[1] - 1)
    full_mapping = mapping + [pad_index] * (seq_len - len(mapping))
    indices = torch.tensor(full_mapping, dtype=torch.long, device=device)
    return translated_logits.index_select(1, indices)


class TextDataset:
    def __init__(self, texts: list[str], tokenizer, config: DemoConfig) -> None:
        enc = tokenize_texts(tokenizer, texts, config)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ids = self.input_ids[idx]
        return {
            "input_ids": ids,
            "attention_mask": self.attention_mask[idx],
            "labels": ids.clone(),
        }
