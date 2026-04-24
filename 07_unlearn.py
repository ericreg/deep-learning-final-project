#!/usr/bin/env python3
"""Phase 3 unlearning loop: fit a fresh LoRA adapter to cached generic logits."""

import sys
import traceback
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SUBSETS = ["wmdp-bio", "wmdp-cyber"]
GENERIC_LABELS_PATH = "generic_labels.pt"
OUTPUT_ADAPTER_DIR = "./unlearned_adapter"
NUM_SEQUENCES = 10
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5


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


def load_wmdp_texts(limit: int) -> List[str]:
    subset_datasets: List[Dataset] = []
    for subset in SUBSETS:
        print(f"Loading dataset subset: {subset}")
        ds_dict = load_dataset("cais/wmdp", subset)
        ds = get_preferred_split(ds_dict)
        ds = ds.map(lambda ex: {"text": pick_text_field(ex)})
        subset_datasets.append(ds)

    merged = concatenate_datasets(subset_datasets)
    merged = merged.select(range(min(limit, len(merged))))
    print(f"Loaded {len(merged)} original sequences for unlearning.")
    return [str(x) for x in merged["text"]]


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        candidate_pad = "<|finetune_right_pad_id|>"
        if candidate_pad in tokenizer.get_vocab():
            tokenizer.pad_token = candidate_pad
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
    model.train()
    model.print_trainable_parameters()
    return model


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Unlearning requires an NVIDIA GPU.")

    print(f"Loading cached generic logits from: {GENERIC_LABELS_PATH}")
    payload = torch.load(GENERIC_LABELS_PATH, map_location="cpu")
    if not isinstance(payload, dict) or "v_generic" not in payload:
        raise ValueError("generic_labels.pt must be a dict containing key 'v_generic'.")

    v_generic = payload["v_generic"].to(dtype=torch.float32)
    if v_generic.ndim != 3:
        raise ValueError(f"Expected v_generic with shape [B, S, V], got {tuple(v_generic.shape)}")

    num_sequences, seq_len, vocab_size = v_generic.shape
    if num_sequences < NUM_SEQUENCES:
        raise ValueError(
            f"generic_labels.pt has only {num_sequences} sequences; expected at least {NUM_SEQUENCES}."
        )

    v_generic = v_generic[:NUM_SEQUENCES]

    tokenizer = build_tokenizer()
    texts = load_wmdp_texts(NUM_SEQUENCES)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=seq_len,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    model = build_model()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(
        "Starting unlearning optimization with cross-entropy to target probabilities "
        f"for {EPOCHS} epochs..."
    )

    global_step = 0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        for start in range(0, NUM_SEQUENCES, BATCH_SIZE):
            end = min(start + BATCH_SIZE, NUM_SEQUENCES)

            batch_input_ids = input_ids[start:end].to(device)
            batch_attention_mask = attention_mask[start:end].to(device)
            target_logits = v_generic[start:end].to(device)

            # Convert cached v_generic logits to target probabilities for this batch.
            target_probs = torch.softmax(target_logits, dim=-1)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            model_logits = outputs.logits

            if model_logits.shape[-1] != vocab_size:
                raise ValueError(
                    f"Vocab mismatch: model={model_logits.shape[-1]} vs target={vocab_size}"
                )

            loss = F.cross_entropy(
                model_logits.view(-1, vocab_size),
                target_probs.view(-1, vocab_size),
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            print(
                f"step={global_step} epoch={epoch + 1} "
                f"batch={start // BATCH_SIZE + 1} loss={loss.item():.6f}"
            )

            del batch_input_ids
            del batch_attention_mask
            del target_logits
            del target_probs
            del outputs
            del model_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Saving unlearned adapter to: {OUTPUT_ADAPTER_DIR}")
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)
    print("Unlearning optimization completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            print("OOM ERROR: CUDA out of memory during unlearning optimization.")
        else:
            print(f"RUNTIME ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
