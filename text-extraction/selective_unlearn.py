import json
import math
import random
import shutil
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import torch
import torch.nn.functional as F


DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DEFAULT_RESULTS_DIR = Path("results") / "hp"
DEFAULT_CHUNKS_FILE = DEFAULT_RESULTS_DIR / "chunks.json"
DEFAULT_SURPRISAL_RESULTS = DEFAULT_RESULTS_DIR / "surprisal" / "surprisal_results_BAAI_bge-base-en-v1.5.json"
DEFAULT_EMOTION_RESULTS = (
    DEFAULT_RESULTS_DIR / "emotion" / "volatility_results_j-hartmann_emotion-english-distilroberta-base.json"
)
DEFAULT_RANDOM_RESULTS = DEFAULT_RESULTS_DIR / "random" / "random_results.json"
DEFAULT_OUTPUT_DIR = Path("selective_unlearning_runs")
DEFAULT_CONTROL_TEXT = (
    "The mitochondria is the powerhouse of the cell and produces ATP through "
    "oxidative phosphorylation. Water boils at one hundred degrees Celsius at "
    "standard atmospheric pressure. The French Revolution began in 1789 and "
    "fundamentally transformed European political structures."
)

FIXED_SAMPLE_PROMPTS = [
    "Harry looked at Hermione and said",
    "Dumbledore raised his wand and",
    "At Hogwarts, the students were",
]

@dataclass(frozen=True)
class CorpusChunk:
    index: int
    text: str


@dataclass(frozen=True)
class ResultEntry:
    source: str
    source_path: str
    metric: str
    rank: int | None
    source_chunk_index: int | None
    score: float | None
    chunk: str


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)
        f.write("\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def model_load_error_message(model_id: str, artifact: str, exc: Exception) -> str:
    details = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    message = [f"Could not load {artifact} for {model_id}."]

    error_text = str(exc).lower()
    if "not in the authorized list" in error_text or "403" in error_text:
        message.extend(
            [
                "The Hugging Face token is valid, but this account is not authorized for the gated model.",
                f"Request/accept access on https://huggingface.co/{model_id}, then rerun this command.",
            ]
        )
    elif any(marker in error_text for marker in ("gated repo", "401", "unauthorized", "restricted")):
        message.extend(
            [
                "This Hugging Face repo is gated/restricted.",
                "Make sure your Hugging Face account has access to the model, then run:",
                "  uv run hf auth login",
                "You can verify the token with:",
                "  uv run hf auth whoami",
            ]
        )
    else:
        message.append("Check Hugging Face access, network/cache state, or pass --model-id pointing to a local model path.")

    message.append(f"Original error: {details}")
    return "\n".join(message)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as exc:
        raise click.ClickException(model_load_error_message(model_id, "tokenizer", exc)) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_chunk_manifest(path: Path) -> list[CorpusChunk]:
    if not path.exists():
        raise click.ClickException(f"Missing chunk manifest: {path}. Run text_extract.py first, then retry.")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Could not parse chunk manifest at {path}: {exc}") from exc

    raw_chunks = data.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise click.ClickException(f"Chunk manifest {path} has no usable 'chunks' entries. Run text_extract.py first.")

    chunks = []
    for position, raw in enumerate(raw_chunks):
        if isinstance(raw, str):
            chunk_index = position
            chunk_text = raw
        elif isinstance(raw, dict):
            chunk_index = raw.get("chunk_index", position)
            chunk_text = raw.get("chunk")
        else:
            raise click.ClickException(f"Chunk manifest {path} contains an invalid chunk entry at position {position}.")

        if not isinstance(chunk_index, int) or not isinstance(chunk_text, str) or not chunk_text.strip():
            raise click.ClickException(f"Chunk manifest {path} contains an unusable chunk entry at position {position}.")
        chunks.append(CorpusChunk(index=chunk_index, text=chunk_text))

    return chunks


def load_control_texts(control_chunks_file: Path | None) -> list[str]:
    if control_chunks_file is None:
        return [DEFAULT_CONTROL_TEXT]
    return [chunk.text for chunk in load_chunk_manifest(control_chunks_file)]


def split_train_eval(
    chunks: Sequence[CorpusChunk],
    eval_holdout: float,
    seed: int,
) -> tuple[list[CorpusChunk], list[CorpusChunk]]:
    if not chunks:
        return [], []

    indexes = [chunk.index for chunk in chunks]
    random.Random(seed).shuffle(indexes)
    eval_count = int(round(len(indexes) * eval_holdout))
    if eval_holdout > 0 and eval_count == 0 and len(indexes) > 1:
        eval_count = 1
    eval_ids = set(indexes[:eval_count])

    train_chunks = [chunk for chunk in chunks if chunk.index not in eval_ids]
    eval_chunks = [chunk for chunk in chunks if chunk.index in eval_ids]
    return train_chunks, eval_chunks


def load_result_entries(path: Path, source: str, top_n: int) -> list[ResultEntry]:
    if not path.exists():
        raise click.ClickException(
            f"Missing {source} results file: {path}. Run text_extract.py first, then retry."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Could not parse {source} results JSON at {path}: {exc}") from exc

    top = data.get("top")
    if not isinstance(top, list) or not top:
        raise click.ClickException(
            f"{source} results file {path} has no usable 'top' entries. Run text_extract.py first."
        )

    metric = str(data.get("metric") or source)
    selected = []
    for raw in top[:top_n]:
        if not isinstance(raw, dict) or not isinstance(raw.get("chunk"), str) or not raw["chunk"].strip():
            raise click.ClickException(
                f"{source} results file {path} contains a top entry without a non-empty 'chunk'."
            )

        score = raw.get(metric)
        if score is None:
            score = raw.get("surprisal", raw.get("volatility", raw.get("random_score")))

        selected.append(
            ResultEntry(
                source=source,
                source_path=str(path),
                metric=metric,
                rank=raw.get("rank") if isinstance(raw.get("rank"), int) else None,
                source_chunk_index=raw.get("chunk_index") if isinstance(raw.get("chunk_index"), int) else None,
                score=float(score) if isinstance(score, (int, float)) else None,
                chunk=raw["chunk"],
            )
        )

    if not selected:
        raise click.ClickException(f"{source} results file {path} did not contain any selected chunks.")
    return selected


def load_selection_entries(
    selection: str,
    top_n: int,
    surprisal_results: Path,
    emotion_results: Path,
    random_results: Path,
) -> list[ResultEntry]:
    if selection == "full":
        return []
    if selection == "surprisal":
        return load_result_entries(surprisal_results, "surprisal", top_n)
    if selection == "emotion":
        return load_result_entries(emotion_results, "emotion", top_n)
    if selection == "random":
        return load_result_entries(random_results, "random", top_n)
    if selection != "combined":
        raise click.ClickException(f"Unsupported selection: {selection}")

    entries = load_result_entries(surprisal_results, "surprisal", top_n)
    entries.extend(load_result_entries(emotion_results, "emotion", top_n))
    return entries


def build_selected_manifest(
    selection: str,
    train_chunks: Sequence[CorpusChunk],
    result_entries: Sequence[ResultEntry],
    max_train_chunks: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if selection == "full":
        selected_chunks = list(train_chunks)
        if max_train_chunks is not None:
            selected_chunks = selected_chunks[:max_train_chunks]
        manifest = [
            {
                "chunk_index": chunk.index,
                "sources": [{"source": "full"}],
                "text": chunk.text,
            }
            for chunk in selected_chunks
        ]
        return manifest, []

    chunks_by_index = {chunk.index: chunk for chunk in train_chunks}
    chunks_by_text = {chunk.text: chunk for chunk in train_chunks}
    by_chunk_index: dict[int, dict[str, Any]] = {}
    skipped = []

    for entry in result_entries:
        chunk = chunks_by_index.get(entry.source_chunk_index) if entry.source_chunk_index is not None else None
        if chunk is None:
            chunk = chunks_by_text.get(entry.chunk)

        if chunk is None:
            skipped.append(
                {
                    "source": entry.source,
                    "rank": entry.rank,
                    "source_chunk_index": entry.source_chunk_index,
                    "reason": "source chunk is not present in the training split",
                    "chunk_preview": entry.chunk[:300],
                }
            )
            continue

        source_record = {
            "source": entry.source,
            "source_path": entry.source_path,
            "metric": entry.metric,
            "rank": entry.rank,
            "source_chunk_index": entry.source_chunk_index,
            "score": entry.score,
            "result_chunk_preview": entry.chunk[:500],
        }

        if chunk.index not in by_chunk_index:
            by_chunk_index[chunk.index] = {
                "chunk_index": chunk.index,
                "sources": [source_record],
                "text": chunk.text,
            }
        else:
            by_chunk_index[chunk.index]["sources"].append(source_record)

    manifest = sorted(by_chunk_index.values(), key=lambda item: item["chunk_index"])
    if max_train_chunks is not None:
        manifest = manifest[:max_train_chunks]
    return manifest, skipped


def import_training_deps():
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise click.ClickException(
            "Training dependencies are missing. Install/update the project dependencies "
            "so peft, accelerate, and transformers are available."
        ) from exc

    return AutoModelForCausalLM, LoraConfig, PeftModel, TaskType, get_peft_model


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise click.ClickException(f"Unsupported dtype: {dtype_name}")


def load_causal_lm(model_id: str, dtype_name: str):
    AutoModelForCausalLM, *_ = import_training_deps()
    dtype = torch_dtype_from_name(dtype_name)
    kwargs = {"device_map": "auto", "dtype": dtype}
    try:
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            kwargs.pop("dtype")
            kwargs["torch_dtype"] = dtype
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        raise click.ClickException(model_load_error_message(model_id, "model", exc)) from exc


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def maybe_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_optimizer(params: Sequence[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
    try:
        return torch.optim.AdamW(params, lr=lr, fused=torch.cuda.is_available())
    except TypeError:
        return torch.optim.AdamW(params, lr=lr)


def lora_config(lora_r: int, lora_alpha: int):
    _, LoraConfig, _, TaskType, _ = import_training_deps()
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def tokenize_batch(tokenizer, texts: Sequence[str], max_length: int, device: torch.device, labels: bool = False) -> dict:
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    if labels:
        label_ids = enc["input_ids"].clone()
        label_ids[enc["attention_mask"] == 0] = -100
        enc["labels"] = label_ids
    return enc


def batches(items: Sequence[str], batch_size: int) -> list[list[str]]:
    return [list(items[start : start + batch_size]) for start in range(0, len(items), batch_size)]


def train_reinforced_adapter(
    texts: Sequence[str],
    tokenizer,
    run_dir: Path,
    model_id: str,
    dtype_name: str,
    chunk_token_size: int,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
    lora_r: int,
    lora_alpha_value: int,
    logging_steps: int,
) -> tuple[Path, list[dict[str, float]]]:
    _, _, _, _, get_peft_model = import_training_deps()

    adapter_dir = run_dir / "reinforced_adapter"
    base = load_causal_lm(model_id, dtype_name)
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    model = get_peft_model(base, lora_config(lora_r, lora_alpha_value))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    optimizer = make_optimizer([p for p in model.parameters() if p.requires_grad], lr)
    optimizer.zero_grad(set_to_none=True)

    device = model_device(model)
    loss_history = []
    global_step = 0

    click.echo(f"Training reinforced adapter on {len(texts)} chunks.")
    for epoch in range(epochs):
        order = list(texts)
        random.Random(seed + epoch).shuffle(order)
        for batch in batches(order, batch_size):
            enc = tokenize_batch(tokenizer, batch, chunk_token_size, device, labels=True)
            loss = model(**enc).loss
            (loss / grad_accum).backward()

            global_step += 1
            loss_history.append({"step": global_step, "loss": float(loss.detach().cpu())})

            if global_step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                maybe_empty_cache()

            if logging_steps > 0 and global_step % logging_steps == 0:
                click.echo(f"  reinforced epoch {epoch + 1}/{epochs} step {global_step} loss={loss.item():.5f}")

            del enc, loss

    if global_step % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    dump_json(run_dir / "reinforced_loss_log.json", loss_history)

    del model, base, optimizer
    maybe_empty_cache()
    click.echo(f"Reinforced adapter saved to {adapter_dir}")
    return adapter_dir, loss_history


def masked_kl_loss(logits: torch.Tensor, target_probs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    per_token = F.kl_div(
        F.log_softmax(logits, dim=-1),
        target_probs,
        reduction="none",
    ).sum(dim=-1)
    mask = attention_mask.to(per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


def train_unlearn_adapter(
    texts: Sequence[str],
    tokenizer,
    run_dir: Path,
    reinforced_adapter_dir: Path,
    model_id: str,
    dtype_name: str,
    chunk_token_size: int,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
    alpha: float,
    lora_r: int,
    lora_alpha_value: int,
    logging_steps: int,
):
    _, _, PeftModel, _, _ = import_training_deps()

    unlearned_dir = run_dir / "unlearned_adapter"
    base = load_causal_lm(model_id, dtype_name)
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    model = PeftModel.from_pretrained(
        base,
        str(reinforced_adapter_dir),
        adapter_name="reinforced",
        is_trainable=False,
    )
    model.add_adapter("unlearn", lora_config(lora_r, lora_alpha_value))

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name and ".unlearn." in name

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.set_adapter("unlearn")
    model.train()

    optimizer = make_optimizer([p for p in model.parameters() if p.requires_grad], lr)
    optimizer.zero_grad(set_to_none=True)

    device = model_device(model)
    loss_history = []
    global_step = 0

    click.echo(f"Training unlearn adapter on {len(texts)} chunks with alpha={alpha}.")
    for epoch in range(epochs):
        order = list(texts)
        random.Random(seed + 10_000 + epoch).shuffle(order)
        for batch in batches(order, batch_size):
            enc = tokenize_batch(tokenizer, batch, chunk_token_size, device, labels=False)

            with torch.no_grad():
                with model.disable_adapter():
                    v_base = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
                model.set_adapter("reinforced")
                v_reinf = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits

            v_generic = v_base - alpha * torch.relu(v_reinf - v_base)
            p_target = torch.softmax(v_generic, dim=-1).detach()

            model.set_adapter("unlearn")
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            loss = masked_kl_loss(logits, p_target, enc["attention_mask"])
            (loss / grad_accum).backward()

            global_step += 1
            loss_history.append({"step": global_step, "loss": float(loss.detach().cpu())})

            if global_step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                maybe_empty_cache()

            if logging_steps > 0 and global_step % logging_steps == 0:
                click.echo(f"  unlearn epoch {epoch + 1}/{epochs} step {global_step} loss={loss.item():.5f}")

            del enc, v_base, v_reinf, v_generic, p_target, logits, loss

    if global_step % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.set_adapter("unlearn")
    model.save_pretrained(unlearned_dir, selected_adapters=["unlearn"])
    tokenizer.save_pretrained(unlearned_dir)
    dump_json(run_dir / "unlearn_loss_log.json", loss_history)

    del optimizer
    maybe_empty_cache()
    click.echo(f"Unlearned adapter saved to {unlearned_dir}")
    return unlearned_dir, model, loss_history


def adapter_context(model, adapter_name: str | None):
    if adapter_name is None:
        return model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    model.set_adapter(adapter_name)
    return nullcontext()


def model_loss(
    model,
    tokenizer,
    text: str,
    chunk_token_size: int,
    adapter_name: str | None,
) -> float:
    device = model_device(model)
    enc = tokenize_batch(tokenizer, [text], chunk_token_size, device, labels=True)
    with torch.no_grad():
        with adapter_context(model, adapter_name):
            loss = model(**enc).loss
    return float(loss.detach().cpu())


def compute_perplexity(
    model,
    tokenizer,
    texts: Sequence[str],
    chunk_token_size: int,
    adapter_name: str | None,
    limit: int,
) -> float | None:
    selected = list(texts[:limit])
    if not selected:
        return None
    losses = [model_loss(model, tokenizer, text, chunk_token_size, adapter_name) for text in selected]
    mean_loss = safe_mean(losses)
    return safe_exp(mean_loss) if mean_loss is not None else None


def compute_anchor_recall(
    model,
    tokenizer,
    eval_texts: Sequence[str],
    chunk_token_size: int,
    anchor_threshold: float,
    limit: int,
) -> dict[str, Any]:
    device = model_device(model)
    base_chunk_recalls = []
    unlearn_chunk_recalls = []
    total_anchors = 0
    evaluated_chunks = 0

    for text in eval_texts[:limit]:
        enc = tokenize_batch(tokenizer, [text], chunk_token_size, device, labels=False)
        seq_len = int(enc["attention_mask"][0].sum().item())
        if seq_len < 2:
            continue

        input_ids = enc["input_ids"][:, :seq_len]
        attention_mask = enc["attention_mask"][:, :seq_len]
        true_next = input_ids[0, 1:]
        positions = torch.arange(seq_len - 1, device=device)

        with torch.no_grad():
            with model.disable_adapter():
                base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, :-1]
            model.set_adapter("reinforced")
            reinforced_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, :-1]
            model.set_adapter("unlearn")
            unlearn_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, :-1]

            p_base = torch.softmax(base_logits, dim=-1)
            p_reinf = torch.softmax(reinforced_logits, dim=-1)
            ratio = p_reinf[positions, true_next] / (p_base[positions, true_next] + 1e-9)
            is_anchor = ratio >= anchor_threshold

            anchor_positions = is_anchor.nonzero(as_tuple=True)[0]
            if len(anchor_positions) > 0:
                base_preds = base_logits.argmax(dim=-1)
                unlearn_preds = unlearn_logits.argmax(dim=-1)
                base_hits = (base_preds[anchor_positions] == true_next[anchor_positions]).sum().item()
                unlearn_hits = (unlearn_preds[anchor_positions] == true_next[anchor_positions]).sum().item()
                base_chunk_recalls.append(base_hits / len(anchor_positions))
                unlearn_chunk_recalls.append(unlearn_hits / len(anchor_positions))
                total_anchors += len(anchor_positions)

        evaluated_chunks += 1
        del enc, input_ids, attention_mask, base_logits, reinforced_logits, unlearn_logits
        maybe_empty_cache()

    return {
        "evaluated_chunks": evaluated_chunks,
        "total_anchor_tokens": total_anchors,
        "base_anchor_recall": safe_mean(base_chunk_recalls),
        "unlearn_anchor_recall": safe_mean(unlearn_chunk_recalls),
    }


def generate_samples(
    model,
    tokenizer,
    prompts: Sequence[str],
    chunk_token_size: int,
    adapter_name: str | None,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    device = model_device(model)
    samples = []
    model.eval()
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=chunk_token_size)
        enc = {key: value.to(device) for key, value in enc.items()}
        with torch.no_grad():
            with adapter_context(model, adapter_name):
                generated = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        samples.append({"prompt": prompt, "completion": tokenizer.decode(generated[0], skip_special_tokens=True)})
    return samples


def corpus_run_name(results_dir: Path) -> str:
    """Return the corpus folder name to use for run artifacts."""
    name = results_dir.name
    if name == "iliad":
        return "iliad"
    return name


def make_run_dir(output_dir: Path, results_dir: Path, selection: str) -> Path:
    run_dir = output_dir / corpus_run_name(results_dir) / selection
    if run_dir.exists() or run_dir.is_symlink():
        if run_dir.is_dir() and not run_dir.is_symlink():
            shutil.rmtree(run_dir)
        else:
            run_dir.unlink()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_summary(metrics: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "selection": metrics["selection"],
        "dry_run": metrics["dry_run"],
        "selected_forget_chunks": metrics["selected_forget_chunks"],
        "train_chunk_count": metrics["train_chunk_count"],
        "eval_chunk_count": metrics["eval_chunk_count"],
        "source_result_files": metrics["source_result_files"],
        "base_hp_perplexity": metrics.get("base_hp_perplexity"),
        "unlearn_hp_perplexity": metrics.get("unlearn_hp_perplexity"),
        "base_control_perplexity": metrics.get("base_control_perplexity"),
        "unlearn_control_perplexity": metrics.get("unlearn_control_perplexity"),
        "base_anchor_recall": metrics.get("anchor_recall", {}).get("base_anchor_recall"),
        "unlearn_anchor_recall": metrics.get("anchor_recall", {}).get("unlearn_anchor_recall"),
    }


@click.command()
@click.option(
    "--chunks-file",
    default=DEFAULT_CHUNKS_FILE,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Chunk manifest written by text_extract.py.",
)
@click.option(
    "--control-chunks-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional control chunk manifest. Uses a built-in control text if omitted.",
)
@click.option("--model-id", default=DEFAULT_MODEL_ID, show_default=True)
@click.option(
    "--selection",
    default="full",
    show_default=True,
    type=click.Choice(["full", "surprisal", "emotion", "combined", "random"]),
)
@click.option("--top-n", default=50, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--results-dir",
    default=DEFAULT_RESULTS_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--surprisal-results",
    default=DEFAULT_SURPRISAL_RESULTS,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--emotion-results",
    default=DEFAULT_EMOTION_RESULTS,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--random-results",
    default=DEFAULT_RANDOM_RESULTS,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--dry-run", is_flag=True, help="Prepare chunks and manifests, then exit before model training.")
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--chunk-token-size", default=512, show_default=True, type=click.IntRange(min=32))
@click.option("--eval-holdout", default=0.1, show_default=True, type=click.FloatRange(min=0.0, max=0.9))
@click.option("--max-train-chunks", default=None, type=click.IntRange(min=1))
@click.option("--dtype", default="bfloat16", show_default=True, type=click.Choice(["bfloat16", "float16", "float32"]))
@click.option("--reinforced-epochs", default=1, show_default=True, type=click.IntRange(min=1))
@click.option("--unlearn-epochs", default=3, show_default=True, type=click.IntRange(min=1))
@click.option("--reinforced-batch-size", default=8, show_default=True, type=click.IntRange(min=1))
@click.option("--unlearn-batch-size", default=1, show_default=True, type=click.IntRange(min=1))
@click.option("--reinforced-grad-accum", default=2, show_default=True, type=click.IntRange(min=1))
@click.option("--unlearn-grad-accum", default=4, show_default=True, type=click.IntRange(min=1))
@click.option("--reinforced-lr", default=2e-4, show_default=True, type=float)
@click.option("--unlearn-lr", default=2e-5, show_default=True, type=float)
@click.option("--alpha", default=5.0, show_default=True, type=float)
@click.option("--anchor-threshold", default=2.0, show_default=True, type=float)
@click.option("--lora-r", default=8, show_default=True, type=click.IntRange(min=1))
@click.option("--lora-alpha", "lora_alpha_value", default=16, show_default=True, type=click.IntRange(min=1))
@click.option("--eval-chunks", default=50, show_default=True, type=click.IntRange(min=1))
@click.option("--ppl-chunks", default=10, show_default=True, type=click.IntRange(min=1))
@click.option("--logging-steps", default=10, show_default=True, type=click.IntRange(min=0))
@click.option("--generate-samples/--no-generate-samples", default=False, show_default=True)
@click.option("--sample-max-new-tokens", default=80, show_default=True, type=click.IntRange(min=1))
def cli(**kwargs):
    """Run selective unlearning from precomputed text-extraction results."""
    started = time.perf_counter()
    set_seed(kwargs["seed"])

    if kwargs["results_dir"] != DEFAULT_RESULTS_DIR:
        if kwargs["chunks_file"] == DEFAULT_CHUNKS_FILE:
            kwargs["chunks_file"] = kwargs["results_dir"] / DEFAULT_CHUNKS_FILE.name
        if kwargs["surprisal_results"] == DEFAULT_SURPRISAL_RESULTS:
            kwargs["surprisal_results"] = kwargs["results_dir"] / "surprisal" / DEFAULT_SURPRISAL_RESULTS.name
        if kwargs["emotion_results"] == DEFAULT_EMOTION_RESULTS:
            kwargs["emotion_results"] = kwargs["results_dir"] / "emotion" / DEFAULT_EMOTION_RESULTS.name
        if kwargs["random_results"] == DEFAULT_RANDOM_RESULTS:
            kwargs["random_results"] = kwargs["results_dir"] / "random" / DEFAULT_RANDOM_RESULTS.name

    selection = kwargs["selection"]
    hp_chunks = load_chunk_manifest(kwargs["chunks_file"])
    train_chunks, eval_chunks = split_train_eval(hp_chunks, kwargs["eval_holdout"], kwargs["seed"])
    if not train_chunks:
        raise click.ClickException("No training chunks were loaded from the chunk manifest.")

    result_entries = load_selection_entries(
        selection,
        kwargs["top_n"],
        kwargs["surprisal_results"],
        kwargs["emotion_results"],
        kwargs["random_results"],
    )
    source_files = sorted({entry.source_path for entry in result_entries})
    control_texts = load_control_texts(kwargs["control_chunks_file"])

    selected_manifest, skipped_matches = build_selected_manifest(
        selection,
        train_chunks,
        result_entries,
        kwargs["max_train_chunks"],
    )
    if not selected_manifest:
        raise click.ClickException(
            "No forget chunks were selected from the extracted chunk manifest. "
            "Rerun text_extract.py and make sure the analysis JSONs match the same chunks.json file."
        )

    run_dir = make_run_dir(kwargs["output_dir"], kwargs["results_dir"], selection)
    config = dict(kwargs)
    config["run_dir"] = run_dir
    config["source_result_files"] = source_files
    dump_json(run_dir / "config.json", config)

    click.echo(f"Run directory: {run_dir}")
    selected_payload = {
        "selection": selection,
        "top_n": kwargs["top_n"],
        "chunks_file": str(kwargs["chunks_file"]),
        "source_result_files": source_files,
        "total_target_chunks": len(hp_chunks),
        "total_hp_chunks": len(hp_chunks),
        "train_chunk_count": len(train_chunks),
        "eval_chunk_count": len(eval_chunks),
        "selected_forget_chunks": len(selected_manifest),
        "skipped_selection_count": len(skipped_matches),
        "skipped_selection": skipped_matches,
        "chunks": selected_manifest,
    }
    dump_json(run_dir / "selected_chunks.json", selected_payload)

    metrics: dict[str, Any] = {
        "selection": selection,
        "dry_run": kwargs["dry_run"],
        "run_dir": str(run_dir),
        "chunks_file": str(kwargs["chunks_file"]),
        "source_result_files": source_files,
        "total_target_chunks": len(hp_chunks),
        "total_hp_chunks": len(hp_chunks),
        "train_chunk_count": len(train_chunks),
        "eval_chunk_count": len(eval_chunks),
        "control_chunk_count": len(control_texts),
        "selected_forget_chunks": len(selected_manifest),
        "skipped_selection_count": len(skipped_matches),
        "elapsed_seconds": time.perf_counter() - started,
    }

    click.echo(
        f"Selected {len(selected_manifest)} forget chunks "
        f"from {len(train_chunks)} train chunks; held out {len(eval_chunks)} target eval chunks."
    )

    if kwargs["dry_run"]:
        dump_json(run_dir / "metrics.json", metrics)
        dump_json(run_dir / "summary.json", build_summary(metrics, run_dir))
        click.echo("Dry run complete. No adapters were trained.")
        return

    if not torch.cuda.is_available():
        raise click.ClickException("Training requires a CUDA/ROCm GPU. Use --dry-run for chunk validation on CPU.")

    click.echo(f"Loading tokenizer: {kwargs['model_id']}")
    tokenizer = load_tokenizer(kwargs["model_id"])

    train_texts = [item["text"] for item in selected_manifest]
    train_started = time.perf_counter()
    reinforced_dir, reinforced_losses = train_reinforced_adapter(
        train_texts,
        tokenizer,
        run_dir,
        kwargs["model_id"],
        kwargs["dtype"],
        kwargs["chunk_token_size"],
        kwargs["reinforced_epochs"],
        kwargs["reinforced_batch_size"],
        kwargs["reinforced_grad_accum"],
        kwargs["reinforced_lr"],
        kwargs["seed"],
        kwargs["lora_r"],
        kwargs["lora_alpha_value"],
        kwargs["logging_steps"],
    )
    unlearned_dir, model, unlearn_losses = train_unlearn_adapter(
        train_texts,
        tokenizer,
        run_dir,
        reinforced_dir,
        kwargs["model_id"],
        kwargs["dtype"],
        kwargs["chunk_token_size"],
        kwargs["unlearn_epochs"],
        kwargs["unlearn_batch_size"],
        kwargs["unlearn_grad_accum"],
        kwargs["unlearn_lr"],
        kwargs["seed"],
        kwargs["alpha"],
        kwargs["lora_r"],
        kwargs["lora_alpha_value"],
        kwargs["logging_steps"],
    )
    training_seconds = time.perf_counter() - train_started

    model.eval()
    eval_texts = [chunk.text for chunk in eval_chunks]
    metrics.update(
        {
            "reinforced_adapter_dir": str(reinforced_dir),
            "unlearned_adapter_dir": str(unlearned_dir),
            "reinforced_loss_last": reinforced_losses[-1]["loss"] if reinforced_losses else None,
            "unlearn_loss_last": unlearn_losses[-1]["loss"] if unlearn_losses else None,
            "training_seconds": training_seconds,
            "base_hp_perplexity": compute_perplexity(
                model,
                tokenizer,
                eval_texts,
                kwargs["chunk_token_size"],
                None,
                kwargs["ppl_chunks"],
            ),
            "unlearn_hp_perplexity": compute_perplexity(
                model,
                tokenizer,
                eval_texts,
                kwargs["chunk_token_size"],
                "unlearn",
                kwargs["ppl_chunks"],
            ),
            "base_control_perplexity": compute_perplexity(
                model,
                tokenizer,
                control_texts,
                kwargs["chunk_token_size"],
                None,
                kwargs["ppl_chunks"],
            ),
            "unlearn_control_perplexity": compute_perplexity(
                model,
                tokenizer,
                control_texts,
                kwargs["chunk_token_size"],
                "unlearn",
                kwargs["ppl_chunks"],
            ),
            "anchor_recall": compute_anchor_recall(
                model,
                tokenizer,
                eval_texts,
                kwargs["chunk_token_size"],
                kwargs["anchor_threshold"],
                kwargs["eval_chunks"],
            ),
        }
    )

    if kwargs["generate_samples"]:
        metrics["samples"] = {
            "base": generate_samples(
                model,
                tokenizer,
                FIXED_SAMPLE_PROMPTS,
                kwargs["chunk_token_size"],
                None,
                kwargs["sample_max_new_tokens"],
            ),
            "unlearn": generate_samples(
                model,
                tokenizer,
                FIXED_SAMPLE_PROMPTS,
                kwargs["chunk_token_size"],
                "unlearn",
                kwargs["sample_max_new_tokens"],
            ),
        }

    metrics["elapsed_seconds"] = time.perf_counter() - started
    dump_json(run_dir / "metrics.json", metrics)
    dump_json(run_dir / "summary.json", build_summary(metrics, run_dir))
    click.echo(f"Run complete. Summary written to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    cli()
