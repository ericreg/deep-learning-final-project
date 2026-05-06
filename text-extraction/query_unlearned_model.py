import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import torch

from selective_unlearn import dump_json, import_training_deps, load_causal_lm, load_tokenizer, model_device


# Choose which built-in question set to use when no --question or
# --questions-file is supplied. Use "auto" to infer it from the run config/path.
BUILTIN_QUESTION_SET = "auto"
DEFAULT_QUESTION_SETS = {
    "hp": [
        ["What company did Mr. Dursley direct?", "Grunnings, a firm that made drills."],
        ["Where did Harry sleep at the Dursleys' house?", "In the cupboard under the stairs."],
        ["Who brought baby Harry to Privet Drive on a motorcycle?", "Hagrid."],
        ["Who lent Hagrid the motorcycle?", "Sirius Black."],
        ["What does the Mirror of Erised show?", "The deepest, most desperate desire of a person's heart."],
    ],
    "bible": [
        ["What happened after God said, Let there be light?", "There was light."],
        ["What did God command Noah to build the ark out of?", "Gopher wood, with rooms and pitch inside and outside."],
        ["What name did God tell Moses to give the children of Israel?", "I AM hath sent me unto you."],
        ["How did David defeat the Philistine?", "David used a sling and a stone, and had no sword in his hand."],
        ["How does Psalm 23 describe the LORD?", "The LORD is David's shepherd, and David says he shall not want."],
    ],
    "iliad": [
        ["Which two captives were allotted to Agamemnon and Achilles?", "Chryseïs was allotted to Agamemnon, and Briseïs to Achilles."],
        ["Why did Achilles withdraw from the rest of the Greeks?", "Agamemnon seized Briseïs in revenge, and Achilles withdrew in discontent."],
        ["What emotion is described as bringing trouble to Greece?", "Achilles' wrath."],
        ["Whose death makes Achilles hate to live?", "Patroclus' death."],
        ["What does Priam ask Achilles to return?", "Hector's body."],
    ],
}

QUESTION_SET_ALIASES = {
    "iliad": "iliad",
    "iliad": "iliad",
}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Could not parse JSON file {path}: {exc}") from exc


def load_run_config(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}

    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}

    data = read_json(config_path)
    if not isinstance(data, dict):
        raise click.ClickException(f"Run config {config_path} must contain a JSON object.")
    return data


def infer_corpus_name(run_dir: Path | None, config: dict[str, Any], adapter_dir: Path | None) -> str:
    for key in ("results_dir", "chunks_file"):
        value = config.get(key)
        if value:
            path = Path(str(value))
            return path.parent.name if key == "chunks_file" else path.name

    if run_dir is not None:
        selected_chunks = run_dir / "selected_chunks.json"
        if selected_chunks.exists():
            data = read_json(selected_chunks)
            chunks_file = data.get("chunks_file") if isinstance(data, dict) else None
            if chunks_file:
                return Path(str(chunks_file)).parent.name

    if adapter_dir is not None:
        return adapter_dir.parent.name

    return "generic"


def resolve_builtin_question_set(corpus_name: str) -> str:
    selected_question_set = BUILTIN_QUESTION_SET.lower().strip()
    if selected_question_set == "auto":
        selected_question_set = corpus_name.lower()
    selected_question_set = QUESTION_SET_ALIASES.get(selected_question_set, selected_question_set)

    if selected_question_set not in DEFAULT_QUESTION_SETS:
        valid_question_sets = ", ".join(sorted(DEFAULT_QUESTION_SETS))
        raise click.ClickException(
            f"Unknown built-in question set '{selected_question_set}'. "
            f"Set BUILTIN_QUESTION_SET to 'auto' or one of: {valid_question_sets}."
        )

    return selected_question_set


def normalize_question_item(item: Any, question_id: str) -> dict[str, str]:
    if isinstance(item, str):
        question = item
        expected_answer = None
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        question, expected_answer = item
    elif isinstance(item, dict):
        question = item.get("question", item.get("prompt"))
        expected_answer = item.get("answer", item.get("expected_answer", item.get("completion")))
        question_id = str(item.get("id", question_id))
    else:
        raise click.ClickException(
            "Questions must be strings, [prompt, answer] pairs, or objects with question/prompt fields."
        )

    if not isinstance(question, str) or not question.strip():
        raise click.ClickException(f"Question entry {question_id} is empty or invalid.")

    normalized = {"id": question_id, "question": question.strip()}
    if expected_answer is not None:
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise click.ClickException(f"Expected answer for {question_id} is empty or invalid.")
        normalized["expected_answer"] = expected_answer.strip()
    return normalized


def normalize_question_items(items: list[Any], id_prefix: str) -> list[dict[str, str]]:
    return [normalize_question_item(item, f"{id_prefix}_{index}") for index, item in enumerate(items, 1)]


def default_questions(corpus_name: str) -> tuple[list[dict[str, str]], str]:
    question_set = resolve_builtin_question_set(corpus_name)
    return normalize_question_items(DEFAULT_QUESTION_SETS[question_set], question_set), question_set


def normalize_questions(raw_questions: list[str]) -> list[dict[str, str]]:
    return normalize_question_items(raw_questions, "question")


def load_questions_file(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        data = read_json(path)
        if isinstance(data, dict):
            data = data.get("questions")

        if not isinstance(data, list):
            raise click.ClickException(f"Question JSON {path} must be a list or an object with a 'questions' list.")

        return normalize_question_items(data, "question")

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return normalize_questions([line for line in lines if line])


def resolve_questions(
    question: tuple[str, ...],
    questions_file: Path | None,
    corpus_name: str,
) -> tuple[list[dict[str, str]], str]:
    if question:
        return normalize_questions([item.strip() for item in question if item.strip()]), "cli"
    if questions_file is not None:
        return load_questions_file(questions_file), str(questions_file)
    questions, question_set = default_questions(corpus_name)
    return questions, f"default:{question_set}"


def resolve_adapter_paths(run_dir: Path | None, adapter_dir: Path | None, adapter_name: str) -> tuple[Path, Path]:
    adapter_root = adapter_dir or (run_dir / "unlearned_adapter" if run_dir is not None else None)
    if adapter_root is None:
        raise click.ClickException("Pass --run-dir or --adapter-dir.")
    if not adapter_root.exists():
        raise click.ClickException(f"Missing adapter directory: {adapter_root}")

    named_adapter = adapter_root / adapter_name
    if (named_adapter / "adapter_config.json").exists():
        return adapter_root, named_adapter
    if (adapter_root / "adapter_config.json").exists():
        return adapter_root, adapter_root

    raise click.ClickException(
        f"Could not find adapter_config.json in {adapter_root} or {named_adapter}. "
        "Pass --adapter-dir pointing at the unlearned adapter directory."
    )


def infer_model_id(model_id: str | None, config: dict[str, Any], adapter_path: Path) -> str:
    if model_id:
        return model_id

    config_model_id = config.get("model_id")
    if isinstance(config_model_id, str) and config_model_id:
        return config_model_id

    adapter_config = read_json(adapter_path / "adapter_config.json")
    base_model = adapter_config.get("base_model_name_or_path") if isinstance(adapter_config, dict) else None
    if isinstance(base_model, str) and base_model:
        return base_model

    raise click.ClickException("Could not infer the base model id. Pass --model-id.")


def tokenizer_source(adapter_root: Path, model_id: str) -> str:
    if (adapter_root / "tokenizer_config.json").exists() or (adapter_root / "tokenizer.json").exists():
        return str(adapter_root)
    return model_id


def apply_prompt_template(question: str, prompt_template: str) -> str:
    if "{question}" in prompt_template:
        return prompt_template.format(question=question)
    return f"{prompt_template}{question}"


def answer_payload(
    item: dict[str, str],
    unlearned_generated: dict[str, Any],
    base_model_generated: dict[str, Any],
    include_prompts: bool,
) -> dict[str, Any]:
    payload = {
        "id": item["id"],
        "question": item["question"],
        "unlearned_answer": unlearned_generated["response"],
        "base_model_answer": base_model_generated["response"],
    }
    if "expected_answer" in item:
        payload["correct_answer"] = item["expected_answer"]
    if include_prompts:
        payload.update(
            {
                "prompt": unlearned_generated["prompt"],
                "input_tokens": unlearned_generated["input_tokens"],
                "generated_tokens": unlearned_generated["generated_tokens"],
                "full_completion": unlearned_generated["full_completion"],
                "unlearned_input_tokens": unlearned_generated["input_tokens"],
                "unlearned_generated_tokens": unlearned_generated["generated_tokens"],
                "unlearned_full_completion": unlearned_generated["full_completion"],
                "base_model_input_tokens": base_model_generated["input_tokens"],
                "base_model_generated_tokens": base_model_generated["generated_tokens"],
                "base_model_full_completion": base_model_generated["full_completion"],
            }
        )
    return payload


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    device = model_device(model)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    enc = {key: value.to(device) for key, value in enc.items()}
    input_tokens = int(enc["input_ids"].shape[-1])

    generate_kwargs = {
        **enc,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }
    if no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(**generate_kwargs)

    output_ids = generated[0]
    response_ids = output_ids[input_tokens:]
    return {
        "input_tokens": input_tokens,
        "generated_tokens": int(response_ids.shape[-1]),
        "response": tokenizer.decode(response_ids, skip_special_tokens=True).strip(),
        "full_completion": tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
    }


@click.command()
@click.option(
    "--run-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Selective unlearning run directory containing unlearned_adapter/.",
)
@click.option(
    "--adapter-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Unlearned adapter directory. Defaults to RUN_DIR/unlearned_adapter.",
)
@click.option("--model-id", default=None, help="Base Llama model id/path. Defaults to the run config or adapter config.")
@click.option("--adapter-name", default="unlearn", show_default=True, help="Adapter name saved by selective_unlearn.py.")
@click.option("--question", multiple=True, help="Question/prompt to ask. Can be passed multiple times.")
@click.option(
    "--questions-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="TXT file with one question per line, or JSON list/object with questions.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write responses. Defaults to RUN_DIR/query_responses.json.",
)
@click.option("--dtype", default="bfloat16", show_default=True, type=click.Choice(["bfloat16", "float16", "float32"]))
@click.option("--max-input-tokens", default=512, show_default=True, type=click.IntRange(min=1))
@click.option("--max-new-tokens", default=64, show_default=True, type=click.IntRange(min=1))
@click.option("--sample/--greedy", default=False, show_default=True, help="Use sampling or deterministic decoding.")
@click.option("--temperature", default=0.7, show_default=True, type=click.FloatRange(min=0.0, min_open=True))
@click.option("--top-p", default=0.9, show_default=True, type=click.FloatRange(min=0.0, max=1.0, min_open=True))
@click.option("--repetition-penalty", default=1.15, show_default=True, type=click.FloatRange(min=1.0))
@click.option("--no-repeat-ngram-size", default=4, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--prompt-template",
    default="Question: {question}\nAnswer:",
    show_default=True,
    help="Template used to turn each question into a model prompt.",
)
@click.option("--include-prompts", is_flag=True, help="Include prompts and token counts in responses.json.")
def cli(
    run_dir: Path | None,
    adapter_dir: Path | None,
    model_id: str | None,
    adapter_name: str,
    question: tuple[str, ...],
    questions_file: Path | None,
    output_file: Path | None,
    dtype: str,
    max_input_tokens: int,
    max_new_tokens: int,
    sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    prompt_template: str,
    include_prompts: bool,
):
    """Load an unlearned LoRA adapter, ask questions, and save model responses."""
    started = time.perf_counter()
    config = load_run_config(run_dir)
    adapter_root, adapter_path = resolve_adapter_paths(run_dir, adapter_dir, adapter_name)
    resolved_model_id = infer_model_id(model_id, config, adapter_path)
    corpus_name = infer_corpus_name(run_dir, config, adapter_dir)
    questions, questions_source = resolve_questions(question, questions_file, corpus_name)
    if not questions:
        raise click.ClickException("No questions were provided.")

    response_path = output_file or ((run_dir or adapter_root) / "query_responses.json")

    _, _, PeftModel, _, _ = import_training_deps()

    click.echo(f"Loading tokenizer: {tokenizer_source(adapter_root, resolved_model_id)}")
    tokenizer = load_tokenizer(tokenizer_source(adapter_root, resolved_model_id))

    click.echo(f"Loading base model: {resolved_model_id}")
    base_model = load_causal_lm(resolved_model_id, dtype)
    base_model.eval()

    prompt_items = [(item, apply_prompt_template(item["question"], prompt_template)) for item in questions]
    base_model_generations: dict[str, dict[str, Any]] = {}
    for item, prompt in prompt_items:
        click.echo(f"Asking base model {item['id']}: {item['question']}")
        generated = generate_one(
            base_model,
            tokenizer,
            prompt,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        generated["prompt"] = prompt
        base_model_generations[item["id"]] = generated

    click.echo(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path), adapter_name=adapter_name, is_trainable=False)
    model.set_adapter(adapter_name)
    model.eval()

    responses = []
    for item, prompt in prompt_items:
        click.echo(f"Asking unlearned model {item['id']}: {item['question']}")
        generated = generate_one(
            model,
            tokenizer,
            prompt,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        generated["prompt"] = prompt
        responses.append(
            answer_payload(
                item,
                generated,
                base_model_generations[item["id"]],
                include_prompts,
            )
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "adapter_root": str(adapter_root),
        "adapter_path": str(adapter_path),
        "adapter_name": adapter_name,
        "model_id": resolved_model_id,
        "base_model_id": resolved_model_id,
        "corpus_name": corpus_name,
        "questions_source": questions_source,
        "generation": {
            "dtype": dtype,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "temperature": temperature if sample else None,
            "top_p": top_p if sample else None,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "prompt_template": prompt_template,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "responses": responses,
    }
    dump_json(response_path, payload)
    click.echo(f"Wrote {len(responses)} responses to {response_path}")


if __name__ == "__main__":
    cli()
