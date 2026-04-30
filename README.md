# WMDP-Cyber Approximate Unlearning Hello World

This repo contains a small prototype of the approximate unlearning recipe from
`docs/harry_potter.pdf`, scored against WMDP-Cyber from `docs/wdmp.pdf`.

The implementation intentionally uses WMDP-Cyber benchmark questions for
evaluation only. Training data comes from the official WMDP cyber forget/retain
corpora, not from `cais/wmdp` benchmark rows.

## Environment

Use Python 3.11 or 3.12. The system Python in this workspace may be newer than
the PyTorch stack supports.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,eval,cuda]"
```

For ROCm, install the ROCm-enabled PyTorch wheel that matches your ROCm
runtime first, then install the project without the CUDA extra:

```bash
# Pick the exact ROCm index URL from the PyTorch install selector for your system.
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocmX.Y
uv pip install -e ".[dev,eval]"
```

Set a Hugging Face token with access to `meta-llama/Meta-Llama-3-8B-Instruct`:

```bash
export HF_TOKEN=...
```

## Commands

```bash
unlearning-demo preflight
unlearning-demo prepare
unlearning-demo reinforce
unlearning-demo unlearn
unlearning-demo score --adapter none
unlearning-demo score --adapter outputs/unlearn_adapter
unlearning-demo retain-eval --adapter outputs/unlearn_adapter
unlearning-demo report
```

On ROCm, disable the CUDA-oriented bitsandbytes 4-bit path:

```bash
unlearning-demo preflight --no-4bit
unlearning-demo score --no-4bit --adapter none
unlearning-demo reinforce --no-4bit
unlearning-demo unlearn --no-4bit
unlearning-demo score --no-4bit --adapter outputs/unlearn_adapter
```

For a dependency-light local check that does not require Llama, an accelerator, or WMDP:

```bash
python -m pytest
unlearning-demo smoke
```

## What This Prototype Does

1. Loads official WMDP cyber forget/retain corpora.
2. Trains a reinforced LoRA adapter on cyber forget text.
3. Builds generic translated versions of forget samples by replacing
   domain-specific anchor terms with generic placeholders.
4. Computes soft generic targets with:

   `v_generic = v_baseline_translated - alpha * relu(v_reinforced_original - v_baseline_translated)`

5. Trains a fresh LoRA adapter toward those generic targets.
6. Scores baseline/reinforced/unlearned models on the WMDP-Cyber `test` split
   with zero-shot four-choice log-likelihood over `A/B/C/D`.

This is a mechanics-first prototype, not evidence of robust hazardous-knowledge
removal.
