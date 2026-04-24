# Targeted Unlearning via Discrepancy Minimization

## 1. Project Objective
This project studies targeted unlearning on meta-llama/Meta-Llama-3-8B-Instruct to remove hazardous biological and cybersecurity knowledge while preserving general language-model utility. The core method is Discrepancy Minimization, where optimization is driven by Kullback-Leibler divergence against a dynamically constructed generic probability distribution that suppresses hazardous responses without broad catastrophic forgetting.

## 2. Hardware & Environment
- Hardware Limit: 1x NVIDIA RTX 3090 (24GB VRAM).
- Quantization: 4-bit NormalFloat (NF4) via bitsandbytes.
- Frameworks: PyTorch, Hugging Face transformers, peft, datasets.

## 3. Pipeline Architecture
- 00_preflight.sh: Validates GPU visibility/VRAM constraints and confirms Hugging Face gated model access before any expensive run.
- 01_profile_model.py: Runs a controlled NF4 + LoRA memory telemetry pass to measure peak allocation and verify the configuration fits 24GB VRAM.
- 02_prepare_data.py: Downloads, normalizes, and merges cais/wmdp subsets wmdp-bio and wmdp-cyber into a unified training/evaluation stream.
- 03_baseline_eval.py: Establishes pre-intervention utility baselines on ARC Challenge, BoolQ, and WinoGrande using lm-evaluation-harness.
- 04_train_reinforced.py: Trains a reinforced LoRA adapter that intentionally amplifies hazardous domain salience for discrepancy construction.
- 05a_extract_entities.py & 05_translate_data.py: Uses an NLTK POS pipeline to extract Out-of-Distribution technical nouns and map them to generic hypernyms to build x_translated.
- 08_dynamic_unlearn.py: Executes dynamic PEFT multi-adapter switching to compute the KL-target distribution online and optimize unlearning without full-logit disk caching under tight VRAM constraints. Note: Scripts 06 and 07 were static-caching proofs-of-concept and are deprecated.

## 4. Parallel Workstreams
- Optimization (Current): Tune the dynamic unlearning loop, including alpha penalty scaling, learning rate stability, and gradient accumulation behavior.
- Evaluation (Pending): Integrate WMDP multiple-choice evaluation into lm-evaluation-harness to quantify targeted forgetting versus retained general utility.
- Red Teaming (Pending): Build jailbreak and representation-space stress tests to validate that hazardous knowledge is functionally erased rather than superficially masked.
