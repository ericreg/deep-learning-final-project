# Approximate LLM Unlearning

A generalized, fully unsupervised implementation of [Eldan & Russinovich (2023) — "Who's Harry Potter? Approximate Unlearning in LLMs"](https://arxiv.org/abs/2310.02238).

Surgically removes a target book's knowledge from an LLM while preserving general language capabilities without any hardcoded entity lists, labeled Q&A pairs, or book-specific knowledge.



---

## How It Works

The method uses the book's own content as the only signal for forgetting. The pipeline:

1. **Load corpus** — extract and chunk text from any PDF into 512-token windows
2. **Train reinforced adapter** — fine-tune a LoRA adapter on the book, amplifying book-specific knowledge
3. **Detect anchor tokens** — identify positions where the reinforced model diverges from the frozen baseline using a logit ratio scan:

$$\text{ratio}_i = \frac{p_{\text{reinforced}}(x_i \mid x_{<i})}{p_{\text{baseline}}(x_i \mid x_{<i})}$$

   Positions where `ratio ≥ ANCHOR_THRESHOLD` are *anchor tokens* — the model over-knows these from the book.

4. **Unlearn** — train a second LoRA adapter using soft replacement labels that suppress only book-boosted tokens:

$$v_{\text{generic}} = v_{\text{baseline}} - \alpha \cdot \text{ReLU}(v_{\text{reinforced}} - v_{\text{baseline}})$$
$$\mathcal{L} = \text{KL}(\log \sigma(v_{\text{unlearn}}) \| \sigma(v_{\text{generic}}))$$

5. **Evaluate** — measure forgetting and capability retention using fully unsupervised metrics

All evaluation is driven by the book's own text — no labeled test sets required.

---

## Requirements

- **GPU**: CUDA-capable GPU with ≥ 16 GB VRAM (bfloat16 / float16 precision)
- **HuggingFace account** with access to [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (requires accepting Meta's license)
- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

There are two ways to run the pipeline.

### Option A — Colab Notebook (recommended)

Open `unlearning.ipynb` in Google Colab with a GPU runtime.

**Setup:**
1. Add your HuggingFace token as a Colab Secret named `HF_TOKEN` (left sidebar → key icon), or paste it into the `HF_TOKEN` variable in the auth cell
2. Upload your book PDF(s) to `/content/` and set `PDF_GLOB` in the config cell
3. Run cells top to bottom — each step caches its output, so you can re-run safely after a session restart

Artifacts are saved to Google Drive at `MyDrive/book_unlearning/` if Drive is mounted, or locally at `/content/book_unlearning/` otherwise.

**Evaluation:**
Run the eval section twice — first with `EVAL_ADAPTER = None` (base model baseline), then with `EVAL_ADAPTER = str(UNLEARNED_DIR / "unlearn")` — to generate the before/after comparison plots.

### Option B — CLI Scripts (Harry Potter)

The numbered scripts in the repo root run the same pipeline from the command line. They expect HP PDF files (`hp*.pdf`) in the current directory.

```bash
# Step 1: Extract and chunk text from PDFs
python 01_load_corpus.py

# Step 2: Train reinforced LoRA adapter
python 02_train_reinforced.py

# Step 3: Detect anchor tokens
python 03_detect_anchor_tokens.py

# Step 4: Train unlearn adapter
python 04_unlearn.py

# Step 5: Evaluate
python 05_eval_hp.py                                  # base model baseline
python 05_eval_hp.py --adapter ./hp_unlearned_adapter/unlearn  # unlearned model
```

Pass `--hp-text-file <path>` to `05_eval_hp.py` to include book perplexity in the evaluation. Use `--skip-benchmarks` to skip `lm-eval` and run faster.

---

## Configuration

Key hyperparameters in the notebook config cell (or at the top of each script):

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `meta-llama/Meta-Llama-3-8B` | Base model to unlearn from |
| `CHUNK_SIZE` | `512` | Token window size for corpus chunking |
| `EVAL_HOLDOUT` | `0.1` | Fraction of chunks held out for evaluation |
| `ANCHOR_THRESHOLD` | `2.0` | Logit-ratio cutoff for flagging anchor tokens |
| `ALPHA` | `5.0` | Suppression strength in soft label formula |
| `REINF_LR` | `2e-4` | Learning rate for reinforced adapter |
| `REINF_EPOCHS` | `1` | Epochs for reinforced adapter training |
| `UNLEARN_LR` | `2e-5` | Learning rate for unlearn adapter |
| `UNLEARN_EPOCHS` | `3` | Epochs for unlearn adapter training |

---

## Evaluation Metrics

| Metric | What it measures | Expected direction after unlearning |
|---|---|---|
| **Anchor recall** | % of anchor-flagged positions the model still predicts correctly | Down |
| **Book perplexity** | Model's surprise on held-out book text | Up |
| **Control perplexity** | Model's surprise on non-book text | Flat |
| **Cloze top-1/5/10** | Rank of expected HP-specific tokens (HP scripts only) | Down |
| **ARC-Easy / BoolQ / Winogrande** | General reasoning benchmarks via `lm-eval` | Flat |

---

## File Structure

```
.
├── unlearning.ipynb          # Main Colab notebook — any PDF
├── hp_unlearning.ipynb       # Harry Potter–specific Colab notebook
├── 01_load_corpus.py         # Extract and chunk PDF text
├── 02_train_reinforced.py    # Train reinforced LoRA adapter
├── 03_detect_anchor_tokens.py# Compute per-token logit ratios, save anchor masks
├── 04_unlearn.py             # Train unlearn adapter on soft replacement labels
├── 05_eval_hp.py             # Evaluate cloze accuracy, perplexity, and benchmarks
├── text_extract.py           # Standalone PDF text extraction utility
└── requirements.txt          # Python dependencies
```

---

## Reference

Eldan, R., & Russinovich, M. (2023). *Who's Harry Potter? Approximate Unlearning in LLMs*. arXiv:2310.02238. https://arxiv.org/abs/2310.02238