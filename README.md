# LLM Unlearning 

Three independent implementations of machine unlearning for LLMs, each in its own folder. All are grounded in or extend [Eldan & Russinovich (2023) — *Who's Harry Potter? Approximate Unlearning in LLMs*](https://arxiv.org/abs/2310.02238).

---

## supervised/

Targeted unlearning of hazardous biological and cybersecurity knowledge from Meta-Llama-3-8B-Instruct using **Discrepancy Minimization**, adapting the Eldan & Russinovich framework to the [WMDP benchmark](https://arxiv.org/abs/2403.03218). A reinforced LoRA adapter amplifies hazardous domain salience; a second adapter minimizes KL divergence against a generic replacement distribution constructed from out-of-distribution hypernyms.

Numbered scripts `00` → `10` form a sequential pipeline. See `supervised/README.md` for details.

---

## text-extraction/

Selective unlearning of a target book (Harry Potter) that extends Eldan & Russinovich by adding **chunk selection strategies** to identify the most "book-specific" passages before unlearning. Three selection methods are compared: surprisal (embedding-based), emotion volatility, and random baseline. The unlearning step follows the same reinforced adapter + KL suppression pattern from the source paper.

Entry points: `text_extract.py` (chunk selection) and `selective_unlearn.py` (unlearning). Results and figures are in `selective_unlearning_runs/`.

---

## unsupervised/

A faithful, fully unsupervised reimplementation of [Eldan & Russinovich (2023)](https://arxiv.org/abs/2310.02238). Requires no labeled data or entity lists. The model's own logit ratios identify "anchor tokens" where it over-knows the target book; a second LoRA adapter is trained on soft replacement labels to suppress only those positions.

Runnable as a Colab notebook (`hp_unlearning.ipynb`) or via numbered CLI scripts `01` → `05`. See `unsupervised/README.md` for full details.
