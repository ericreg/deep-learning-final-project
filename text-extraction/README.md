```sh
CUDA_VISIBLE_DEVICES=1 uv run python selective_unlearn.py --results-dir results/iliad --selection surprisal --max-train-chunks 500 --top-n 625
CUDA_VISIBLE_DEVICES=1 uv run python selective_unlearn.py --results-dir results/iliad --selection emotion --max-train-chunks 500 --top-n 625
CUDA_VISIBLE_DEVICES=1 uv run python selective_unlearn.py --results-dir results/iliad --selection random --max-train-chunks 500 --top-n 625

CUDA_VISIBLE_DEVICES=1 uv run python selective_unlearn.py --results-dir results/bible --selection surprisal --max-train-chunks 500 --top-n 625
CUDA_VISIBLE_DEVICES=1 uv run python selective_unlearn.py --results-dir results/bible --selection emotion --max-train-chunks 500 --top-n 625
CUDA_VISIBLE_DEVICES=1 uv run python selective_unlearn.py --results-dir results/bible --selection random --max-train-chunks 500 --top-n 625

CUDA_VISIBLE_DEVICES=1 uv run python selective_unlearn.py --results-dir results/hp --selection surprisal --max-train-chunks 500 --top-n 625
CUDA_VISIBLE_DEVICES=1 uv run python selective_unlearn.py --results-dir results/hp --selection emotion --max-train-chunks 500 --top-n 625
CUDA_VISIBLE_DEVICES=1 uv run python selective_unlearn.py --results-dir results/hp --selection random --max-train-chunks 500 --top-n 625


CUDA_VISIBLE_DEVICES=1 uv run python benchmark_unlearned_model.py --run-dir selective_unlearning_runs/bible/surprisal                                              
CUDA_VISIBLE_DEVICES=1 uv run python benchmark_unlearned_model.py --run-dir selective_unlearning_runs/bible/random

CUDA_VISIBLE_DEVICES=1 uv run python benchmark_unlearned_model.py --run-dir selective_unlearning_runs/iliad/emotion
CUDA_VISIBLE_DEVICES=1 uv run python benchmark_unlearned_model.py --run-dir selective_unlearning_runs/iliad/surprisal
CUDA_VISIBLE_DEVICES=1 uv run python benchmark_unlearned_model.py --run-dir selective_unlearning_runs/iliad/random

CUDA_VISIBLE_DEVICES=1 uv run python benchmark_unlearned_model.py --run-dir selective_unlearning_runs/hp/emotion
CUDA_VISIBLE_DEVICES=1 uv run python benchmark_unlearned_model.py --run-dir selective_unlearning_runs/hp/surprisal
CUDA_VISIBLE_DEVICES=1 uv run python benchmark_unlearned_model.py --run-dir selective_unlearning_runs/hp/random



CUDA_VISIBLE_DEVICES=0 uv run python query_unlearned_model.py --run-dir selective_unlearning_runs/iliad/surprisal --output-file qual/iliad/surprisal/responses.json
CUDA_VISIBLE_DEVICES=0 uv run python query_unlearned_model.py --run-dir selective_unlearning_runs/iliad/emotion --output-file qual/iliad/emotion/responses.json
CUDA_VISIBLE_DEVICES=0 uv run python query_unlearned_model.py --run-dir selective_unlearning_runs/iliad/random --output-file qual/iliad/random/responses.json
CUDA_VISIBLE_DEVICES=0 uv run python query_unlearned_model.py --run-dir selective_unlearning_runs/bible/surprisal --output-file qual/bible/surprisal/responses.json
CUDA_VISIBLE_DEVICES=0 uv run python query_unlearned_model.py --run-dir selective_unlearning_runs/bible/emotion --output-file qual/bible/emotion/responses.json

CUDA_VISIBLE_DEVICES=1 uv run python query_unlearned_model.py --run-dir selective_unlearning_runs/bible/random --output-file qual/bible/random/responses.json
CUDA_VISIBLE_DEVICES=1 uv run python query_unlearned_model.py --run-dir selective_unlearning_runs/hp/surprisal --output-file qual/hp/surprisal/responses.json
CUDA_VISIBLE_DEVICES=1 uv run python query_unlearned_model.py --run-dir selective_unlearning_runs/hp/emotion --output-file qual/hp/emotion/responses.json
CUDA_VISIBLE_DEVICES=1 uv run python query_unlearned_model.py --run-dir selective_unlearning_runs/hp/random --output-file qual/hp/random/responses.json
```