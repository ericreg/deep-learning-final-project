#!/usr/bin/env bash
set -u

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
PRE_LOG="$LOG_DIR/00_preflight.log"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

# On Windows/Git Bash, command -v can resolve to the Microsoft Store alias.
if [[ "${PYTHON_BIN}" == *"WindowsApps/python"* ]] || [[ "${PYTHON_BIN}" == *"WindowsApps/python3"* ]]; then
  if [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python executable not found." | tee "$PRE_LOG"
  exit 1
fi

echo "Using Python: $PYTHON_BIN" | tee "$PRE_LOG"
echo "Running preflight checks..." | tee -a "$PRE_LOG"

"$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$PRE_LOG"
import os
import sys

import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

MODEL_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"


def fail(message: str, code: int = 1) -> None:
    print(f"[FAIL] {message}")
    sys.exit(code)


print("[CHECK] torch.cuda.is_available()")
cuda_ok = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {cuda_ok}")
if not cuda_ok:
    fail("CUDA is not available. A CUDA-enabled PyTorch runtime is required.")

print("[CHECK] GPU recognition")
device_count = torch.cuda.device_count()
if device_count < 1:
    fail("No CUDA devices were detected.")

for idx in range(device_count):
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / (1024 ** 3)
    print(f"GPU[{idx}]: {name} | total_vram_gb={total_gb:.2f}")

print("[CHECK] Hugging Face token presence")
token = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)
if not token:
    fail(
        "No Hugging Face token found in environment variables "
        "(HF_TOKEN or HUGGINGFACE_HUB_TOKEN)."
    )

print("[CHECK] Hugging Face token validity + gated repo access")
api = HfApi(token=token)
try:
    who = api.whoami(token=token)
    username = who.get("name") or who.get("email") or "unknown-user"
    print(f"Authenticated as: {username}")
except Exception as exc:
    fail(f"Invalid Hugging Face token or auth failure: {exc}")

try:
    info = api.model_info(MODEL_REPO, token=token)
    print(f"Model access OK: {info.id}")
except HfHubHTTPError as exc:
    fail(
        "Token is valid but access to meta-llama/Meta-Llama-3-8B-Instruct is missing "
        f"or forbidden: {exc}"
    )
except Exception as exc:
    fail(f"Unexpected Hugging Face model access error: {exc}")

print("[PASS] Preflight checks completed successfully.")
PY

status=${PIPESTATUS[0]}
if [[ $status -ne 0 ]]; then
  echo "Preflight FAILED. Aborting pipeline." | tee -a "$PRE_LOG"
  exit $status
fi

echo "Preflight PASSED." | tee -a "$PRE_LOG"
exit 0
