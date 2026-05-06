#!/usr/bin/env bash
set -u

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
PHASE2_SYSTEM_LOG="$LOG_DIR/system_phase2.log"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

# On Windows/Git Bash, command -v can resolve to the Microsoft Store alias.
if [[ "${PYTHON_BIN}" == *"WindowsApps/python"* ]] || [[ "${PYTHON_BIN}" == *"WindowsApps/python3"* ]]; then
  if [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python executable not found." | tee "$PHASE2_SYSTEM_LOG"
  exit 1
fi

echo "Using Python: $PYTHON_BIN" | tee "$PHASE2_SYSTEM_LOG"

PIP_PACKAGES=(
  "torch"
  "transformers"
  "peft"
  "bitsandbytes"
  "datasets"
  "accelerate"
)

echo "Installing dependencies for Phase 2..." | tee -a "$PHASE2_SYSTEM_LOG"
"$PYTHON_BIN" -m pip install --upgrade pip 2>&1 | tee "$LOG_DIR/pip_upgrade_phase2.log"
"$PYTHON_BIN" -m pip install "${PIP_PACKAGES[@]}" 2>&1 | tee "$LOG_DIR/pip_install_phase2.log"
PIP_STATUS=${PIPESTATUS[0]}

if [[ $PIP_STATUS -ne 0 ]]; then
  echo "Dependency installation failed. See $LOG_DIR/pip_install_phase2.log" | tee -a "$PHASE2_SYSTEM_LOG"
  exit $PIP_STATUS
fi

echo "Running preflight gate (00_preflight.sh)..." | tee -a "$PHASE2_SYSTEM_LOG"
bash "00_preflight.sh" 2>&1 | tee "$LOG_DIR/00_preflight_phase2.log"
PREFLIGHT_STATUS=${PIPESTATUS[0]}

if [[ $PREFLIGHT_STATUS -ne 0 ]]; then
  echo "Preflight failed. Aborting without running Phase 2 scripts." | tee -a "$PHASE2_SYSTEM_LOG"
  exit $PREFLIGHT_STATUS
fi

run_step() {
  local script_name="$1"
  local log_file="$LOG_DIR/${script_name%.py}.log"

  echo "Running $script_name..." | tee -a "$PHASE2_SYSTEM_LOG"
  "$PYTHON_BIN" "$script_name" >"$log_file" 2>&1
  local status=$?

  if [[ $status -ne 0 ]]; then
    echo "FAILED: $script_name (exit code $status)" | tee -a "$PHASE2_SYSTEM_LOG"
    if grep -qi "out of memory\|oom" "$log_file"; then
      echo "Detected OOM signature in $log_file" | tee -a "$PHASE2_SYSTEM_LOG"
    fi
    echo "--- Begin $script_name log ---" | tee -a "$PHASE2_SYSTEM_LOG"
    cat "$log_file" | tee -a "$PHASE2_SYSTEM_LOG"
    echo "--- End $script_name log ---" | tee -a "$PHASE2_SYSTEM_LOG"
    return $status
  fi

  echo "PASSED: $script_name" | tee -a "$PHASE2_SYSTEM_LOG"
  cat "$log_file"
  return 0
}

FAILURES=0
run_step "04_train_reinforced.py" || FAILURES=$((FAILURES + 1))
run_step "05_translate_data.py" || FAILURES=$((FAILURES + 1))

if [[ -d "./reinforced_adapter" ]] && [[ -d "./wmdp_translated" ]]; then
  run_step "06_cache_generic_logits.py" || FAILURES=$((FAILURES + 1))
else
  echo "SKIPPED: 06_cache_generic_logits.py (missing ./reinforced_adapter or ./wmdp_translated)" | tee -a "$PHASE2_SYSTEM_LOG"
fi

if [[ $FAILURES -ne 0 ]]; then
  echo "Phase 2 completed with $FAILURES failure(s). See logs in $LOG_DIR/." | tee -a "$PHASE2_SYSTEM_LOG"
  exit 1
fi

echo "Phase 2 scripts completed successfully." | tee -a "$PHASE2_SYSTEM_LOG"
