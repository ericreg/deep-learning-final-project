#!/usr/bin/env bash
set -u

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

# On Windows/Git Bash, command -v can resolve to the Microsoft Store alias.
if [[ "${PYTHON_BIN}" == *"WindowsApps/python"* ]] || [[ "${PYTHON_BIN}" == *"WindowsApps/python3"* ]]; then
  if [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python executable not found." | tee "$LOG_DIR/system.log"
  exit 1
fi

echo "Using Python: $PYTHON_BIN" | tee "$LOG_DIR/system.log"

PIP_PACKAGES=(
  "torch"
  "transformers"
  "peft"
  "bitsandbytes"
  "datasets"
  "accelerate"
  "lm-eval"
)

echo "Installing dependencies..." | tee -a "$LOG_DIR/system.log"
"$PYTHON_BIN" -m pip install --upgrade pip 2>&1 | tee "$LOG_DIR/pip_upgrade.log"
"$PYTHON_BIN" -m pip install "${PIP_PACKAGES[@]}" 2>&1 | tee "$LOG_DIR/pip_install.log"
PIP_STATUS=${PIPESTATUS[0]}

if [[ $PIP_STATUS -ne 0 ]]; then
  echo "Dependency installation failed. See $LOG_DIR/pip_install.log" | tee -a "$LOG_DIR/system.log"
  exit $PIP_STATUS
fi

echo "Running preflight gate (00_preflight.sh)..." | tee -a "$LOG_DIR/system.log"
bash "00_preflight.sh" 2>&1 | tee "$LOG_DIR/00_preflight.log"
PREFLIGHT_STATUS=${PIPESTATUS[0]}

if [[ $PREFLIGHT_STATUS -ne 0 ]]; then
  echo "Preflight failed. Aborting without running profiling/data/eval scripts." | tee -a "$LOG_DIR/system.log"
  exit $PREFLIGHT_STATUS
fi

run_step() {
  local script_name="$1"
  local log_file="$LOG_DIR/${script_name%.py}.log"

  echo "Running $script_name..." | tee -a "$LOG_DIR/system.log"
  "$PYTHON_BIN" "$script_name" >"$log_file" 2>&1
  local status=$?

  if [[ $status -ne 0 ]]; then
    echo "FAILED: $script_name (exit code $status)" | tee -a "$LOG_DIR/system.log"
    if grep -qi "out of memory\|oom" "$log_file"; then
      echo "Detected OOM signature in $log_file" | tee -a "$LOG_DIR/system.log"
    fi
    echo "--- Begin $script_name log ---" | tee -a "$LOG_DIR/system.log"
    cat "$log_file" | tee -a "$LOG_DIR/system.log"
    echo "--- End $script_name log ---" | tee -a "$LOG_DIR/system.log"
    return $status
  fi

  echo "PASSED: $script_name" | tee -a "$LOG_DIR/system.log"
  cat "$log_file"
  return 0
}

FAILURES=0
run_step "01_profile_model.py" || FAILURES=$((FAILURES + 1))
run_step "02_prepare_data.py" || FAILURES=$((FAILURES + 1))
run_step "03_baseline_eval.py" || FAILURES=$((FAILURES + 1))

if [[ $FAILURES -ne 0 ]]; then
  echo "Completed with $FAILURES failure(s). See logs in $LOG_DIR/." | tee -a "$LOG_DIR/system.log"
  exit 1
fi

echo "All scripts completed successfully." | tee -a "$LOG_DIR/system.log"
