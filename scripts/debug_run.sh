#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

LOG_DIR="${LOG_DIR:-$ROOT_DIR/results/debug}"
mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/debug_run_${RUN_ID}.log"

echo "Running TradingAgents debug harness..."
echo "Python: $PYTHON_BIN"
echo "Log: $LOG_FILE"
echo

set +e
"$PYTHON_BIN" scripts/debug_run.py "$@" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

echo
echo "Exit code: $STATUS"
echo "Error summary (if any):"
rg -n "^\[ERROR\]|Traceback|Codex API error" "$LOG_FILE" || true
echo
echo "Full log saved to: $LOG_FILE"

exit "$STATUS"
