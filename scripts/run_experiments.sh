#!/usr/bin/env bash
# Background launcher for the accuracy-ladder experiments.
# Usage:
#   bash scripts/run_experiments.sh --list
#   bash scripts/run_experiments.sh --all
#   bash scripts/run_experiments.sh --only noul
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-/data/venv/bin/python3}"
LOG_DIR="${LOG_DIR:-/data/laya-mm}"
mkdir -p "$LOG_DIR" "$ROOT/checkpoints/experiments"

# GPU device nodes (nvidia-caps can be root-only after reload)
chmod 666 /dev/nvidia-caps/nvidia-cap* 2>/dev/null || sudo chmod 666 /dev/nvidia-caps/nvidia-cap* 2>/dev/null || true

export HF_HOME="${HF_HOME:-/data/laya-mm/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
unset LD_LIBRARY_PATH || true

cd "$ROOT"

if [[ "${1:-}" == "--list" ]] || [[ "${1:-}" == "" ]]; then
  exec "$PY" scripts/run_experiments.py --list "$@"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/experiments_${STAMP}.log"
PIDFILE="$LOG_DIR/experiments.pid"

nohup "$PY" scripts/run_experiments.py --python "$PY" "$@" \
  >"$LOG" 2>&1 &
echo $! | tee "$PIDFILE"
echo "started PID=$(cat "$PIDFILE")"
echo "log: $LOG"
echo "tail -f $LOG"
