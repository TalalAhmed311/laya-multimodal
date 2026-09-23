#!/usr/bin/env bash
# Run training headless, with periodic checkpoint sync to S3.
# Run inside tmux so an SSH drop does not take the job with it.
set -euo pipefail

PROJ_DIR="${PROJ_DIR:-$HOME/laya-multimodal}"
MM="$PROJ_DIR"
S3_BUCKET="${S3_BUCKET:-}"
CKPT_DIR="${CKPT_DIR:-/mnt/data/checkpoints}"

cd "$MM"
mkdir -p "$CKPT_DIR"

# Periodic sync, so an interruption at a bad moment loses at most 10 minutes.
if [ -n "$S3_BUCKET" ]; then
  ( while true; do sleep 600
      aws s3 sync "$CKPT_DIR" "$S3_BUCKET/checkpoints/" --only-show-errors || true
    done ) & SYNC_PID=$!
  trap 'kill $SYNC_PID 2>/dev/null || true' EXIT
fi

echo "== baseline first — you need the floor before any multimodal number means anything =="
python3 scripts/baseline_text_only.py --vqa /mnt/data/vqa --limit 4000 \
  2>&1 | tee "$CKPT_DIR/baseline_text_only.log"

echo "== training =="
jupyter nbconvert --to notebook --execute notebooks/train_multimodal.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output "$CKPT_DIR/train_executed.ipynb" 2>&1 | tee "$CKPT_DIR/train.log"

[ -n "$S3_BUCKET" ] && aws s3 sync "$CKPT_DIR" "$S3_BUCKET/checkpoints/" --only-show-errors
echo "done — artifacts in $CKPT_DIR"
