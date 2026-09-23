#!/usr/bin/env bash
# Fetch VQA v2 + COCO train2014, and extract SigLIP features.
#
# Everything lands on the ephemeral NVMe. Set S3_BUCKET and the expensive
# artifact (the feature cache) is pulled from / pushed to S3, so a spot
# interruption costs you a download rather than a re-extraction.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/mnt/data}"
PROJ_DIR="${PROJ_DIR:-$HOME/laya-multimodal}"
MM="$PROJ_DIR"
S3_BUCKET="${S3_BUCKET:-}"
VISION="${VISION:-google/siglip-base-patch16-224}"
MAX_IMAGES="${MAX_IMAGES:-0}"                 # 0 = all of COCO train2014
STEM="$DATA_DIR/cache/siglip_train2014"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
have() { [ -s "$1" ]; }

say "VQA v2 annotations"
mkdir -p "$DATA_DIR/vqa" && cd "$DATA_DIR/vqa"
for z in v2_Questions_Train_mscoco v2_Annotations_Train_mscoco; do
  have "$z.zip" || wget -q --show-progress -c \
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/$z.zip"
done
unzip -oq '*.zip'; ls -1 *.json

say "COCO train2014 images (~13 GB)"
mkdir -p "$DATA_DIR/coco" && cd "$DATA_DIR/coco"
if [ ! -d train2014 ]; then
  wget -q --show-progress -c http://images.cocodataset.org/zips/train2014.zip
  unzip -oq train2014.zip && rm -f train2014.zip
fi
echo "  $(ls train2014 | wc -l) images"

say "SigLIP features"
if [ -n "$S3_BUCKET" ] && aws s3 ls "$S3_BUCKET/siglip_train2014.npy" >/dev/null 2>&1; then
  echo "  found in S3 — downloading instead of re-extracting"
  aws s3 cp "$S3_BUCKET/siglip_train2014.npy"  "$STEM.npy"
  aws s3 cp "$S3_BUCKET/siglip_train2014.json" "$STEM.json"
elif have "$STEM.npy"; then
  echo "  already extracted locally"
else
  mkdir -p "$DATA_DIR/cache"
  cd "$MM"
  ARGS=(--images "$DATA_DIR/coco/train2014" --out "$STEM" --model "$VISION" --batch 64)
  [ "$MAX_IMAGES" -gt 0 ] && ARGS+=(--max-images "$MAX_IMAGES")
  python3 scripts/extract_features.py "${ARGS[@]}"
  if [ -n "$S3_BUCKET" ]; then
    echo "  uploading to S3 so an interruption does not cost a re-extraction"
    aws s3 cp "$STEM.npy"  "$S3_BUCKET/siglip_train2014.npy"
    aws s3 cp "$STEM.json" "$S3_BUCKET/siglip_train2014.json"
  fi
fi

say "Done"
du -sh "$DATA_DIR"/* 2>/dev/null || true
df -h "$DATA_DIR" | tail -1
