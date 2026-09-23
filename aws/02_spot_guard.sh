#!/usr/bin/env bash
# Spot-interruption watcher.
#
# AWS gives a two-minute warning before reclaiming a spot instance. This polls
# for it and syncs checkpoints to S3 when it lands. Without this, an
# interruption silently eats whatever the run had not saved.
#
#   nohup bash aws/02_spot_guard.sh > /tmp/spot_guard.log 2>&1 &
set -uo pipefail

S3_BUCKET="${S3_BUCKET:-}"
CKPT_DIR="${CKPT_DIR:-/mnt/data/checkpoints}"
INTERVAL="${INTERVAL:-5}"

[ -z "$S3_BUCKET" ] && { echo "S3_BUCKET unset — nothing to sync to, exiting"; exit 1; }

md() {  # IMDSv2 requires a token; IMDSv1 is disabled on newer AMIs
  local t; t=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" --max-time 2) || return 1
  curl -s -H "X-aws-ec2-metadata-token: $t" --max-time 2 \
    -o /dev/null -w '%{http_code}' "http://169.254.169.254/latest/meta-data/$1"
}

echo "$(date -Is) spot guard up — polling every ${INTERVAL}s, target $S3_BUCKET"
while true; do
  code=$(md "spot/instance-action" || echo "000")
  if [ "$code" = "200" ]; then
    echo "$(date -Is) INTERRUPTION NOTICE — ~2 minutes. Syncing."
    aws s3 sync "$CKPT_DIR" "$S3_BUCKET/checkpoints/" --only-show-errors
    echo "$(date -Is) sync complete"
    exit 0
  fi
  sleep "$INTERVAL"
done
