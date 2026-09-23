#!/usr/bin/env bash
# Bootstrap a g5.2xlarge spot instance for multimodal Laya training.
#
# Launch with:
#   AMI       Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)
#   type      g5.2xlarge          A10G 24GB, 8 vCPU, 32GB RAM, 450GB NVMe
#   market    spot, persistent request, interruption behaviour = stop
#   EBS root  100 GB gp3          (code + env only — bulk data goes on NVMe)
#   IAM role  S3 read/write on your bucket
#
# Then:  bash aws/00_bootstrap.sh
#
# Idempotent: safe to re-run after a spot restart.
set -euo pipefail

REPO_URL="${REPO_URL:-}"                      # your laya-multimodal GitHub repo
DATA_DIR="${DATA_DIR:-/mnt/data}"
PROJ_DIR="${PROJ_DIR:-$HOME/laya-multimodal}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "no GPU visible — wrong AMI or the driver is still initialising"; exit 1; }

say "Instance-store NVMe -> $DATA_DIR"
# g5.2xlarge ships one ephemeral NVMe. It is NOT formatted at launch, and it is
# WIPED on stop — which for a spot instance means every interruption. Bulk data
# lives here because it is free and fast; anything you need to survive goes to S3.
# CAREFUL. `lsblk -d` lists whole disks, and a whole disk has no mountpoint even
# when its partition holds /. Selecting on "empty MOUNTPOINT" therefore matches
# the EBS ROOT disk and mkfs would destroy the instance. Match on the NVMe model
# string instead, and refuse outright if the candidate is the root disk.
ROOT_SRC=$(findmnt -no SOURCE / 2>/dev/null || true)
ROOT_DISK=$(lsblk -no PKNAME "$ROOT_SRC" 2>/dev/null | head -1 || true)
if [ -z "$ROOT_DISK" ]; then          # `[ x ] && y` under `set -e` exits when the test fails
  ROOT_DISK=$(basename "${ROOT_SRC:-none}" | sed 's/p\?[0-9]*$//')
fi
DISK=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/ {print $1; exit}')
echo "  root disk: ${ROOT_DISK:-unknown} | instance store: ${DISK:-none found}"
if [ -n "${DISK:-}" ] && [ "$DISK" = "$ROOT_DISK" ]; then
  echo "  REFUSING: instance-store detection matched the root disk. Not formatting."
  DISK=""
fi
if [ -n "${DISK:-}" ] && [ ! -d "$DATA_DIR/.ready" ]; then
  sudo mkfs -t xfs -f "/dev/$DISK"
  sudo mkdir -p "$DATA_DIR"
  sudo mount "/dev/$DISK" "$DATA_DIR"
  sudo chown -R "$USER:$USER" "$DATA_DIR"
  mkdir -p "$DATA_DIR/.ready"
  echo "formatted and mounted /dev/$DISK"
else
  sudo mkdir -p "$DATA_DIR"; sudo chown -R "$USER:$USER" "$DATA_DIR"
  echo "already mounted or no ephemeral disk found — using $DATA_DIR on EBS"
fi
df -h "$DATA_DIR" | tail -1

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq unzip awscli git tmux htop >/dev/null

say "Project"
if [ -n "$REPO_URL" ] && [ ! -d "$PROJ_DIR/.git" ]; then
  git clone "$REPO_URL" "$PROJ_DIR"
elif [ -d "$PROJ_DIR/.git" ]; then
  git -C "$PROJ_DIR" pull --ff-only || echo "  (pull skipped — local changes)"
else
  echo "  REPO_URL not set; expecting $PROJ_DIR to already exist"
fi

say "Python env"
# The DLAMI ships torch+CUDA already. Install only what it lacks, and never
# reinstall torch — pip will happily replace the CUDA build with a CPU wheel.
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install -q --no-deps laya
pip install -q transformers safetensors huggingface_hub pillow numpy

say "Symlinks"
mkdir -p "$DATA_DIR"/{vqa,coco,cache,checkpoints}
MM="$PROJ_DIR"
for d in data cache checkpoints; do
  [ -L "$MM/$d" ] || { rm -rf "${MM:?}/$d"; ln -s "$DATA_DIR" "$MM/$d" 2>/dev/null || true; }
done
ln -sfn "$DATA_DIR/vqa"         "$MM/data_vqa"
ln -sfn "$DATA_DIR/coco"        "$MM/data_coco"
ln -sfn "$DATA_DIR/cache"       "$MM/cache_dir"
ln -sfn "$DATA_DIR/checkpoints" "$MM/ckpt"

say "GPU capability"
python3 - <<'PYEOF'
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("  compute capability %d.%d" % cap)
    if cap[0] >= 8:
        print("  bf16 supported -> set AMP_DTYPE='bf16' in the notebook and drop")
        print("  the GradScaler. Same speed, wider range, no loss-scale tuning.")
    else:
        print("  fp16 only -> keep the GradScaler")
    torch.backends.cuda.matmul.allow_tf32 = True
    print("  TF32 enabled for matmuls")
PYEOF

say "Ready"
cat <<TXT
  project   $PROJ_DIR
  data      $DATA_DIR   (ephemeral — wiped on spot stop)

  next:
    export S3_BUCKET=s3://your-bucket/laya-mm
    bash aws/01_data.sh
    nohup bash aws/02_spot_guard.sh > /tmp/spot_guard.log 2>&1 &
    tmux new -s train        # so an SSH drop does not kill the run
TXT
