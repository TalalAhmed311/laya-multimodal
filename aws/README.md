# Training on AWS — g5.2xlarge spot

## Launch

| | |
|---|---|
| AMI | Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04) |
| Instance | `g5.2xlarge` — A10G 24GB, 8 vCPU, 32GB RAM, 450GB NVMe |
| Market | **Spot**, persistent request, interruption behaviour = `stop` |
| EBS root | 100 GB gp3 — code and env only |
| IAM role | S3 read/write on your bucket |
| Security group | SSH from your IP |

~$0.42/hr spot against ~$1.21 on-demand. Runs are ~30 minutes and fully
checkpointed, so an interruption costs almost nothing.

**Use the DLAMI.** It ships a working CUDA + PyTorch. Installing drivers on
bare Ubuntu costs an hour and is the most common way to lose an afternoon here.

## Run

```bash
export REPO_URL=https://github.com/<you>/laya-multimodal.git
export S3_BUCKET=s3://your-bucket/laya-mm

bash aws/00_bootstrap.sh                 # NVMe, deps, clone
bash aws/01_data.sh                      # VQA + COCO + SigLIP features
nohup bash aws/02_spot_guard.sh > /tmp/spot_guard.log 2>&1 &
tmux new -s train
bash aws/03_train.sh
```

## Why each piece exists

**`00_bootstrap.sh`** — formats and mounts the instance-store NVMe. It ships
unformatted and is **wiped on stop**, which for a spot instance means every
interruption. Bulk data lives there because it is free and fast; anything that
must survive goes to S3.

It also installs `laya` with `--no-deps`. Left alone, pip will happily replace
the DLAMI's CUDA torch build with a CPU wheel, and you will not notice until
training is mysteriously 30× slow.

**`01_data.sh`** — checks S3 for the feature cache before extracting. Extraction
is ~7 minutes for all of COCO, but a download is faster and deterministic.

**`02_spot_guard.sh`** — polls the instance metadata for the two-minute
interruption notice (IMDSv2, token required — v1 is off on current AMIs) and
syncs checkpoints when it lands. Without it, an interruption silently eats
whatever was not saved.

**`03_train.sh`** — runs the text-only baseline **first**, then training, with a
background S3 sync every 10 minutes so a bad-moment interruption costs at most
10 minutes of work.

## Expected timings (A10G)

| | |
|---|---|
| Bootstrap + COCO download | ~25 min (network-bound) |
| SigLIP feature extraction, full COCO | ~7 min (I/O-bound, not GPU) |
| Baseline text-only, 4k items | ~2 min |
| Training, VQA yes/no 440k × 3 epochs | **~28 min** |
| Training, full mixture 1.2M × 3 epochs | ~75 min |

The cost is iteration, not any single run. Budget for 15–20 runs across
debugging and the ablation ladder: ~10 hours ≈ **$4 spot**.

## Cost traps

- **EBS is billed whether or not the instance runs.** ~$0.08/GB-month — a 500GB
  volume is $40/month idle. Keep the root small and use the free NVMe.
- **S3 at ~$0.023/GB-month** is where the feature cache belongs between sessions.
- Set interruption behaviour to **stop**, not terminate, so the root volume
  survives and a restart is a `git pull` rather than a rebuild.
