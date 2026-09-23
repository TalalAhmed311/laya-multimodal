# laya-multimodal

A Laya-style typed-decision model over **image + text**: ask several typed
questions about one image in a single forward pass and get calibrated
probabilities back, with no text generation.

Same three primitives as text Laya — `choice`, `score`, `noul` — and the same
RLCD training against strictly proper scoring rules. What changes is where the
state comes from.

**Weights (Hugging Face):** https://huggingface.co/TalalML123/laya-multimodal-best-all-tasks

---

## Best checkpoint (kept)

Trained on AWS **`g5.2xlarge`** (1× **NVIDIA A10G 24GB**) as part of a 14-run
ladder (~3.4 h training wall time). The winning run is the **joint all-tasks
stage-2** model (`noul` + `score` + `choice`, bridge warm-start then
`--unfreeze-text 4`).

| | |
|---|---|
| **Path** | [`checkpoints/best_all_tasks/`](checkpoints/best_all_tasks/) |
| **Weights** | `checkpoints/best_all_tasks/best.pt` |
| **Curves** | `checkpoints/best_all_tasks/curves.png` |
| **History** | `checkpoints/best_all_tasks/history.jsonl` |
| **Best val accuracy** | **~0.692** (~69%) |
| **Vision** | SigLIP `google/siglip-base-patch16-224` (P=196, d_v=768) |

All other experiment runs and the VQA annotation download were **deleted** to
save disk. Re-download VQA if you want to train again.

`data/`, `cache/`, and `checkpoints/` are **gitignored** (too large for git).
Copy `checkpoints/best_all_tasks/best.pt` to S3 or a release if you need it
off this box.

---

## The architecture

Both towers stay **frozen by default**. The bridge (projector + cross-attention)
is always trained; the decision head/scorer can be frozen in stage 1
(`--freeze-head`) and the last text-encoder layers can be unlocked in stage 2
(`--unfreeze-text N`).

```
image ──► SigLIP / CLIP (frozen) ──► [P, d_v] ──► projector ──► [P, d]
                                                                  │
                                                            cross-attention
                                                                  │
text  ──► ModernBERT (frozen-ish) ──► [CLS] question [SEP] [MASK]opt0 [MASK]opt1 [SEP] caption [SEP]
                                                       │
                                              gather at markers
                                                       │
                                              scorer: Linear(d, 1)
                                                       │
                                        mask ─► temperature ─► softmax
```

The scorer is unchanged from text Laya — one vector in, one number out, applied
per marker. K never enters a weight matrix, so the label set stays open-ended.

Frozen vision is extracted once and cached; training never re-runs the tower.

---

## Data

### Still on this machine

| asset | path | notes |
|---|---|---|
| SigLIP feature cache | `cache/siglip_train2014.{npy,json}` | ~25 GB — matches `best.pt` |
| Best checkpoint | `checkpoints/best_all_tasks/` | see above |

### Removed

- **All of `data/`** — VQA annotations and COCO train2014 images deleted after training
- Other experiment checkpoints (`noul_*`, `score_*`, `choice_*`, `mm_laya_v1`, …)

### To train again

```bash
python3 scripts/download_data.py --root data          # VQA + COCO (~13 GB+)
python3 scripts/download_data.py --root data --val    # optional val JSON
# reuse cache/siglip_train2014 if present, else re-extract
```

Train split (when VQA is present) by `answer_type`:

| subset | ~train count | primitive |
|---|---|---|
| yes/no | ~167k | `noul` |
| number | ~58k | `score` |
| other | ~219k | `choice` |

### Future datasets (not on disk — next experiments)

| dataset | ~size | fits | why later |
|---|---|---|---|
| SNLI-VE | ~565k | `choice` | compositional entailment |
| Visual7W | ~327k Q | `choice` | clean 4-option MC |
| MMHS150K | ~150k | `noul` | social multimodal |
| NLVR2 | ~107k | `noul` | hard compositional yes/no |
| A-OKVQA | ~25k | `choice` | knowledge-heavy MC |
| MM-IMDb | ~25k | multi-`noul` | multi-label |
| Hateful Memes | 10k | **held-out eval only** | fusion stress test |

---

## Experiment summary (g5.2xlarge)

Ladder: stage-1 `--freeze-head` (high bridge LR / CE-heavy variants) → stage-2
`--unfreeze-text 4`, early-stop at val ≥ 0.70 when hit.

| family | best val | kept? |
|---|---|---|
| noul only | ~0.52 | deleted |
| score only | ~0.55 | deleted |
| choice only | ~0.90 (likely leakage) | deleted |
| **all tasks (s2)** | **~0.69** | **`checkpoints/best_all_tasks/`** |

Re-run the ladder after restoring VQA:

```bash
python3 scripts/run_experiments.py --list
bash scripts/run_experiments.sh --all
```

---

## Evaluation

Prefer **coverage at fixed precision** plus ECE, not accuracy alone. After
re-downloading VQA, run `scripts/baseline_text_only.py` before trusting a
multimodal number.

---

## Compute

Primary box: **`g5.2xlarge` (A10G)**. Also fits 2×T4 if you cache SigLIP at
224px and keep bulk data on fast disk / NVMe.

---

## Run it

```bash
# 1. data
python3 scripts/download_data.py --root data

# 2. features (skip if cache/siglip_train2014.* already present)
python3 scripts/extract_features.py \
    --images data/coco/train2014 --out cache/siglip_train2014 \
    --model google/siglip-base-patch16-224 --batch 64

# 3. baseline floor
python3 scripts/baseline_text_only.py --vqa data/vqa --limit 4000

# 4. train (example: joint + early stop)
python3 scripts/train.py --data-root data --features cache/siglip_train2014 \
    --kinds noul score choice --freeze-head --early-stop-acc 0.70 --curves
```

Useful flags: `--limit`, `--kinds`, `--freeze-head`, `--unfreeze-text 4`,
`--init-ckpt`, `--early-stop-acc`, `--patience`, `--fresh`.

```bash
python3 scripts/plot_curves.py --history checkpoints/best_all_tasks/history.jsonl
```

## Setup

```bash
git clone https://github.com/<you>/laya-multimodal && cd laya-multimodal
pip install -r requirements.txt     # install torch first, matched to your CUDA
```

Needs `laya` + a GPU. Cloud: [`aws/README.md`](aws/README.md).
Upstream: `git clone https://github.com/NandhaKishorM/laya`

## Layout

```
src/                              model, data loaders
scripts/                          download / extract / train / experiments
notebooks/                        interactive
aws/                              g5 spot helpers
data/                             empty (gitignored) — re-download to train
cache/siglip_train2014.*          features (gitignored)
checkpoints/best_all_tasks/       **kept best joint run** (gitignored)
```

## Status

End-to-end training completed on **`g5.2xlarge`**. Disk cleaned: **no raw
datasets** under `data/`. Kept SigLIP cache + **`checkpoints/best_all_tasks/`**
(val ≈ **0.692**). Re-download data to continue experiments.

## Credits

Builds on [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations
(Apache 2.0). This repo is an experiment around it, not a fork.
