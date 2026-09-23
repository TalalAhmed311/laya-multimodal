#!/usr/bin/env python3
"""Rung 1 of the ablation ladder: how far does text alone get you?

Runs stock Laya on the VQA questions with NO image and reports accuracy, ECE
and coverage-at-threshold. This is the floor every multimodal number has to
beat, and it is worth an afternoon before building anything.

VQA v2 was specifically constructed to balance language priors — each question
is paired with two images that give opposite answers — so a text-only model
should land near chance on yes/no. If it lands well above that, something in the
item construction is leaking the answer and you want to know now, not after a
training run.

    python3 scripts/baseline_text_only.py --vqa data/vqa --limit 4000
"""
import argparse, math, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def ece(conf, correct, bins=15):
    if not conf:
        return float("nan")
    e, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i in range(n) if lo < conf[i] <= hi]
        if sel:
            e += (len(sel) / n) * abs(sum(conf[i] for i in sel) / len(sel)
                                      - sum(correct[i] for i in sel) / len(sel))
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa", default="data/vqa")
    ap.add_argument("--split", default="train2014")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--repo", default="convaiinnovations/laya")
    ap.add_argument("--kind", default="noul", choices=["noul", "score", "choice"])
    a = ap.parse_args()

    import torch
    from src.model import load_laya_backbone
    from src.data import build_items, collate
    from laya.common import confidence_from_probs, temp_bucket

    model, tok, cfg = load_laya_backbone(a.repo)
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(dev).eval()
    print("device:", dev)

    items = build_items(a.vqa, tok, cfg, split=a.split,
                        kinds=(a.kind,), limit=a.limit)
    if not items:
        sys.exit("no items built — check --vqa path and --kind")

    temps = cfg.get("temperature", [1.0, 1.0, 1.0])
    tbo = cfg.get("temperature_by_options", {})

    confs, corrs, n = [], [], 0
    with torch.no_grad():
        for s in range(0, len(items), 32):
            chunk = items[s:s + 32]
            b = collate(chunk, tok.pad_token_id)
            logits, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                              b["marker_pos"].to(dev), b["marker_mask"].to(dev),
                              b["qtype"].to(dev))
            logits = logits.float().cpu()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                t = tbo.get(temp_bucket(it["qtype"], k), temps[it["qtype"]])
                z = logits[r, :k] / max(1e-3, float(t))
                p = torch.softmax(z, -1).numpy()
                pred = int(p.argmax())
                gold = int(max(range(k), key=lambda i: it["target"][i]))
                corrs.append(1.0 if pred == gold else 0.0)
                confs.append(confidence_from_probs(p, k))
                n += 1
            sys.stdout.write("\r  %d/%d" % (n, len(items))); sys.stdout.flush()
    print()

    acc = sum(corrs) / len(corrs)
    golds = [int(max(range(len(i["target"])), key=lambda j: i["target"][j])) for i in items]
    base = max(golds.count(v) for v in set(golds)) / len(golds)
    K = len(items[0]["markers"])

    print("\n" + "=" * 68)
    print("  RUNG 1 — TEXT ONLY, NO IMAGE   (%s, n=%d)" % (a.kind, len(corrs)))
    print("=" * 68)
    print("  accuracy           %.4f" % acc)
    print("  majority baseline  %.4f" % base)
    print("  random (1/K)       %.4f" % (1.0 / K))
    print("  ECE                %.4f" % ece(confs, corrs))
    print("  mean confidence    %.4f" % (sum(confs) / len(confs)))

    print("\n  coverage at threshold")
    print("    %-10s %9s %10s" % ("threshold", "coverage", "accuracy"))
    for t in (0.0, 0.3, 0.5, 0.7, 0.9):
        sel = [i for i in range(len(confs)) if confs[i] >= t]
        if sel:
            print("    %-10.2f %8.1f%% %10.4f"
                  % (t, 100 * len(sel) / len(confs),
                     sum(corrs[i] for i in sel) / len(sel)))

    print("\n  READ THIS AS")
    if acc > base + 0.10:
        print("    Text alone beats the majority baseline by >10 points. On VQA v2,")
        print("    which is explicitly balanced against language priors, that is a")
        print("    RED FLAG — check item construction for answer leakage before")
        print("    interpreting any multimodal gain.")
    else:
        print("    Text alone is near the baseline, which is what a balanced dataset")
        print("    should give. This is your floor: every point above it is the bridge")
        print("    doing real work.")


if __name__ == "__main__":
    main()
