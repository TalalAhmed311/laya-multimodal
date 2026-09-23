#!/usr/bin/env python3
"""Multi-experiment ladder aimed at 70–80% val accuracy.

Runs sequentially (one GPU). Each experiment writes its own checkpoint dir
under checkpoints/experiments/<name>/ with best.pt, curves.png, history.jsonl.

Does NOT start until you launch it. Dry-run with --list.

    # show plan
    python3 scripts/run_experiments.py --list

    # run everything (hours)
    python3 scripts/run_experiments.py --all

    # run one family
    python3 scripts/run_experiments.py --only noul
    python3 scripts/run_experiments.py --only noul_s1_bridge_highlr

Design (per task + one joint model):
  Stage 1  freeze head, high bridge LR, CE-heavy  → force image path to learn
  Stage 2  load stage-1 best, unfreeze text-4 + head, lower LR
  Variant  higher LR / more cross layers / CE-only (w_rps=0)

Early stop: val acc >= --early-stop-acc (default 0.70) OR patience exhausted.
best.pt is always kept.

Honest note: 70% on choice (4-way) is much harder than noul (2-way). The
runner still targets 0.70 everywhere; choice may stop on patience instead.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "scripts", "train.py")
EXP_ROOT = os.path.join(ROOT, "checkpoints", "experiments")


def _base(**over: Any) -> dict:
    """Shared defaults for every run."""
    d = dict(
        data_root="data",
        features="cache/siglip_train2014",
        epochs=5,
        batch=16,
        accum=2,
        group=4,
        eval_every=1000,
        eval_n=2000,
        save_every=500,
        log_every=50,
        early_stop_acc=0.70,
        patience=8,
        fresh=True,
        curves=True,
        seed=1234,
        cross_layers=2,
        w_ce=1.0,
        w_sph=0.75,
        w_rps=1.0,
        sigma_start=0.4,
        sigma_end=0.1,
    )
    d.update(over)
    return d


def build_experiments() -> list[dict]:
    """Ordered list. Stage-2 entries depend on stage-1 best.pt via init_from name."""
    exps: list[dict] = []

    def add(name: str, kinds: list[str], init_from: str | None = None, **kw):
        exps.append({
            "name": name,
            "kinds": kinds,
            "init_from": init_from,   # experiment name whose best.pt to load
            "args": _base(**kw),
        })

    # ---------- NOUL (yes/no) ----------
    add("noul_s1_bridge_highlr", ["noul"],
        freeze_head=True, lr_bridge=8e-4, lr_head=1e-4, w_ce=2.0, w_rps=0.5,
        epochs=4, early_stop_acc=0.70)
    add("noul_s1_bridge_ce_only", ["noul"],
        freeze_head=True, lr_bridge=1e-3, w_ce=3.0, w_rps=0.0, w_sph=0.0,
        sigma_start=0.05, sigma_end=0.05, epochs=4, early_stop_acc=0.70)
    add("noul_s1_cross4", ["noul"],
        freeze_head=True, cross_layers=4, lr_bridge=6e-4, w_ce=2.0,
        epochs=4, early_stop_acc=0.70)
    add("noul_s2_unfreeze4", ["noul"], init_from="noul_s1_bridge_highlr",
        unfreeze_text=4, lr_bridge=1e-4, lr_head=5e-5, w_ce=1.5,
        epochs=3, early_stop_acc=0.80, patience=6)
    add("noul_s2_from_ce", ["noul"], init_from="noul_s1_bridge_ce_only",
        unfreeze_text=4, lr_bridge=1e-4, lr_head=5e-5, w_ce=1.5,
        epochs=3, early_stop_acc=0.80, patience=6)

    # ---------- SCORE (number) ----------
    add("score_s1_bridge_highlr", ["score"],
        freeze_head=True, lr_bridge=8e-4, w_ce=2.0, w_rps=0.5,
        epochs=5, early_stop_acc=0.70, eval_n=1500)
    add("score_s1_bridge_ce_only", ["score"],
        freeze_head=True, lr_bridge=1e-3, w_ce=3.0, w_rps=0.0, w_sph=0.0,
        sigma_start=0.05, sigma_end=0.05, epochs=5, early_stop_acc=0.70)
    add("score_s2_unfreeze4", ["score"], init_from="score_s1_bridge_highlr",
        unfreeze_text=4, lr_bridge=1e-4, lr_head=5e-5, w_ce=1.5,
        epochs=4, early_stop_acc=0.75, patience=6)

    # ---------- CHOICE (other) — 4-way; 70% is ambitious ----------
    add("choice_s1_bridge_highlr", ["choice"],
        freeze_head=True, lr_bridge=8e-4, w_ce=2.0, w_rps=0.5,
        epochs=5, early_stop_acc=0.70, eval_n=2000, patience=10)
    add("choice_s1_bridge_ce_only", ["choice"],
        freeze_head=True, lr_bridge=1e-3, w_ce=3.0, w_rps=0.0, w_sph=0.0,
        sigma_start=0.05, sigma_end=0.05, epochs=5, early_stop_acc=0.70, patience=10)
    add("choice_s2_unfreeze4", ["choice"], init_from="choice_s1_bridge_highlr",
        unfreeze_text=4, lr_bridge=1e-4, lr_head=5e-5, w_ce=1.5,
        epochs=4, early_stop_acc=0.70, patience=8)

    # ---------- ALL TASKS (joint) ----------
    add("all_s1_bridge_highlr", ["noul", "score", "choice"],
        freeze_head=True, lr_bridge=6e-4, w_ce=2.0, w_rps=0.5,
        epochs=4, early_stop_acc=0.70, eval_n=3000, eval_every=1500, patience=8)
    add("all_s1_bridge_ce_only", ["noul", "score", "choice"],
        freeze_head=True, lr_bridge=8e-4, w_ce=3.0, w_rps=0.0, w_sph=0.0,
        sigma_start=0.05, sigma_end=0.05, epochs=4, early_stop_acc=0.70,
        eval_n=3000, eval_every=1500, patience=8)
    add("all_s2_unfreeze4", ["noul", "score", "choice"],
        init_from="all_s1_bridge_highlr",
        unfreeze_text=4, lr_bridge=1e-4, lr_head=5e-5, w_ce=1.5,
        epochs=3, early_stop_acc=0.75, eval_n=3000, eval_every=1500, patience=6)

    return exps


def to_cli(exp: dict) -> list[str]:
    a = exp["args"]
    out = os.path.join(EXP_ROOT, exp["name"])
    cmd = [
        sys.executable, TRAIN,
        "--data-root", a["data_root"],
        "--features", a["features"],
        "--kinds", *exp["kinds"],
        "--out", out,
        "--epochs", str(a["epochs"]),
        "--batch", str(a["batch"]),
        "--accum", str(a["accum"]),
        "--group", str(a["group"]),
        "--lr-bridge", str(a["lr_bridge"]),
        "--lr-head", str(a.get("lr_head", 1e-4)),
        "--w-ce", str(a["w_ce"]),
        "--w-sph", str(a["w_sph"]),
        "--w-rps", str(a["w_rps"]),
        "--sigma-start", str(a["sigma_start"]),
        "--sigma-end", str(a["sigma_end"]),
        "--cross-layers", str(a["cross_layers"]),
        "--eval-every", str(a["eval_every"]),
        "--eval-n", str(a["eval_n"]),
        "--save-every", str(a["save_every"]),
        "--log-every", str(a["log_every"]),
        "--early-stop-acc", str(a["early_stop_acc"]),
        "--patience", str(a["patience"]),
        "--seed", str(a["seed"]),
    ]
    if a.get("fresh"):
        cmd.append("--fresh")
    if a.get("curves"):
        cmd.append("--curves")
    if a.get("freeze_head"):
        cmd.append("--freeze-head")
    if a.get("unfreeze_text"):
        cmd += ["--unfreeze-text", str(a["unfreeze_text"])]
    if exp.get("init_from"):
        ckpt = os.path.join(EXP_ROOT, exp["init_from"], "best.pt")
        cmd += ["--init-ckpt", ckpt]
    return cmd


def filter_exps(exps: list[dict], only: str | None) -> list[dict]:
    if not only:
        return exps
    key = only.strip().lower()
    # exact name
    hit = [e for e in exps if e["name"] == key]
    if hit:
        # include dependency chain for stage-2
        names = {e["name"] for e in hit}
        for e in hit:
            if e.get("init_from"):
                names.add(e["init_from"])
        return [e for e in exps if e["name"] in names]
    # family prefix: noul / score / choice / all
    return [e for e in exps if e["name"].startswith(key + "_") or e["name"] == key]


def summarize_best(name: str) -> dict | None:
    hist = os.path.join(EXP_ROOT, name, "history.jsonl")
    best_pt = os.path.join(EXP_ROOT, name, "best.pt")
    if not os.path.exists(hist):
        return None
    best = -1.0
    last = None
    for line in open(hist):
        if not line.strip():
            continue
        r = json.loads(line)
        if "val_acc" in r:
            last = r
            best = max(best, r["val_acc"])
    return {
        "name": name,
        "best_val": best if best >= 0 else None,
        "last_val": last.get("val_acc") if last else None,
        "early_stop": last.get("early_stop") if last else None,
        "has_best_pt": os.path.exists(best_pt),
        "curves": os.path.exists(os.path.join(EXP_ROOT, name, "curves.png")),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print experiment plan and exit")
    ap.add_argument("--all", action="store_true", help="run the full ladder")
    ap.add_argument("--only", default=None,
                    help="run one name or a family: noul|score|choice|all")
    ap.add_argument("--dry-run", action="store_true", help="print commands, do not execute")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip experiments that already have best.pt")
    ap.add_argument("--python", default=sys.executable, help="python binary to use")
    a = ap.parse_args()

    exps = build_experiments()
    if a.only:
        exps = filter_exps(exps, a.only)
        if not exps:
            sys.exit("no experiments matched --only %r" % a.only)

    print("=" * 72)
    print("EXPERIMENT PLAN (%d runs)  early-stop@70%% (stage2 often @75–80%%)" % len(exps))
    print("=" * 72)
    for i, e in enumerate(exps, 1):
        deps = "  <- %s" % e["init_from"] if e.get("init_from") else ""
        flags = []
        if e["args"].get("freeze_head"):
            flags.append("freeze-head")
        if e["args"].get("unfreeze_text"):
            flags.append("unfreeze-%d" % e["args"]["unfreeze_text"])
        flags.append("lr_b=%.0e" % e["args"]["lr_bridge"])
        flags.append("w_ce=%s" % e["args"]["w_ce"])
        print("%2d. %-28s  kinds=%-20s  %s%s"
              % (i, e["name"], "+".join(e["kinds"]), " ".join(flags), deps))
    print("checkpoints -> %s/<name>/{best,final,latest}.pt + curves.png" % EXP_ROOT)
    print()

    if a.list and not a.all and not a.only:
        return
    if not a.all and not a.only:
        print("Nothing launched. Use --list, --all, or --only <name|family>.")
        print("Waiting for your heads-up before a long run is intentional.")
        return

    os.makedirs(EXP_ROOT, exist_ok=True)
    summary_path = os.path.join(EXP_ROOT, "summary.jsonl")
    results = []

    for e in exps:
        out_dir = os.path.join(EXP_ROOT, e["name"])
        best_pt = os.path.join(out_dir, "best.pt")
        if a.skip_done and os.path.exists(best_pt):
            print("SKIP (best.pt exists): %s" % e["name"])
            results.append(summarize_best(e["name"]))
            continue
        if e.get("init_from"):
            dep = os.path.join(EXP_ROOT, e["init_from"], "best.pt")
            if not os.path.exists(dep):
                print("SKIP %s — missing dependency %s" % (e["name"], dep))
                results.append({"name": e["name"], "error": "missing_init", "init_from": e["init_from"]})
                continue

        cmd = to_cli(e)
        cmd[0] = a.python
        log_path = os.path.join(EXP_ROOT, e["name"] + ".log")
        os.makedirs(out_dir, exist_ok=True)
        print("\n" + "#" * 72)
        print("# START %s" % e["name"])
        print("# log -> %s" % log_path)
        print("# cmd:", " ".join(cmd))
        print("#" * 72, flush=True)
        if a.dry_run:
            continue

        t0 = time.time()
        with open(log_path, "w") as logf:
            logf.write("CMD: %s\n\n" % " ".join(cmd))
            logf.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        rec = summarize_best(e["name"]) or {"name": e["name"]}
        rec["returncode"] = proc.returncode
        rec["seconds"] = round(dt, 1)
        results.append(rec)
        with open(summary_path, "a") as sf:
            sf.write(json.dumps(rec) + "\n")
        print("DONE %s  rc=%s  best_val=%s  %.0fs"
              % (e["name"], proc.returncode, rec.get("best_val"), dt), flush=True)
        if proc.returncode != 0:
            print("WARNING: non-zero exit — see %s" % log_path)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        if not r:
            continue
        print("  %-28s  best=%s  early=%s  best.pt=%s"
              % (r.get("name"), r.get("best_val"), r.get("early_stop"), r.get("has_best_pt")))
    print("wrote %s" % summary_path)


if __name__ == "__main__":
    main()
