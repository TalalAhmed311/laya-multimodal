#!/usr/bin/env python3
"""Step 2 of 2 — RLCD training for the multimodal bridge.

Same recipe as text Laya: a GRPO-style policy gradient against a strictly
proper scoring rule, with a full-weight soft cross-entropy anchor. The CE term
does the learning; the RL term shapes calibration. Pure policy gradient at this
data scale is far too noisy on its own.

Only the bridge trains — a projector and two cross-attention blocks, ~30M
parameters. Both towers stay frozen, and the image features are already cached,
so the vision tower never runs here at all.

    python3 scripts/train.py --data-root data --features cache/siglip_train2014

Resume is automatic: re-run the same command after an interruption and it picks
up from latest.pt at the exact item it stopped on.
"""
import argparse, json, math, os, random, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from laya.common import confidence_from_probs, proper_reward
from src.data import FeatureStore, build_items, collate, split_items
from src.model import build


# ----------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--data-root", default="data")
    d.add_argument("--features", default="cache/siglip_train2014", help=".npy/.json stem")
    d.add_argument("--split", default="train2014")
    d.add_argument("--kinds", nargs="+", default=["noul"],
                   choices=["noul", "score", "choice"],
                   help="which primitives to train on; start with noul")
    d.add_argument("--limit", type=int, default=0, help="cap annotations, for a smoke test")
    d.add_argument("--val-frac", type=float, default=0.05)
    d.add_argument("--item-cache", default="cache",
                   help="where to cache tokenised items (~140s to rebuild at 1.1M)")
    d.add_argument("--rebuild-items", action="store_true", help="ignore the item cache")

    m = p.add_argument_group("model")
    m.add_argument("--repo", default="convaiinnovations/laya")
    m.add_argument("--cross-layers", type=int, default=2)
    m.add_argument("--unfreeze-text", type=int, default=0,
                   help="stage 2: unfreeze the last N encoder layers (0 = keep frozen)")
    m.add_argument("--freeze-head", action="store_true",
                   help="stage 1: freeze Laya head+scorer so only the bridge learns")
    m.add_argument("--init-ckpt", default=None,
                   help="load projector/cross/head/scorer from a prior best.pt (fresh optim)")

    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch", type=int, default=16, help="micro-batch per step")
    t.add_argument("--accum", type=int, default=2, help="gradient accumulation")
    t.add_argument("--group", type=int, default=4, help="GRPO group size")
    t.add_argument("--lr-bridge", type=float, default=2e-4)
    t.add_argument("--lr-head", type=float, default=1e-4)
    t.add_argument("--sigma-start", type=float, default=0.4)
    t.add_argument("--sigma-end", type=float, default=0.1)
    t.add_argument("--w-sph", type=float, default=0.75)
    t.add_argument("--w-rps", type=float, default=1.0)
    t.add_argument("--w-ce", type=float, default=1.0)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=1234)
    t.add_argument("--early-stop-acc", type=float, default=0.0,
                   help="stop when periodic val accuracy reaches this (0 = disabled)")
    t.add_argument("--patience", type=int, default=0,
                   help="stop after N evals with no val improvement (0 = disabled)")

    o = p.add_argument_group("io")
    o.add_argument("--out", default="checkpoints/mm_laya_v1")
    o.add_argument("--log-every", type=int, default=50)
    o.add_argument("--save-every", type=int, default=500)
    o.add_argument("--eval-every", type=int, default=2000)
    o.add_argument("--eval-n", type=int, default=2000, help="val subset for periodic eval")
    o.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    o.add_argument("--curves", action="store_true", help="render curves.png at the end")
    return p.parse_args()

# ----------------------------------------------------------------- utilities
def pick_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        cap = torch.cuda.get_device_capability(0)
        # Ampere+ (A10G is 8.6) has bf16: wider range, and no loss-scale tuning.
        # Turing (T4, 7.5) does not, so fp16 + GradScaler there.
        amp = torch.bfloat16 if cap[0] >= 8 else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        print("device: %s  (sm_%d%d)  autocast=%s  scaler=%s"
              % (torch.cuda.get_device_name(0), cap[0], cap[1],
                 str(amp).split(".")[-1], amp is torch.float16))
        return dev, amp
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device: %s (no CUDA — this will be slow)" % dev)
    return dev, torch.float32


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


def gold_index(item):
    return int(max(range(len(item["target"])), key=lambda j: item["target"][j]))


# ----------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(mm, items, feats, tok, dev, amp, batch=64, temps=None):
    mm.eval()
    Z, keep = [], []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        b = collate(chunk, tok.pad_token_id, feats)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=amp, enabled=(dev.type == "cuda")):
            lg = mm(b["input_ids"], b["attention_mask"], b["marker_pos"],
                    b["marker_mask"], b["qtype"], b["image_feats"].float())
        lg = lg.float().cpu()
        for r, it in enumerate(chunk):
            Z.append(lg[r, :len(it["markers"])]); keep.append(it)
    accs, confs = [], []
    for z, it in zip(Z, keep):
        t = 1.0 if temps is None else temps[it["qtype"]]
        p = torch.softmax(z / max(1e-3, float(t)), -1).numpy()
        accs.append(1.0 if int(p.argmax()) == gold_index(it) else 0.0)
        confs.append(confidence_from_probs(p, len(p)))
    mm.train()
    return accs, confs, Z, keep


def fit_temperatures(Z, keep):
    """One scalar per question type, minimising NLL against the soft target."""
    temps = [1.0, 1.0, 1.0]
    for qt in range(3):
        sel = [(z, it["target"]) for z, it in zip(Z, keep) if it["qtype"] == qt]
        if len(sel) < 25:
            continue
        best, bt = float("inf"), 1.0
        for i in range(70):                       # geometric grid, 0.2 .. ~11
            t = 0.2 * (1.06 ** i)
            tot = sum(-(torch.tensor(tg) * torch.log_softmax(z / t, -1)).sum().item()
                      for z, tg in sel)
            if tot < best:
                best, bt = tot, t
        temps[qt] = round(bt, 4)
    return temps


# ----------------------------------------------------------------- checkpoints
def trainable_state(mm):
    return {"projector": mm.projector.state_dict(), "cross": mm.cross.state_dict(),
            "head": mm.head.state_dict(), "scorer": mm.scorer.state_dict()}


def save_ckpt(path, mm, opt, sched, scaler, epoch, step_in_epoch, gstep, args, extra=None):
    """Temp file then rename — an interruption mid-write must not truncate."""
    tmp = path + ".tmp"
    torch.save({**trainable_state(mm), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "step_in_epoch": step_in_epoch, "global_step": gstep,
                "d_v": mm.d_v, "cross_layers": len(mm.cross),
                "args": vars(args), **(extra or {})}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, mm, opt, sched, scaler, dev):
    ck = torch.load(path, map_location=dev)
    mm.projector.load_state_dict(ck["projector"]); mm.cross.load_state_dict(ck["cross"])
    mm.head.load_state_dict(ck["head"]);           mm.scorer.load_state_dict(ck["scorer"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"])
    return ck["epoch"], ck["step_in_epoch"], ck["global_step"]


# ----------------------------------------------------------------- main
def main():
    a = parse_args()
    dev, amp = pick_device()
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    LATEST, BEST, HIST = (os.path.join(a.out, f) for f in ("latest.pt", "best.pt", "history.jsonl"))

    # ---- model ----
    print("\n== model ==")
    mm, tok, cfg = build(repo=a.repo, d_v=768, cross_layers=a.cross_layers, freeze_text=True)

    # ---- data ----
    print("\n== data ==")
    feats = FeatureStore(a.features)
    if feats.d_v != mm.d_v:                 # rebuild rather than fail 20 min into training
        print("  d_v mismatch (%d vs %d) — rebuilding the model" % (feats.d_v, mm.d_v))
        mm, tok, cfg = build(repo=a.repo, d_v=feats.d_v,
                             cross_layers=a.cross_layers, freeze_text=True)
    mm = mm.to(dev)
    if a.init_ckpt:
        ck = torch.load(a.init_ckpt, map_location=dev)
        mm.projector.load_state_dict(ck["projector"])
        mm.cross.load_state_dict(ck["cross"])
        mm.head.load_state_dict(ck["head"])
        mm.scorer.load_state_dict(ck["scorer"])
        print("loaded weights from %s (optimiser starts fresh)" % a.init_ckpt)
    if a.freeze_head:
        for p in list(mm.head.parameters()) + list(mm.scorer.parameters()):
            p.requires_grad = False
        print("froze head+scorer — bridge only")
    if a.unfreeze_text:
        mm.unfreeze_text(last_n=a.unfreeze_text)
    mm.trainable_report()

    items = build_items(os.path.join(a.data_root, "vqa"), tok, cfg, split=a.split,
                        kinds=tuple(a.kinds), limit=(a.limit or None), seed=a.seed,
                        cache_dir=(None if a.rebuild_items else a.item_cache))
    items = [i for i in items if i["image_id"] in feats]
    if not items:
        sys.exit("no items have cached features — check --features covers these images")
    train_items, val_items = split_items(items, val_frac=a.val_frac, seed=a.seed)

    # ---- optimiser ----
    bridge = list(mm.projector.parameters()) + list(mm.cross.parameters())
    headp = [p for p in list(mm.head.parameters()) + list(mm.scorer.parameters())
             if p.requires_grad]
    enc = [p for p in mm.encoder.parameters() if p.requires_grad]
    groups = [{"params": bridge, "lr": a.lr_bridge}]
    if headp:
        groups.append({"params": headp, "lr": a.lr_head})
    if enc:
        groups.append({"params": enc, "lr": a.lr_head / 4})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = max(1, (len(train_items) // (a.batch * a.accum)) * a.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda" and amp is torch.float16))
    print("\n%d optimizer steps planned (%d epochs, effective batch %d)"
          % (steps, a.epochs, a.batch * a.accum))
    if a.early_stop_acc > 0:
        print("early-stop when val acc >= %.2f" % a.early_stop_acc)
    if a.patience > 0:
        print("early-stop patience: %d evals without improvement" % a.patience)

    # ---- resume ----
    e0, s0, gstep, best_val = 0, 0, 0, -1.0
    bad_evals = 0
    stop_training = False
    if os.path.exists(LATEST) and not a.fresh and not a.init_ckpt:
        e0, s0, gstep = load_ckpt(LATEST, mm, opt, sched, scaler, dev)
        print("RESUMED from epoch %d item %d (global step %d)" % (e0, s0, gstep))
    else:
        open(HIST, "w").close()
        print("fresh run")
    def log(rec):
        with open(HIST, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- train ----
    print("\n== training ==")
    t0 = time.time()
    for epoch in range(e0, a.epochs):
        if stop_training:
            break
        mm.train()
        # Seeded per epoch, so resuming mid-epoch lands on the items a fresh run
        # would have seen rather than a different permutation.
        order = list(range(len(train_items)))
        random.Random(a.seed + epoch).shuffle(order)
        begin = s0 if epoch == e0 else 0
        sigma = a.sigma_start + (a.sigma_end - a.sigma_start) * (epoch / max(1, a.epochs - 1))
        opt.zero_grad(set_to_none=True)
        run, nb, accum = 0.0, 0, 0

        for s in range(begin, len(order), a.batch):
            chunk = [train_items[i] for i in order[s:s + a.batch]]
            if not chunk:
                continue
            b = collate(chunk, tok.pad_token_id, feats)
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            mask, target = b["marker_mask"], b["target"]

            with torch.autocast("cuda", dtype=amp, enabled=(dev.type == "cuda")):
                logits = mm(b["input_ids"], b["attention_mask"], b["marker_pos"],
                            mask, b["qtype"], b["image_feats"].float())
            logits = logits.float()
            k = mask.sum(-1, keepdim=True).float()

            # 1. a group of G perturbed distributions. The zero-mean projection
            #    matters: softmax is shift-invariant, so un-projected noise wastes
            #    part of every sample on a direction that cannot change the output.
            eps = torch.randn((a.group,) + logits.shape, device=dev) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. score them; the group's own mean is the baseline (no value network)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), b["qtype"], mask,
                                  w_sph=a.w_sph, w_rps=a.w_rps)
                adv = r - r.mean(0, keepdim=True)      # reduce over the GROUP axis
                adv = adv / (adv.std() + 1e-6)

            # 3. REINFORCE on an isotropic Gaussian policy over logits, + CE anchor
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + a.w_ce * loss_ce) / a.accum

            scaler.scale(loss).backward(); accum += 1
            if accum % a.accum == 0 or (s + a.batch) >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in mm.parameters() if p.requires_grad], a.clip)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)

            run += loss.item() * a.accum; nb += 1; gstep += 1

            if gstep % a.log_every == 0:
                rec = {"t": round(time.time() - t0, 1), "epoch": epoch, "step": gstep,
                       "loss": round(run / max(1, nb), 4),
                       "loss_rl": round(loss_rl.item(), 4), "loss_ce": round(loss_ce.item(), 4),
                       "reward": round(r.mean().item(), 4),
                       "lr": sched.get_last_lr()[0], "sigma": round(sigma, 3)}
                log(rec)
                print("  ep%d step%-7d loss %.4f  ce %.4f  rl %.4f  reward %.3f  lr %.2e  %.0fs"
                      % (epoch, gstep, rec["loss"], rec["loss_ce"], rec["loss_rl"],
                         rec["reward"], rec["lr"], time.time() - t0), flush=True)

            if gstep % a.save_every == 0:
                save_ckpt(LATEST, mm, opt, sched, scaler, epoch, s + a.batch, gstep, a)

            if gstep % a.eval_every == 0:
                ac, cf, _, _ = evaluate(mm, val_items[:a.eval_n], feats, tok, dev, amp)
                acc, e_ = sum(ac) / len(ac), ece(cf, ac)
                improved = acc > best_val
                log({"t": round(time.time() - t0, 1), "epoch": epoch, "step": gstep,
                     "val_acc": round(acc, 4), "val_ece": round(e_, 4)})
                print("    val@%d  acc %.4f  ECE %.4f%s"
                      % (gstep, acc, e_, "  <- best" if improved else ""), flush=True)
                if improved:
                    best_val = acc
                    bad_evals = 0
                    save_ckpt(BEST, mm, opt, sched, scaler, epoch, s + a.batch, gstep, a,
                              extra={"val_acc": acc, "val_ece": e_})
                else:
                    bad_evals += 1

                if a.early_stop_acc > 0 and acc >= a.early_stop_acc:
                    print("EARLY STOP: val acc %.4f >= %.2f — keeping best.pt"
                          % (acc, a.early_stop_acc), flush=True)
                    log({"t": round(time.time() - t0, 1), "epoch": epoch, "step": gstep,
                         "early_stop": "target_acc", "val_acc": round(acc, 4)})
                    stop_training = True
                    break
                if a.patience > 0 and bad_evals >= a.patience:
                    print("EARLY STOP: no val improvement for %d evals (best %.4f)"
                          % (a.patience, best_val), flush=True)
                    log({"t": round(time.time() - t0, 1), "epoch": epoch, "step": gstep,
                         "early_stop": "patience", "best_val": round(best_val, 4)})
                    stop_training = True
                    break

        save_ckpt(LATEST, mm, opt, sched, scaler, epoch + 1, 0, gstep, a)
        print("== epoch %d done | avg loss %.4f | %.0fs ==" % (epoch, run / max(1, nb), time.time() - t0))
        s0 = 0

    # ---- final eval + calibration ----
    print("\n== final evaluation ==")
    if os.path.exists(BEST):
        load_ckpt(BEST, mm, opt, sched, scaler, dev)
        print("  loaded best.pt")
    accs, confs, Z, keep = evaluate(mm, val_items, feats, tok, dev, amp)
    print("  accuracy %.4f | ECE %.4f | mean conf %.4f"
          % (sum(accs) / len(accs), ece(confs, accs), sum(confs) / len(confs)))

    temps = fit_temperatures(Z, keep)
    accs_t, confs_t, _, _ = evaluate(mm, val_items, feats, tok, dev, amp, temps=temps)
    print("  fitted temperatures (choice, score, noul): %s" % temps)
    print("  ECE %.4f -> %.4f   (accuracy unchanged by design: %.4f)"
          % (ece(confs, accs), ece(confs_t, accs_t), sum(accs_t) / len(accs_t)))

    print("\n  %-10s %9s %10s" % ("threshold", "coverage", "accuracy"))
    for th in (0.0, 0.3, 0.5, 0.7, 0.9):
        sel = [i for i in range(len(confs_t)) if confs_t[i] >= th]
        if sel:
            print("  %-10.2f %8.1f%% %10.4f"
                  % (th, 100 * len(sel) / len(confs_t),
                     sum(accs_t[i] for i in sel) / len(sel)))

    final = os.path.join(a.out, "final.pt")
    save_ckpt(final, mm, opt, sched, scaler, a.epochs, 0, gstep, a,
              extra={"temperature": temps, "val_acc": sum(accs_t) / len(accs_t),
                     "val_ece": ece(confs_t, accs_t)})
    print("\nsaved %s" % final)
    print("compare against scripts/baseline_text_only.py — that is the floor.")

    if a.curves:
        os.system("%s %s --history %s --out %s" %
                  (sys.executable, os.path.join(os.path.dirname(__file__), "plot_curves.py"),
                   HIST, os.path.join(a.out, "curves.png")))


if __name__ == "__main__":
    main()
