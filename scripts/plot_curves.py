#!/usr/bin/env python3
"""Render training curves from history.jsonl.

Reads the file rather than in-memory state, so it works mid-run in another
shell, or after an interruption, without re-training anything.

    python3 scripts/plot_curves.py --history checkpoints/mm_laya_v1/history.jsonl
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="checkpoints/mm_laya_v1/history.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.history), "curves.png")

    import matplotlib
    matplotlib.use("Agg")                      # headless on a server
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for l in open(a.history) if l.strip()]
    tr = [r for r in rows if "loss" in r]
    va = [r for r in rows if "val_acc" in r]
    if not tr:
        print("no training points in %s yet" % a.history); return
    print("%d train points, %d val points" % (len(tr), len(va)))

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("multimodal Laya — bridge training", fontsize=12)
    S = [r["step"] for r in tr]

    ax[0, 0].plot(S, [r["loss"] for r in tr], lw=1)
    ax[0, 0].set_title("total loss")

    ax[0, 1].plot(S, [r["loss_ce"] for r in tr], lw=1, label="CE")
    ax[0, 1].plot(S, [r["loss_rl"] for r in tr], lw=1, label="RL")
    ax[0, 1].set_title("loss split — CE learns, RL calibrates")
    ax[0, 1].legend()

    ax[1, 0].plot(S, [r["reward"] for r in tr], lw=1, color="tab:green")
    ax[1, 0].set_title("proper-scoring reward (higher is better)")

    if va:
        a2 = ax[1, 1]
        a2.plot([r["step"] for r in va], [r["val_acc"] for r in va], "o-", label="accuracy")
        a2.set_ylabel("accuracy")
        a3 = a2.twinx()
        a3.plot([r["step"] for r in va], [r["val_ece"] for r in va], "s--",
                color="tab:red", label="ECE")
        a3.set_ylabel("ECE (lower better)")
        a2.set_title("validation")
        a2.legend(loc="lower left"); a3.legend(loc="upper right")
    else:
        ax[1, 1].text(.5, .5, "no validation points yet", ha="center", va="center")
        ax[1, 1].axis("off")

    for x in ax.flat:
        x.set_xlabel("step"); x.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)

    if va:
        print("\nAccuracy rising while ECE rises too means the model is getting more")
        print("right AND more overconfident — that is the temperature fit's job at")
        print("the end of training, not a reason to stop early.")


if __name__ == "__main__":
    main()
