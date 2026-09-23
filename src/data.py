"""VQA v2 -> Laya typed-decision items.

The reason VQA v2 anchors this project: every question carries **10 human
answers**. That is a real answer distribution, not a hard label, and RLCD trains
against distributions under a strictly proper scoring rule. Most datasets force
you to synthesise soft targets from a teacher model; here they come from human
disagreement for free.

    8 annotators said "yes", 2 said "no"  ->  target [0.2, 0.8]

Mapping to the three primitives:

    answer_type == "yes/no"   ->  noul    ~38% of the corpus (~440k questions)
    answer_type == "number"   ->  score   bucketed 0/1/2/3+
    answer_type == "other"    ->  choice  needs candidate options (see below)

Tokenisation goes through laya.common.build_sequence — the *same* function the
runtime uses — so train/serve skew is impossible by construction.

Files expected (from visualqa.org):
    data/vqa/v2_OpenEnded_mscoco_train2014_questions.json
    data/vqa/v2_mscoco_train2014_annotations.json
"""
import json, os, random, re
from collections import Counter

from laya.common import QTYPES, build_sequence, render_options

NUM_LEVELS = ["0", "1", "2", "3 or more"]


# ------------------------------------------------------------------ targets
def noul_target(answers):
    """10 annotator strings -> [p(false), p(true)]. None if not a clean yes/no."""
    yes = sum(1 for a in answers if a == "yes")
    no = sum(1 for a in answers if a == "no")
    tot = yes + no
    if tot < 6:                      # too muddy to call a yes/no question
        return None
    return [no / tot, yes / tot]


def score_target(answers):
    """10 annotator strings -> distribution over NUM_LEVELS. None if not numeric."""
    hits = [0] * len(NUM_LEVELS)
    n = 0
    for a in answers:
        if not a.isdigit():
            continue
        v = int(a)
        hits[min(v, 3)] += 1
        n += 1
    if n < 6:
        return None
    return [h / n for h in hits]


def choice_target(answers, options):
    """10 annotator strings -> distribution over `options`. None if uncovered."""
    c = Counter(answers)
    hits = [c.get(o, 0) for o in options]
    tot = sum(hits)
    if tot < 6:
        return None
    return [h / tot for h in hits]


# ------------------------------------------------------------------ schemas
def make_question(qtext, kind, options=None):
    """Render a VQA question as a Laya typed question.

    The VQA question becomes the *instructions*. The image is the state and
    arrives through cross-attention, so the text state is empty or just OCR.
    """
    q = qtext.strip().rstrip("?") + "?"
    if kind == "noul":
        return {"type": "noul", "instructions": q}
    if kind == "score":
        return {"type": "score", "instructions": q, "criteria": NUM_LEVELS}
    return {"type": "choice", "instructions": q,
            "criteria": {o: None for o in options}}


# ------------------------------------------------------------------ loading
def load_vqa(root, split="train2014", limit=None):
    qf = os.path.join(root, "v2_OpenEnded_mscoco_%s_questions.json" % split)
    af = os.path.join(root, "v2_mscoco_%s_annotations.json" % split)
    for p in (qf, af):
        if not os.path.exists(p):
            raise FileNotFoundError(
                "%s not found. Download VQA v2 from https://visualqa.org/download.html "
                "into %s" % (os.path.basename(p), root))
    qs = {q["question_id"]: q for q in json.load(open(qf))["questions"]}
    annos = json.load(open(af))["annotations"]
    if limit:
        annos = annos[:limit]
    return qs, annos


def top_answers(annos, n=3000):
    """Most frequent answers — the candidate pool for `choice` distractors."""
    c = Counter()
    for a in annos:
        c[a["multiple_choice_answer"].strip().lower()] += 1
    return [w for w, _ in c.most_common(n)]


def _cache_key(root, cfg, split, kinds, n_choice_opts, limit, seed):
    import hashlib
    h = hashlib.sha1(repr((os.path.abspath(root), split, tuple(sorted(kinds)),
                           n_choice_opts, limit, seed,
                           cfg.get("max_len"), cfg.get("head_max_len"))).encode())
    return "items_%s_%s.pt" % (split, h.hexdigest()[:10])


def build_items(root, tok, cfg, split="train2014", kinds=("noul",),
                n_choice_opts=4, limit=None, seed=0, verbose=True, cache_dir="cache"):
    """-> list of dicts ready for collate_items, each carrying `image_id`.

    Tokenising 1.1M questions takes ~140s, and nothing about it changes between
    runs, so the result is cached on disk keyed by every argument that affects
    it. Pass cache_dir=None to force a rebuild.
    """
    if cache_dir:
        import torch as _t
        os.makedirs(cache_dir, exist_ok=True)
        cpath = os.path.join(cache_dir, _cache_key(root, cfg, split, kinds,
                                                   n_choice_opts, limit, seed))
        if os.path.exists(cpath):
            items = _t.load(cpath, weights_only=False)
            if verbose:
                print("loaded %d items from cache (%s)" % (len(items), os.path.basename(cpath)))
            return items

    rng = random.Random(seed)
    qs, annos = load_vqa(root, split, limit)
    pool = top_answers(annos) if "choice" in kinds else []
    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)

    items, skipped = [], Counter()
    for a in annos:
        atype = a["answer_type"]                      # yes/no | number | other
        kind = {"yes/no": "noul", "number": "score", "other": "choice"}[atype]
        if kind not in kinds:
            skipped["kind_filtered"] += 1
            continue

        answers = [x["answer"].strip().lower() for x in a["answers"]]
        options = None

        if kind == "noul":
            target = noul_target(answers)
        elif kind == "score":
            target = score_target(answers)
        else:
            gold = a["multiple_choice_answer"].strip().lower()
            distract = [w for w in rng.sample(pool, n_choice_opts * 3)
                        if w != gold][:n_choice_opts - 1]
            options = [gold] + distract
            rng.shuffle(options)
            target = choice_target(answers, options)

        if target is None:
            skipped["ambiguous_answers"] += 1
            continue

        qdef = make_question(qs[a["question_id"]]["question"], kind, options)
        q_int = {"t": qdef["type"], "ins": qdef["instructions"],
                 "crit": qdef.get("criteria")}
        seq, markers = build_sequence(tok, {}, q_int, max_len, head_max_len)
        if len(markers) != len(render_options(q_int)):
            skipped["option_truncated"] += 1
            continue
        if len(markers) != len(target):
            skipped["target_mismatch"] += 1
            continue

        items.append({
            "ids": seq, "markers": markers, "qtype": QTYPES[qdef["type"]],
            "target": target, "label": target.index(max(target)),
            "image_id": a["image_id"], "question_id": a["question_id"],
        })

    if verbose:
        print("built %d items from %d annotations" % (len(items), len(annos)))
        for k, v in skipped.most_common():
            print("   skipped %-18s %d" % (k, v))
        by = Counter(i["qtype"] for i in items)
        names = {v: k for k, v in QTYPES.items()}
        for qt, n in sorted(by.items()):
            print("   %-6s %d" % (names[qt], n))
        soft = sum(1 for i in items if max(i["target"]) < 0.95)
        print("   %d/%d (%.0f%%) have genuinely soft targets — annotators disagreed"
              % (soft, len(items), 100 * soft / max(1, len(items))))
        print("   ~%.1f GB held in RAM as python dicts" % (len(items) * 2300 / 1e9))
    if cache_dir:
        import torch as _t
        _t.save(items, cpath)
        if verbose:
            print("   cached -> %s" % cpath)
    return items


# ------------------------------------------------------------------ batching
class FeatureStore:
    """Memory-mapped image features. Opening costs nothing; rows page in on use.

    A dict of tensors from torch.load lands entirely in RAM — fine on a
    workstation, instant OOM on Colab's ~12 GB. This reads the .npy produced by
    scripts/extract_features.py without materialising it.
    """

    def __init__(self, stem):
        import json as _json
        import numpy as np
        stem = stem[:-4] if stem.endswith(".npy") else stem
        meta = _json.load(open(stem + ".json"))
        self.arr = np.load(stem + ".npy", mmap_mode="r")
        self.index = {int(v): k for k, v in enumerate(meta["image_ids"])}
        self.P, self.d_v = meta["P"], meta["d_v"]
        self.model = meta.get("model", "?")
        print("features: %d images | P=%d d_v=%d | %s (memmapped, not in RAM)"
              % (len(self.index), self.P, self.d_v, self.model))
        self._probe_speed(stem)

    def _probe_speed(self, stem):
        """Training does ~64 random reads per step. On NVMe that is ~10ms; on an
        EBS gp3 baseline volume it is ~150ms, which swamps the ~150ms of compute
        and silently halves throughput. Measure rather than assume."""
        import random as _r
        import time as _t
        import numpy as _np
        n = min(64, len(self.index))
        if n < 8:
            return
        rows = _r.sample(range(len(self.index)), n)
        t0 = _t.time()
        for i in rows:
            _np.asarray(self.arr[i]).sum()
        ms = (_t.time() - t0) * 1000
        mb = n * self.P * self.d_v * 2 / 1e6
        print("  random-read probe: %d rows (%.1f MB) in %.0f ms -> %.0f MB/s"
              % (n, mb, ms, mb / max(1e-6, ms / 1000)))
        if ms > 60:
            print("  !! SLOW. At ~%.0f ms per batch this dominates the ~150 ms of" % ms)
            print("     compute. Move the cache to instance-store NVMe (/mnt/data),")
            print("     not EBS. See aws/00_bootstrap.sh.")

    def __contains__(self, iid):
        return int(iid) in self.index

    def __len__(self):
        return len(self.index)

    def get(self, iid):
        import numpy as np
        import torch
        # memmap rows are read-only; copy so torch does not warn / UB on write
        return torch.from_numpy(np.array(self.arr[self.index[int(iid)]], copy=True))


def collate(batch, pad_id, feats=None):
    """laya.common.collate_items + the image features for each row.

    `feats` is a FeatureStore. Rows sharing an image each get a copy here;
    dedupe upstream if you want the split-state saving at training time too.
    """
    import torch
    from laya.common import collate_items

    b = collate_items([batch], pad_id)
    if feats is not None:
        ids = [it["image_id"] for it in batch]
        b["image_feats"] = torch.stack([feats.get(i) for i in ids])
        b["image_ids"] = ids
    return b


def split_items(items, val_frac=0.05, seed=0):
    """Split by IMAGE, not by question — questions about one image must not
    straddle the boundary or your val set is contaminated."""
    rng = random.Random(seed)
    imgs = sorted({i["image_id"] for i in items})
    rng.shuffle(imgs)
    cut = int(len(imgs) * (1 - val_frac))
    train_imgs = set(imgs[:cut])
    tr = [i for i in items if i["image_id"] in train_imgs]
    va = [i for i in items if i["image_id"] not in train_imgs]
    print("split by image: %d train / %d val items  (%d / %d images)"
          % (len(tr), len(va), cut, len(imgs) - cut))
    return tr, va
