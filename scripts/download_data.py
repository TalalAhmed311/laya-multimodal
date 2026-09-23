#!/usr/bin/env python3
"""Step 1 of 2 — fetch VQA v2 + COCO train2014.

    python3 scripts/download_data.py                    # everything
    python3 scripts/download_data.py --vqa-only         # annotations only (~200MB)
    python3 scripts/download_data.py --root /mnt/data   # put it on the NVMe

Downloads resume, so an interrupted 13GB COCO pull picks up where it stopped
rather than starting over. Uses curl/wget when available (better retry
behaviour on large files) and falls back to urllib.

After this:  python3 scripts/extract_features.py ...   then  scripts/train.py
"""
import argparse, os, shutil, subprocess, sys, time, urllib.request, zipfile

VQA = ["https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
       "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"]
VQA_VAL = ["https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
           "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"]
COCO = "http://images.cocodataset.org/zips/train2014.zip"


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return "%.1f%s" % (n, u)
        n /= 1024
    return "%.1fTB" % n


def fetch(url, dest):
    """Resumable download. Prefers curl/wget; urllib with a Range header otherwise."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("  have %s (%s)" % (os.path.basename(dest), human(os.path.getsize(dest))))
        return dest
    part = dest + ".part"
    for tool, cmd in (("curl", ["curl", "-fL", "-C", "-", "-o", part, url]),
                      ("wget", ["wget", "-c", "-O", part, url])):
        if shutil.which(tool):
            print("  %s <- %s" % (os.path.basename(dest), url))
            if subprocess.run(cmd).returncode == 0:
                os.replace(part, dest)
                return dest
            print("  %s failed, trying next" % tool)
    # urllib fallback with byte-range resume
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", "bytes=%d-" % have)
        print("  resuming at %s" % human(have))
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) + have
        t0, got = time.time(), have
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); got += len(chunk)
            if total:
                sys.stdout.write("\r    %s / %s  (%.0f%%)  %s/s   "
                                 % (human(got), human(total), 100*got/total,
                                    human(got/max(1e-9, time.time()-t0))))
                sys.stdout.flush()
    print()
    os.replace(part, dest)
    return dest


def unzip(path, into, marker=None):
    if marker and os.path.exists(os.path.join(into, marker)):
        print("  already extracted -> %s" % marker)
        return
    print("  unzipping %s" % os.path.basename(path))
    with zipfile.ZipFile(path) as z:
        z.extractall(into)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="where to put everything")
    ap.add_argument("--vqa-only", action="store_true")
    ap.add_argument("--coco-only", action="store_true")
    ap.add_argument("--val", action="store_true", help="also fetch the VQA val split")
    ap.add_argument("--keep-zips", action="store_true")
    a = ap.parse_args()

    vqa_dir = os.path.join(a.root, "vqa")
    coco_dir = os.path.join(a.root, "coco")
    os.makedirs(vqa_dir, exist_ok=True); os.makedirs(coco_dir, exist_ok=True)

    if not a.coco_only:
        print("\n== VQA v2 annotations ==")
        for url in VQA + (VQA_VAL if a.val else []):
            z = fetch(url, os.path.join(vqa_dir, os.path.basename(url)))
            unzip(z, vqa_dir)
            if not a.keep_zips:
                os.remove(z)
        js = sorted(f for f in os.listdir(vqa_dir) if f.endswith(".json"))
        for f in js:
            print("  %-52s %s" % (f, human(os.path.getsize(os.path.join(vqa_dir, f)))))
        if not js:
            sys.exit("no VQA json files produced — download failed")

    if not a.vqa_only:
        print("\n== COCO train2014 images (~13 GB) ==")
        if os.path.isdir(os.path.join(coco_dir, "train2014")):
            n = len(os.listdir(os.path.join(coco_dir, "train2014")))
            print("  already have %d images" % n)
        else:
            z = fetch(COCO, os.path.join(coco_dir, "train2014.zip"))
            unzip(z, coco_dir, marker="train2014")
            if not a.keep_zips:
                os.remove(z)
            print("  %d images" % len(os.listdir(os.path.join(coco_dir, "train2014"))))

    print("\n== done ==")
    for d in (vqa_dir, coco_dir):
        if os.path.isdir(d):
            sz = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(d) for f in fs)
            print("  %-28s %s" % (d, human(sz)))
    print("""
next:
  python3 scripts/extract_features.py --images %s/train2014 \\
      --out cache/siglip_train2014 --model google/siglip-base-patch16-224
  python3 scripts/train.py --data-root %s --features cache/siglip_train2014
""" % (coco_dir, a.root))


if __name__ == "__main__":
    main()
