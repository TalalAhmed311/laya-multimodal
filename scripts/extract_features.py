#!/usr/bin/env python3
"""Run SigLIP once over the images and cache the patch features to disk.

This is the single change that makes 2xT4 feasible. The vision tower is frozen,
so it never needs to run again: after this, every training epoch is text-speed
and the tower is not resident in VRAM at all.

    python3 scripts/extract_features.py --images data/coco/train2014 \
        --out cache/siglip_train2014.pt --model google/siglip-so400m-patch14-384

Output is a **memory-mapped** .npy plus a .json sidecar, not a dict of tensors.
That matters: a dict loaded with torch.load lands entirely in RAM, and Colab
gives you ~12 GB. A memmap costs nothing to open and pages in per batch.

    cache/siglip_train2014.npy    [N, P, d_v] float16
    cache/siglip_train2014.json   {image_ids, P, d_v, model}

Storage per image (fp16):

    siglip-base-patch16-224      196 patches x  768  -> 0.30 MB   (25 GB for all COCO)
    siglip-so400m-patch14-224    256 patches x 1152  -> 0.59 MB   (49 GB)
    siglip-so400m-patch14-384    729 patches x 1152  -> 1.68 MB   (139 GB)

Colab free has ~107 GB disk, so the full corpus is out on all three. Use --ids
with a subset (20k images is ~6 GB at base-224). Use 384 only if the model must
read text burned into the image; VQA supplies the question as text and Hateful
Memes ships OCR, so 224 is usually right.
"""
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="directory of images")
    ap.add_argument("--out", required=True, help="output .pt")
    ap.add_argument("--model", default="google/siglip-base-patch16-224",
                    help="base-224 for Colab; so400m-patch14-384 if you have the disk")
    ap.add_argument("--ids", default=None,
                    help="json list of image_ids to restrict to (recommended)")
    ap.add_argument("--prefix", default="COCO_train2014_",
                    help="filename prefix; COCO pads ids to 12 digits")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-images", type=int, default=0, help="cap for a Colab-sized run")
    ap.add_argument("--fp16", action="store_true", default=True)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", dev)

    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).vision_model.to(dev).eval()
    if a.fp16 and dev == "cuda":
        model = model.half()

    if a.ids:
        ids = json.load(open(a.ids))
        paths = [(i, os.path.join(a.images, "%s%012d.jpg" % (a.prefix, i))) for i in ids]
        paths = [(i, p) for i, p in paths if os.path.exists(p)]
    else:
        paths = []
        for f in sorted(os.listdir(a.images)):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                stem = os.path.splitext(f)[0]
                try:
                    paths.append((int(stem.split("_")[-1]), os.path.join(a.images, f)))
                except ValueError:
                    paths.append((stem, os.path.join(a.images, f)))
    if not paths:
        sys.exit("no images found under %s" % a.images)
    if a.max_images:
        paths = paths[:a.max_images]
    print("%d images -> %s" % (len(paths), a.out))

    import numpy as np
    stem = a.out[:-4] if a.out.endswith(".npy") else a.out
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    arr, ids, t0 = None, [], time.time()
    with torch.no_grad():
        for s in range(0, len(paths), a.batch):
            chunk = paths[s:s + a.batch]
            imgs = [Image.open(p).convert("RGB") for _, p in chunk]
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
            if a.fp16 and dev == "cuda":
                px = px.half()
            out = model(pixel_values=px).last_hidden_state    # [B, P, d_v]
            if arr is None:                                   # allocate the memmap once
                P, d_v = out.shape[1], out.shape[2]
                arr = np.lib.format.open_memmap(
                    stem + ".npy", mode="w+", dtype=np.float16,
                    shape=(len(paths), P, d_v))
                print("  allocating %s  [%d, %d, %d] fp16 = %.1f GB"
                      % (stem + ".npy", len(paths), P, d_v,
                         len(paths) * P * d_v * 2 / 1e9))
            arr[len(ids):len(ids) + len(chunk)] = out.to(torch.float16).cpu().numpy()
            ids.extend(int(i) if str(i).isdigit() else i for i, _ in chunk)
            done = s + len(chunk)
            if done % (a.batch * 20) == 0 or done == len(paths):
                el = time.time() - t0
                sys.stdout.write("\r  %d/%d  %.1f img/s  ETA %ds   "
                                 % (done, len(paths), done / max(el, 1e-9),
                                    int(el * (len(paths) - done) / max(1, done))))
                sys.stdout.flush()
    print()

    arr.flush()
    json.dump({"image_ids": ids, "P": int(arr.shape[1]), "d_v": int(arr.shape[2]),
               "model": a.model}, open(stem + ".json", "w"))
    print("cached %d images -> %s(.npy/.json) | %.1f GB"
          % (len(ids), stem, os.path.getsize(stem + ".npy") / 1e9))
    print("P = %d patches, d_v = %d  -> pass d_v=%d to model.build()"
          % (arr.shape[1], arr.shape[2], arr.shape[2]))


if __name__ == "__main__":
    main()
