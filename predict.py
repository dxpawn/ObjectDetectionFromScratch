"""Run inference and write predictions in the required format.

Required usage (per assignment)::

    python predict.py --image_dir /path/to/images --output predictions.json

Loads ``<checkpoint>`` (default ``models/best.pth``), runs detection over every
image in ``--image_dir``, applies confidence thresholding + per-class NMS, maps
boxes back to original pixels, and writes a JSON array. Every image appears
exactly once; images with no detections get ``"boxes": []``.

Test-time augmentation: horizontal flip and multi-scale, both on by default.
The default runs multi-scale 512/640/768 + flip (val mAP@0.5 ~0.858). For a faster
single-scale run pass ``--tta_scales 640`` (~0.851); ``--no_tta`` also drops the flip.
Images are processed in chunks so multi-scale stays memory-safe on large test sets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import STRIDES, FCOSDetector
from utils.dataset import IMAGE_EXTS, InferenceDataset, inference_collate, load_classes
from utils.postprocess import (
    decode_candidates_image,
    finalize_detections,
    results_to_submission,
)
from utils.transforms import ValTransform, invert_letterbox

CHUNK = 2000  # images held in memory at once (bounds multi-scale candidate memory)

# The ~145 MB checkpoint is hosted online (GitHub LFS) and fetched on first run if
# absent, so the submission / Docker image need not bundle it (the grader builds the
# image and runs this script; weights download themselves).
DEFAULT_CHECKPOINT_URL = (
    "https://github.com/dxpawn/ObjectDetectionFromScratch/raw/main/models/best.pth"
)


def ensure_checkpoint(path: str, url: str) -> None:
    """Download the checkpoint to ``path`` if real weights are not already present.

    A git-lfs *pointer* (left when a repo is cloned without ``git lfs pull``) is a
    ~130-byte text stub, not weights -- so we treat any sub-1 MB file as missing and
    fetch the real checkpoint.
    """
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return
    if not url:
        raise FileNotFoundError(f"checkpoint not found at {path} and no --checkpoint_url set")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"checkpoint not found at {path}; downloading from {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "odfs-predict"})
    tmp = path + ".part"
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    os.replace(tmp, path)
    print(f"downloaded {os.path.getsize(path) / 1e6:.1f} MB -> {path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FCOS-style detector inference.")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="models/best.pth")
    p.add_argument("--checkpoint_url", default=DEFAULT_CHECKPOINT_URL,
                   help="auto-downloaded to --checkpoint if that file is missing")
    p.add_argument("--img_size", type=int, default=640, help="single-scale size (if --tta_scales unset)")
    p.add_argument("--tta_scales", default="512,640,768",
                   help='comma list of inference scales; pass one value (e.g. "640") for single-scale')
    p.add_argument("--conf_thresh", type=float, default=0.05)
    p.add_argument("--iou_thresh", type=float, default=0.5)
    p.add_argument("--max_det", type=int, default=100)
    p.add_argument("--no_tta", dest="tta", action="store_false", help="disable flip TTA")
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def run_pass(model, loader, device, conf, flip, acc):
    """One scale/flip pass; appends pre-NMS candidates (original coords) into acc."""
    for imgs, metas in loader:
        imgs = imgs.to(device)
        src = torch.flip(imgs, dims=[3]) if flip else imgs
        cls, reg, ctr = model(src)
        locs = model.locations_for_features(reg, STRIDES)
        for b, meta in enumerate(metas):
            cb, cs, cl = decode_candidates_image(
                [o[b] for o in cls], [o[b] for o in reg], [o[b] for o in ctr], locs, conf
            )
            cb = invert_letterbox(cb, meta["scale"], meta["pad"], meta["orig_w"], meta["orig_h"])
            if flip and cb.numel():  # un-flip x back into the original frame
                w = meta["orig_w"]
                x1 = w - cb[:, 2].clone()
                x2 = w - cb[:, 0].clone()
                cb[:, 0], cb[:, 2] = x1, x2
            bl, sl, ll = acc[meta["image_id"]]
            bl.append(cb); sl.append(cs); ll.append(cl)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device

    ensure_checkpoint(args.checkpoint, args.checkpoint_url)
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt.get("classes") or load_classes("public/classes.json")
    backbone = ckpt.get("backbone", "resnet34")
    scales = [int(x) for x in args.tta_scales.split(",")] if args.tta_scales else [args.img_size]
    flips = [False, True] if args.tta else [False]
    print(f"checkpoint={args.checkpoint} backbone={backbone} scales={scales} "
          f"flip={args.tta} classes={classes} device={device}")

    model = FCOSDetector(num_classes=len(classes), backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_files = sorted(f for f in os.listdir(args.image_dir)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    print(f"found {len(all_files)} images in {args.image_dir}")

    results = []
    for start in tqdm(range(0, len(all_files), CHUNK), desc="predict"):
        chunk = all_files[start:start + CHUNK]
        acc = {f: ([], [], []) for f in chunk}
        for s in scales:
            ds = InferenceDataset(args.image_dir, ValTransform(s), files=chunk)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, collate_fn=inference_collate, pin_memory=True)
            for flip in flips:
                run_pass(model, loader, device, args.conf_thresh, flip, acc)
        for f in chunk:
            bl, sl, ll = acc[f]
            cb, cs, cl = torch.cat(bl), torch.cat(sl), torch.cat(ll)
            boxes, scores, labels = finalize_detections(cb, cs, cl, args.iou_thresh, args.max_det)
            results.append({"image_id": f, "boxes": boxes, "scores": scores, "labels": labels})

    submission = results_to_submission(results, classes)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(submission, fh, ensure_ascii=False)
    total = sum(len(e["boxes"]) for e in submission)
    print(f"wrote {len(submission)} images, {total} boxes -> {args.output}")


if __name__ == "__main__":
    main()
