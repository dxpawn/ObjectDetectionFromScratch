"""Multi-scale + flip TTA evaluation on val (no retraining). Infer at several
letterbox sizes, optionally each flipped, merge all candidates per image, then a
single per-class NMS. Reports val mAP@0.5 to decide whether to adopt into predict.py.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import STRIDES, FCOSDetector
from utils.dataset import InferenceDataset, inference_collate, load_classes
from utils.evaluate import evaluate_map
from utils.postprocess import decode_candidates_image, finalize_detections, results_to_submission
from utils.transforms import ValTransform, invert_letterbox


@torch.no_grad()
def candidates_pass(model, loader, device, conf, flip):
    """One full pass at a fixed scale; returns {image_id: (boxes, scores, labels)}
    pre-NMS candidates in original pixel coords."""
    out = {}
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
            if flip and cb.numel():
                w = meta["orig_w"]
                x1 = w - cb[:, 2].clone()
                x2 = w - cb[:, 0].clone()
                cb[:, 0], cb[:, 2] = x1, x2
            out[meta["image_id"]] = (cb, cs, cl)
    return out


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/best.pth")
    p.add_argument("--val_data", default="public/annotations/val.json")
    p.add_argument("--val_image_dir", default="public/val/images")
    p.add_argument("--scales", default="512,640,768")
    p.add_argument("--no_flip", dest="flip", action="store_false")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--max_det", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    classes = ckpt.get("classes") or load_classes("public/classes.json")
    model = FCOSDetector(num_classes=len(classes), backbone=ckpt.get("backbone", "resnet34"),
                         pretrained=False).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    val_gt = json.load(open(args.val_data, encoding="utf-8"))

    scales = [int(x) for x in args.scales.split(",")]
    acc = defaultdict(lambda: ([], [], []))
    t0 = time.time()
    for s in scales:
        ds = InferenceDataset(args.val_image_dir, ValTransform(s))
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4,
                            collate_fn=inference_collate, pin_memory=True)
        for flip in ([False, True] if args.flip else [False]):
            for iid, (cb, cs, cl) in candidates_pass(model, loader, args.device, args.conf, flip).items():
                acc[iid][0].append(cb)
                acc[iid][1].append(cs)
                acc[iid][2].append(cl)

    results = []
    for iid, (bl, sl, ll) in acc.items():
        cb, cs, cl = torch.cat(bl), torch.cat(sl), torch.cat(ll)
        boxes, scores, labels = finalize_detections(cb, cs, cl, args.nms, args.max_det)
        results.append({"image_id": iid, "boxes": boxes, "scores": scores, "labels": labels})

    res = evaluate_map(results_to_submission(results, classes), val_gt)
    per = "  ".join(f"{c}:{v['ap']:.3f}" for c, v in res["per_class"].items())
    print(f"[scales={scales} flip={args.flip} nms={args.nms}]  "
          f"mAP@0.5={res['mAP']:.4f}  [{per}]  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
