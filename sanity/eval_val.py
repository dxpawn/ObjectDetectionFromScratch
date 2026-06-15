"""Quick-win evaluation harness (no retraining). Measures val mAP@0.5 for
combinations of: inference resolution, horizontal-flip TTA, and Soft-NMS.

Isolated from the deliverable predict.py — used to decide which tricks to keep.
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import STRIDES, FCOSDetector
from utils.boxes import batched_nms, ltrb_to_xyxy, soft_nms_per_class
from utils.dataset import InferenceDataset, inference_collate, load_classes
from utils.evaluate import evaluate_map
from utils.postprocess import results_to_submission
from utils.transforms import ValTransform, invert_letterbox


def decode_candidates(cls_list, reg_list, ctr_list, locs_list, conf, topk_per_level=300):
    """Pre-NMS candidates (letterbox coords) for one image -> (boxes, scores, labels)."""
    ball, sall, lall = [], [], []
    c = cls_list[0].shape[0]
    for cls, reg, ctr, locs in zip(cls_list, reg_list, ctr_list, locs_list):
        cls = cls.permute(1, 2, 0).reshape(-1, c).sigmoid()
        ctr = ctr.permute(1, 2, 0).reshape(-1).sigmoid()
        reg = reg.permute(1, 2, 0).reshape(-1, 4)
        score = (cls * ctr[:, None]).reshape(-1)
        keep = score > conf
        if keep.sum() == 0:
            continue
        idx = keep.nonzero(as_tuple=False).squeeze(1)
        s = score[idx]
        if s.numel() > topk_per_level:
            s, t = s.topk(topk_per_level)
            idx = idx[t]
        ball.append(ltrb_to_xyxy(locs[idx // c], reg[idx // c]))
        sall.append(s)
        lall.append(idx % c)
    if not ball:
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return torch.cat(ball).cpu(), torch.cat(sall).cpu(), torch.cat(lall).cpu()


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/best.pth")
    p.add_argument("--val_data", default="public/annotations/val.json")
    p.add_argument("--val_image_dir", default="public/val/images")
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--soft_nms", action="store_true")
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--conf", type=float, default=0.05)
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
    ds = InferenceDataset(args.val_image_dir, ValTransform(args.img_size))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4,
                        collate_fn=inference_collate, pin_memory=True)

    t0 = time.time()
    results = []
    for imgs, metas in loader:
        imgs = imgs.to(args.device)
        cls, reg, ctr = model(imgs)
        locs = model.locations_for_features(reg, STRIDES)
        if args.flip:
            clsf, regf, ctrf = model(torch.flip(imgs, dims=[3]))
            locsf = model.locations_for_features(regf, STRIDES)

        for b in range(len(metas)):
            meta = metas[b]
            cb, cs, cl = decode_candidates([o[b] for o in cls], [o[b] for o in reg],
                                           [o[b] for o in ctr], locs, args.conf)
            cb = invert_letterbox(cb, meta["scale"], meta["pad"], meta["orig_w"], meta["orig_h"])
            if args.flip:
                fb, fs, fl = decode_candidates([o[b] for o in clsf], [o[b] for o in regf],
                                               [o[b] for o in ctrf], locsf, args.conf)
                fb = invert_letterbox(fb, meta["scale"], meta["pad"], meta["orig_w"], meta["orig_h"])
                if fb.numel():
                    w = meta["orig_w"]
                    x1 = w - fb[:, 2].clone()
                    x2 = w - fb[:, 0].clone()
                    fb[:, 0], fb[:, 2] = x1, x2
                cb = torch.cat([cb, fb]); cs = torch.cat([cs, fs]); cl = torch.cat([cl, fl])

            if cb.numel() == 0:
                boxes, scores, labels = cb, cs, cl
            elif args.soft_nms:
                ki, ds_ = soft_nms_per_class(cb, cs, cl, sigma=args.sigma)
                order = ds_.argsort(descending=True)[:args.max_det]
                sel = ki[order]
                boxes, scores, labels = cb[sel], ds_[order], cl[sel]
            else:
                keep = batched_nms(cb, cs, cl, args.nms)[:args.max_det]
                boxes, scores, labels = cb[keep], cs[keep], cl[keep]
            results.append({"image_id": meta["image_id"], "boxes": boxes,
                            "scores": scores, "labels": labels})

    sub = results_to_submission(results, classes)
    res = evaluate_map(sub, val_gt)
    per = "  ".join(f"{c}:{v['ap']:.3f}" for c, v in res["per_class"].items())
    cfg = f"size={args.img_size} flip={args.flip} soft_nms={args.soft_nms} nms={args.nms}"
    print(f"[{cfg}]  mAP@0.5={res['mAP']:.4f}  [{per}]  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
