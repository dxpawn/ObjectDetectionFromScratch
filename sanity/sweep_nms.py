"""Post-training tuning: sweep NMS IoU (and optionally confidence) on the val set
to maximize mAP@0.5, then save the winning predictions. Run after training.

Usage:
  python sanity/sweep_nms.py --checkpoint models/best.pth \
      --val_data public/annotations/val.json --val_image_dir public/val/images
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
from utils.dataset import InferenceDataset, inference_collate, load_classes
from utils.evaluate import evaluate_map
from utils.postprocess import postprocess_batch, results_to_submission
from utils.transforms import ValTransform


@torch.no_grad()
def run_inference(model, loader, device, conf, nms):
    results = []
    for imgs, metas in loader:
        imgs = imgs.to(device)
        cls, reg, ctr = model(imgs)
        locs = model.locations_for_features(reg, STRIDES)
        results += postprocess_batch(cls, reg, ctr, locs, metas,
                                     conf_thresh=conf, nms_thresh=nms, max_det=100)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/best.pth")
    p.add_argument("--val_data", default="public/annotations/val.json")
    p.add_argument("--val_image_dir", default="public/val/images")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--nms_list", default="0.45,0.50,0.55,0.60,0.65")
    p.add_argument("--save", default="val_predictions.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    classes = ckpt.get("classes") or load_classes("public/classes.json")
    backbone = ckpt.get("backbone", "resnet34")
    img_size = ckpt.get("img_size", 512)
    print(f"checkpoint={args.checkpoint} backbone={backbone} img_size={img_size} "
          f"ckpt_mAP={ckpt.get('mAP')}")

    model = FCOSDetector(num_classes=len(classes), backbone=backbone, pretrained=False).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_gt = json.load(open(args.val_data, encoding="utf-8"))
    ds = InferenceDataset(args.val_image_dir, ValTransform(img_size))
    loader = DataLoader(ds, batch_size=12, shuffle=False, num_workers=4,
                        collate_fn=inference_collate, pin_memory=True)
    print(f"val images={len(ds)}  sweeping nms in {args.nms_list} at conf={args.conf}")

    best = (-1.0, None, None)
    for nms in [float(x) for x in args.nms_list.split(",")]:
        t0 = time.time()
        results = run_inference(model, loader, args.device, args.conf, nms)
        sub = results_to_submission(results, classes)
        res = evaluate_map(sub, val_gt)
        per = "  ".join(f"{c}:{v['ap']:.3f}" for c, v in res["per_class"].items())
        print(f"  nms={nms:.2f}  mAP@0.5={res['mAP']:.4f}  [{per}]  ({time.time()-t0:.0f}s)")
        if res["mAP"] > best[0]:
            best = (res["mAP"], nms, sub)

    print(f"\nBEST: mAP@0.5={best[0]:.4f} at nms={best[1]:.2f}")
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(best[2], f, ensure_ascii=False)
    print(f"saved winning predictions -> {args.save}")


if __name__ == "__main__":
    main()
