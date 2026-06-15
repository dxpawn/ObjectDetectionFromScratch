"""Gate 4: overfit ~50 images. If the whole pipeline is wired correctly the loss
should collapse toward 0 and self-eval mAP@0.5 should approach 1.0. A failure
here means a bug in targets/loss/decode/NMS, not a tuning problem."""

import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import STRIDES, FCOSDetector
from utils.dataset import DetectionDataset, collate_fn, load_classes
from utils.evaluate import evaluate_map, subset_ground_truth
from utils.losses import FCOSLoss
from utils.postprocess import postprocess_batch, results_to_submission
from utils.transforms import ValTransform


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = load_classes("public/classes.json")
    gt = json.load(open("public/annotations/train.json", encoding="utf-8"))

    # 50 images that have at least one annotation.
    with_ann = {a["image_id"] for a in gt["annotations"]}
    subset_ids = [im["id"] for im in gt["images"] if im["id"] in with_ann][:50]

    ds = DetectionDataset(
        "public/annotations/train.json", "public/train/images", classes,
        transform=ValTransform(512), subset_ids=subset_ids,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)

    model = FCOSDetector(num_classes=5, pretrained=True).to(device)
    model.train()
    loss_fn = FCOSLoss(num_classes=5, strides=STRIDES)
    # Mirror train.py's recipe: SGD + warmup + cosine decay (lr -> 0).
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)

    base_lr, warmup, steps, max_steps = 0.01, 100, 0, 1200

    def lr_at(it):
        if it < warmup:
            return base_lr * (it + 1) / warmup
        prog = (it - warmup) / max(1, max_steps - warmup)
        return 0.5 * base_lr * (1 + math.cos(math.pi * min(prog, 1.0)))

    t0 = time.time()
    while steps < max_steps:
        for imgs, boxes, labels, metas in loader:
            lr = lr_at(steps)
            for g in opt.param_groups:
                g["lr"] = lr
            imgs = imgs.to(device)
            cls, reg, ctr = model(imgs)
            locs = model.locations_for_features(reg, STRIDES)
            out = loss_fn(cls, reg, ctr, locs, boxes, labels)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            steps += 1
            if steps % 50 == 0:
                print(f"step {steps:4d}  lr={lr:.4f} loss={out['loss']:.4f} "
                      f"cls={out['cls']:.4f} reg={out['reg']:.4f} ctr={out['ctr']:.4f}")
            if steps >= max_steps:
                break
    print(f"trained {steps} steps in {time.time()-t0:.1f}s")

    # Self-eval on the same 50 images.
    model.eval()
    eval_loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
    all_results = []
    with torch.no_grad():
        for imgs, boxes, labels, metas in eval_loader:
            imgs = imgs.to(device)
            cls, reg, ctr = model(imgs)
            locs = model.locations_for_features(reg, STRIDES)
            all_results += postprocess_batch(cls, reg, ctr, locs, metas,
                                             conf_thresh=0.05, nms_thresh=0.6, max_det=100)
    preds = results_to_submission(all_results, classes)
    sub_gt = subset_ground_truth(gt, subset_ids)
    res = evaluate_map(preds, sub_gt)
    print(f"overfit-50 mAP@0.5 = {res['mAP']:.4f}")
    for c, v in res["per_class"].items():
        if v["num_gt"]:
            print(f"  {c:7s} ap={v['ap']:.3f} gt={v['num_gt']} pred={v['num_pred']} "
                  f"recall={v['recall']:.3f} prec={v['precision']:.3f}")
    print("GATE 4:", "PASS" if res["mAP"] > 0.80 else "NEEDS REVIEW")


if __name__ == "__main__":
    main()
