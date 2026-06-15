"""Diagnostic: hard-overfit a tiny batch and inspect predictions vs GT to find
why the overfit-50 gate failed."""

import json
import os
import sys

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
    with_ann = {a["image_id"] for a in gt["annotations"]}
    subset_ids = [im["id"] for im in gt["images"] if im["id"] in with_ann][:8]

    ds = DetectionDataset(
        "public/annotations/train.json", "public/train/images", classes,
        transform=ValTransform(512), subset_ids=subset_ids,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    imgs, boxes, labels, metas = batch
    imgs = imgs.to(device)

    model = FCOSDetector(num_classes=5, pretrained=True).to(device)
    model.train()
    loss_fn = FCOSLoss(num_classes=5, strides=STRIDES)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    for step in range(800):
        cls, reg, ctr = model(imgs)
        locs = model.locations_for_features(reg, STRIDES)
        out = loss_fn(cls, reg, ctr, locs, boxes, labels)
        opt.zero_grad()
        out["loss"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        opt.step()
        if step % 100 == 0 or step == 799:
            print(f"step {step:4d} loss={out['loss']:.4f} cls={out['cls']:.4f} "
                  f"reg={out['reg']:.4f} ctr={out['ctr']:.4f} pos={out['num_pos']:.0f} gnorm={gnorm:.2f}")

    model.eval()
    with torch.no_grad():
        cls, reg, ctr = model(imgs)
        locs = model.locations_for_features(reg, STRIDES)
        results = postprocess_batch(cls, reg, ctr, locs, metas,
                                    conf_thresh=0.05, nms_thresh=0.6, max_det=100)

    # Inspect image 0: GT vs top predictions, in ORIGINAL coords.
    img0_id = metas[0]["image_id"]
    gt0 = [(a["class"], a["bbox"]) for a in gt["annotations"] if a["image_id"] == img0_id]
    print(f"\nimage {img0_id}  scale={metas[0]['scale']:.4f} pad={metas[0]['pad']} "
          f"orig=({metas[0]['orig_w']}x{metas[0]['orig_h']})")
    print("GT boxes:")
    for cname, bb in gt0:
        print(f"   {cname:7s} {[round(v,1) for v in bb]}")
    r0 = results[0]
    order = r0["scores"].argsort(descending=True)[:6]
    print("Top predicted boxes:")
    for i in order.tolist():
        print(f"   {classes[int(r0['labels'][i])]:7s} conf={float(r0['scores'][i]):.3f} "
              f"{[round(v,1) for v in r0['boxes'][i].tolist()]}")

    preds = results_to_submission(results, classes)
    sub_gt = subset_ground_truth(gt, subset_ids)
    res = evaluate_map(preds, sub_gt)
    print(f"\noverfit-8 mAP@0.5 = {res['mAP']:.4f}")
    for c, v in res["per_class"].items():
        if v["num_gt"]:
            print(f"  {c:7s} ap={v['ap']:.3f} gt={v['num_gt']} pred={v['num_pred']}")


if __name__ == "__main__":
    main()
