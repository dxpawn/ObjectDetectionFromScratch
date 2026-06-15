"""Train the from-scratch FCOS-style detector.

Required usage (per assignment)::

    python train.py \
      --train_data ./public/annotations/train.json \
      --val_data   ./public/annotations/val.json \
      --image_dir  ./public/train/images \
      --val_image_dir ./public/val/images \
      --checkpoint_dir ./models/

Saves the best model (by validation mAP@0.5) to ``<checkpoint_dir>/best.pth`` and
the latest to ``<checkpoint_dir>/last.pth``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models import STRIDES, FCOSDetector
from utils.dataset import DetectionDataset, collate_fn, load_classes
from utils.evaluate import evaluate_map
from utils.losses import FCOSLoss
from utils.postprocess import postprocess_batch, results_to_submission
from utils.transforms import TrainTransform, ValTransform

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FCOS-style detector.")
    p.add_argument("--train_data", required=True)
    p.add_argument("--val_data", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--val_image_dir", required=True)
    p.add_argument("--checkpoint_dir", default="./models/")
    p.add_argument("--classes", default="public/classes.json")
    p.add_argument("--backbone", default="resnet34",
                   choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=500)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--multiscale", action="store_true")
    p.add_argument("--mosaic_prob", type=float, default=0.0,
                   help="probability of building a 4-image mosaic per sample")
    p.add_argument("--close_mosaic", type=int, default=5,
                   help="disable mosaic for the final N epochs")
    p.add_argument("--oversample_chair", action="store_true",
                   help="weight chair-containing images higher in sampling")
    p.add_argument("--chair_boost", type=float, default=3.0)
    p.add_argument("--ema", action="store_true",
                   help="track an EMA of the weights; ship whichever of raw/EMA scores higher")
    p.add_argument("--ema_decay", type=float, default=0.9998)
    p.add_argument("--ema_eval_from", type=int, default=0,
                   help="only evaluate EMA from this epoch on (early EMA is noise; saves eval time)")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--eval_conf", type=float, default=0.05)
    p.add_argument("--eval_nms", type=float, default=0.6)
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class ModelEMA:
    """Exponential Moving Average of model weights (hand-written).

    Keeps a shadow copy updated after each optimizer step as
    ``ema = d*ema + (1-d)*model``. The decay ramps in early
    (``d = decay*(1 - exp(-updates/tau))``) so the average isn't dominated by the
    noisy initial weights. EMA weights are typically a touch more accurate and far
    more stable than the raw weights at the end of training. Non-float buffers
    (e.g. BN ``num_batches_tracked``) are copied, not averaged.
    """

    def __init__(self, model, decay: float = 0.9998, tau: float = 2000.0) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def lr_at(it: int, base_lr: float, warmup: int, total: int) -> float:
    if it < warmup:
        return base_lr * (it + 1) / warmup
    progress = (it - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate_model(model, loader, classes, gt_data, device, conf, nms) -> dict:
    model.eval()
    results = []
    for imgs, _, _, metas in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device)
        cls, reg, ctr = model(imgs)
        locs = model.locations_for_features(reg, STRIDES)
        results += postprocess_batch(cls, reg, ctr, locs, metas, conf_thresh=conf,
                                     nms_thresh=nms, max_det=100)
    preds = results_to_submission(results, classes)
    return evaluate_map(preds, gt_data)


def main() -> None:
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = args.device
    classes = load_classes(args.classes)
    print(f"device={device} classes={classes} backbone={args.backbone} amp={not args.no_amp}")

    train_tf = TrainTransform(args.img_size)
    val_tf = ValTransform(args.img_size)
    train_ds = DetectionDataset(args.train_data, args.image_dir, classes, train_tf,
                                mosaic_prob=args.mosaic_prob, img_size=args.img_size)
    val_ds = DetectionDataset(args.val_data, args.val_image_dir, classes, val_tf)
    val_gt = json.load(open(args.val_data, encoding="utf-8"))
    # Batch-level multi-scale sizes: a band ending at the target inference size.
    ms_sizes = [s for s in (args.img_size - 128, args.img_size - 64, args.img_size) if s >= 320]
    print(f"train images={len(train_ds)} val images={len(val_ds)} "
          f"mosaic={args.mosaic_prob} oversample_chair={args.oversample_chair}")

    sampler = None
    if args.oversample_chair:
        weights = [args.chair_boost if any(a["class"] == "chair" for a in s["annotations"])
                   else 1.0 for s in train_ds.samples]
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                        num_samples=len(train_ds), replacement=True)

    def make_train_loader():
        return DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
            num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, drop_last=True,
            persistent_workers=args.workers > 0,
        )

    train_loader = make_train_loader()
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=args.workers > 0,
    )

    model = FCOSDetector(num_classes=len(classes), backbone=args.backbone, pretrained=True).to(device)
    loss_fn = FCOSLoss(num_classes=len(classes), strides=STRIDES)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scaler = GradScaler(device, enabled=not args.no_amp)

    start_epoch, best_map = 0, -1.0  # -1 so epoch 1 always writes best.pth
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_map = ckpt.get("mAP", 0.0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"resumed from {args.resume} (epoch {start_epoch}, best mAP {best_map:.4f})")

    # EMA is created after any resume so it starts from the loaded weights.
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    if ema is not None:
        print(f"EMA enabled (decay={args.ema_decay}); evaluating raw + EMA each epoch")

    total_iters = args.epochs * len(train_loader)
    it = start_epoch * len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        # Close mosaic for the final epochs so the model fine-tunes on real images.
        if (args.mosaic_prob > 0 and train_ds.mosaic_prob > 0
                and epoch >= args.epochs - args.close_mosaic):
            train_ds.mosaic_prob = 0.0
            train_loader = make_train_loader()  # recreate so workers see mosaic off
            print(f"epoch {epoch+1}: close-mosaic - mosaic disabled")
        model.train()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for imgs, boxes, labels, _ in pbar:
            lr = lr_at(it, args.lr, args.warmup_iters, total_iters)
            for g in opt.param_groups:
                g["lr"] = lr
            imgs = imgs.to(device, non_blocking=True)
            if args.multiscale:
                msize = random.choice(ms_sizes)
                if msize != imgs.shape[-1]:
                    s = msize / imgs.shape[-1]
                    imgs = F.interpolate(imgs, size=(msize, msize), mode="bilinear",
                                         align_corners=False)
                    boxes = [b * s for b in boxes]
            with autocast(device, enabled=not args.no_amp):
                cls, reg, ctr = model(imgs)
                locs = model.locations_for_features(reg, STRIDES)
                out = loss_fn(cls, reg, ctr, locs, boxes, labels)
            opt.zero_grad()
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            it += 1
            if it % 20 == 0:
                pbar.set_postfix(loss=f"{out['loss']:.3f}", cls=f"{out['cls']:.3f}",
                                 reg=f"{out['reg']:.3f}", ctr=f"{out['ctr']:.3f}", lr=f"{lr:.4f}")

        res = evaluate_model(model, val_loader, classes, val_gt, device,
                             args.eval_conf, args.eval_nms)
        mAP = res["mAP"]
        per = "  ".join(f"{c}:{v['ap']:.3f}" for c, v in res["per_class"].items())
        print(f"epoch {epoch+1}: mAP@0.5={mAP:.4f}  [{per}]  ({time.time()-t0:.0f}s)")

        # Candidate to ship: the raw weights, unless EMA scores at least as high.
        ship_state, ship_map, ship_src = model.state_dict(), mAP, "raw"
        if ema is not None and (epoch + 1) >= args.ema_eval_from:
            res_e = evaluate_model(ema.ema, val_loader, classes, val_gt, device,
                                   args.eval_conf, args.eval_nms)
            per_e = "  ".join(f"{c}:{v['ap']:.3f}" for c, v in res_e["per_class"].items())
            print(f"           EMA  mAP@0.5={res_e['mAP']:.4f}  [{per_e}]")
            if res_e["mAP"] >= mAP:
                ship_state, ship_map, ship_src = ema.ema.state_dict(), res_e["mAP"], "ema"

        # last.pth keeps the raw weights so --resume continues training cleanly.
        torch.save({"model": model.state_dict(), "classes": classes, "img_size": args.img_size,
                    "backbone": args.backbone, "mAP": mAP, "epoch": epoch + 1},
                   os.path.join(args.checkpoint_dir, "last.pth"))
        if ship_map > best_map:
            best_map = ship_map
            torch.save({"model": ship_state, "classes": classes, "img_size": args.img_size,
                        "backbone": args.backbone, "mAP": ship_map, "epoch": epoch + 1,
                        "ema": ship_src == "ema"},
                       os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"  -> new best mAP {best_map:.4f} (from {ship_src}), saved best.pth")

    print(f"done. best mAP@0.5 = {best_map:.4f}")


if __name__ == "__main__":
    main()
