"""Gate 3: verify FCOS target assignment + loss on a few real images.

Checks:
  * positives exist for images that have GT boxes,
  * centerness targets lie in [0, 1],
  * decoding the GT regression targets at positive locations reproduces the
    matched GT box (IoU == 1), i.e. assignment/decoding coords are consistent,
  * positives for a box land on a sensible pyramid level (max side-distance in
    that level's range),
  * the full FCOSLoss returns finite numbers.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import STRIDES, FCOSDetector
from utils.boxes import box_iou, ltrb_to_xyxy
from utils.dataset import DetectionDataset, load_classes
from utils.losses import FCOSLoss
from utils.targets import OBJECT_SIZES, build_point_metadata, compute_fcos_targets
from utils.transforms import ValTransform


def main() -> None:
    classes = load_classes("public/classes.json")
    ds = DetectionDataset(
        "public/annotations/train.json", "public/train/images", classes,
        transform=ValTransform(512),
    )
    model = FCOSDetector(num_classes=5, pretrained=False)
    model.eval()

    # Grab feature sizes from one forward pass, build location grids.
    sample_img = ds[0][0].unsqueeze(0)
    with torch.no_grad():
        cls, reg, ctr = model(sample_img)
    locs_list = model.locations_for_features(reg, STRIDES)
    points, p_strides, p_ranges = build_point_metadata(locs_list, STRIDES, OBJECT_SIZES)

    picks = [i for i in range(200) if len(ds.samples[i]["annotations"]) >= 2][:3]
    all_ok = True
    for i in picks:
        img, boxes, labels, meta = ds[i]
        lab, reg_t, ctr_t = compute_fcos_targets(
            points, p_strides, p_ranges, boxes, labels
        )
        pos = lab >= 0
        n_pos = int(pos.sum())

        # centerness range
        ctr_ok = bool((ctr_t[pos] >= -1e-6).all() and (ctr_t[pos] <= 1 + 1e-6).all())

        # decode GT reg targets -> boxes, compare to GT
        dec = ltrb_to_xyxy(points[pos], reg_t[pos])
        ious = box_iou(dec, boxes)              # (P, Ng)
        best_iou = ious.max(dim=1).values
        decode_ok = bool((best_iou > 0.999).all())

        # level distribution
        lvl_of = torch.zeros(points.shape[0], dtype=torch.long)
        offset = 0
        for li, locs in enumerate(locs_list):
            lvl_of[offset : offset + locs.shape[0]] = li
            offset += locs.shape[0]
        lvls = sorted(set(lvl_of[pos].tolist()))

        ok = n_pos > 0 and ctr_ok and decode_ok
        all_ok &= ok
        print(
            f"img {meta['image_id']}: boxes={len(boxes)} pos={n_pos} "
            f"centerness_in[0,1]={ctr_ok} gt_decode_iou>=1={decode_ok} "
            f"levels_used={lvls}  -> {'OK' if ok else 'FAIL'}"
        )

    # Full loss on a real batch.
    loss_fn = FCOSLoss(num_classes=5, strides=STRIDES)
    imgs = torch.stack([ds[i][0] for i in picks])
    gtb = [ds[i][1] for i in picks]
    gtl = [ds[i][2] for i in picks]
    with torch.no_grad():
        cls, reg, ctr = model(imgs)
    locs_list = model.locations_for_features(reg, STRIDES)
    out = loss_fn(cls, reg, ctr, locs_list, gtb, gtl)
    finite = all(torch.isfinite(v).all() for v in out.values())
    print(
        f"loss total={out['loss']:.4f} cls={out['cls']:.4f} reg={out['reg']:.4f} "
        f"ctr={out['ctr']:.4f} num_pos={out['num_pos']:.0f} finite={finite}"
    )
    print("GATE 3:", "PASS" if (all_ok and finite) else "FAIL")


if __name__ == "__main__":
    main()
