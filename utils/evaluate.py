"""mAP@0.5 computation mirroring ``public/tools/evaluate_predictions.py``.

Used during training to select the best checkpoint. The official grader remains
the final authority; this is a faithful in-process reimplementation (VOC-style
per-class AP, greedy IoU matching by descending confidence).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + \
        max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls, precisions) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_map(predictions, ground_truth, iou_threshold: float = 0.5) -> dict[str, Any]:
    """``predictions``: list of {image_id, boxes:[{class,confidence,bbox}]}.
    ``ground_truth``: parsed annotation dict (classes/images/annotations)."""
    classes = ground_truth["classes"]

    gt_by_class = {c: defaultdict(list) for c in classes}
    for ann in ground_truth["annotations"]:
        gt_by_class[ann["class"]][ann["image_id"]].append(
            {"bbox": [float(v) for v in ann["bbox"]], "matched": False}
        )

    pred_by_class = {c: [] for c in classes}
    for entry in predictions:
        image_id = entry["image_id"]
        for box in entry["boxes"]:
            pred_by_class[box["class"]].append(
                {"image_id": image_id, "confidence": float(box["confidence"]),
                 "bbox": [float(v) for v in box["bbox"]]}
            )

    per_class, aps = {}, []
    for c in classes:
        class_gt = gt_by_class[c]
        num_gt = sum(len(v) for v in class_gt.values())
        preds = sorted(pred_by_class[c], key=lambda x: x["confidence"], reverse=True)

        tp, fp = [], []
        for pred in preds:
            candidates = class_gt.get(pred["image_id"], [])
            best_iou, best_idx = 0.0, -1
            for idx, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = bbox_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                candidates[best_idx]["matched"] = True
                tp.append(1); fp.append(0)
            else:
                tp.append(0); fp.append(1)

        ctp = cfp = 0
        recalls, precisions = [], []
        for t, f in zip(tp, fp):
            ctp += t; cfp += f
            recalls.append(ctp / num_gt if num_gt else 0.0)
            precisions.append(ctp / max(ctp + cfp, 1))
        ap = compute_ap(recalls, precisions) if num_gt else 0.0
        if num_gt:
            aps.append(ap)
        per_class[c] = {
            "ap": round(ap, 4),
            "num_gt": num_gt,
            "num_pred": len(preds),
            "recall": round(ctp / num_gt, 4) if num_gt else 0.0,
            "precision": round(ctp / max(ctp + cfp, 1), 4),
        }

    mean_ap = sum(aps) / len(aps) if aps else 0.0
    return {"mAP": mean_ap, "per_class": per_class}


def subset_ground_truth(ground_truth, image_ids) -> dict[str, Any]:
    """Restrict a parsed annotation dict to a set of image ids."""
    ids = set(image_ids)
    return {
        "classes": ground_truth["classes"],
        "images": [im for im in ground_truth["images"] if im["id"] in ids],
        "annotations": [a for a in ground_truth["annotations"] if a["image_id"] in ids],
    }
