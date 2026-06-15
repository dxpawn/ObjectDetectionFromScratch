"""Decode FCOS outputs into detections: score, threshold, per-class NMS, and
map boxes back to original image pixels.

Confidence = sigmoid(class) * sigmoid(centerness), per FCOS.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .boxes import batched_nms, ltrb_to_xyxy
from .transforms import invert_letterbox


@torch.no_grad()
def decode_candidates_image(
    cls_list, reg_list, ctr_list, locations_list,
    conf_thresh: float = 0.05, topk_per_level: int = 300, pre_nms_topk: int = 1000,
):
    """Pre-NMS candidate detections for one image (letterbox coords, on CPU).

    Each *_list element is a per-level tensor: cls (C,H,W), reg (4,H,W),
    ctr (1,H,W); locations (H*W,2). Returns (boxes (M,4), scores (M,), labels (M,)).
    Candidates are capped per level (top-k) then globally (pre-NMS top-k). Exposed
    so flip-TTA can merge candidates from two passes before a single NMS."""
    boxes_all, scores_all, labels_all = [], [], []
    c = cls_list[0].shape[0]

    for cls, reg, ctr, locs in zip(cls_list, reg_list, ctr_list, locations_list):
        cls = cls.permute(1, 2, 0).reshape(-1, c).sigmoid()
        ctr = ctr.permute(1, 2, 0).reshape(-1).sigmoid()
        reg = reg.permute(1, 2, 0).reshape(-1, 4)
        score_flat = (cls * ctr[:, None]).reshape(-1)

        keep = score_flat > conf_thresh
        if keep.sum() == 0:
            continue
        idxs = keep.nonzero(as_tuple=False).squeeze(1)
        sel_scores = score_flat[idxs]
        if sel_scores.numel() > topk_per_level:
            sel_scores, top = sel_scores.topk(topk_per_level)
            idxs = idxs[top]
        boxes_all.append(ltrb_to_xyxy(locs[idxs // c], reg[idxs // c]))
        scores_all.append(sel_scores)
        labels_all.append(idxs % c)

    if not boxes_all:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

    boxes = torch.cat(boxes_all)
    scores = torch.cat(scores_all)
    labels = torch.cat(labels_all)
    if scores.numel() > pre_nms_topk:
        scores, top = scores.topk(pre_nms_topk)
        boxes, labels = boxes[top], labels[top]
    return boxes.cpu(), scores.cpu(), labels.cpu()


def finalize_detections(boxes, scores, labels, nms_thresh: float, max_det: int):
    """Per-class NMS (on CPU) then keep top ``max_det`` by score."""
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = batched_nms(boxes, scores, labels, nms_thresh)[:max_det]
    return boxes[keep], scores[keep], labels[keep]


def postprocess_image(
    cls_list, reg_list, ctr_list, locations_list,
    conf_thresh: float = 0.05, nms_thresh: float = 0.6, max_det: int = 100,
    topk_per_level: int = 300, pre_nms_topk: int = 1000,
):
    """Decode + per-class NMS for one image. Returns boxes (letterbox coords),
    scores, labels."""
    boxes, scores, labels = decode_candidates_image(
        cls_list, reg_list, ctr_list, locations_list, conf_thresh, topk_per_level, pre_nms_topk
    )
    return finalize_detections(boxes, scores, labels, nms_thresh, max_det)


@torch.no_grad()
def postprocess_batch(
    cls_outs, reg_outs, ctr_outs, locations_list, metas,
    conf_thresh: float = 0.05, nms_thresh: float = 0.6, max_det: int = 100,
):
    """Decode a batch and map each image's boxes back to original pixels.

    Returns a list of dicts: {image_id, boxes (M,4) original coords, scores, labels}.
    """
    batch = cls_outs[0].shape[0]
    results = []
    for b in range(batch):
        cls_list = [o[b] for o in cls_outs]
        reg_list = [o[b] for o in reg_outs]
        ctr_list = [o[b] for o in ctr_outs]
        boxes, scores, labels = postprocess_image(
            cls_list, reg_list, ctr_list, locations_list,
            conf_thresh=conf_thresh, nms_thresh=nms_thresh, max_det=max_det,
        )
        meta = metas[b]
        boxes = invert_letterbox(
            boxes.cpu(), meta["scale"], meta["pad"], meta["orig_w"], meta["orig_h"]
        )
        results.append(
            {
                "image_id": meta["image_id"],
                "boxes": boxes,
                "scores": scores.cpu(),
                "labels": labels.cpu(),
            }
        )
    return results


def results_to_submission(results, classes, score_thresh: float = 0.0):
    """Convert decoded results into the required predictions.json structure.

    Filters out degenerate boxes; keeps every image (possibly with empty boxes).
    """
    out = []
    for res in results:
        boxes_json = []
        for box, score, label in zip(
            res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()
        ):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_json.append(
                {
                    "class": classes[label],
                    "confidence": round(float(score), 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )
        out.append({"image_id": res["image_id"], "boxes": boxes_json})
    return out
