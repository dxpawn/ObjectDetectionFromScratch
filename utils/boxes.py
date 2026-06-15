"""Bounding-box operations, all hand-written in PyTorch.

Boxes are ``[xmin, ymin, xmax, ymax]`` (xyxy) unless stated otherwise. The FCOS
head regresses ``(l, t, r, b)`` distances from a location to the four box sides;
``ltrb_to_xyxy`` / ``xyxy_to_ltrb`` convert between the two.
"""

from __future__ import annotations

import torch
from torch import Tensor


def box_area(boxes: Tensor) -> Tensor:
    """Area of each box. ``boxes``: (..., 4) xyxy. Returns (...,)."""
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return w * h


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Pairwise IoU. ``boxes_a``: (N,4), ``boxes_b``: (M,4) -> (N,M)."""
    area_a = box_area(boxes_a)  # (N,)
    area_b = box_area(boxes_b)  # (M,)

    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (N,M,2)
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]  # (N,M)

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-7)


def giou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise GIoU loss (1 - GIoU). ``pred``/``target``: (N,4) xyxy.

    Returns (N,) per-box loss; caller reduces (e.g. centerness-weighted mean).
    """
    # Intersection.
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area_p = box_area(pred)
    area_t = box_area(target)
    union = area_p + area_t - inter + eps
    iou = inter / union

    # Smallest enclosing box.
    enc_lt = torch.min(pred[:, :2], target[:, :2])
    enc_rb = torch.max(pred[:, 2:], target[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[:, 0] * enc_wh[:, 1] + eps

    giou = iou - (enc_area - union) / enc_area
    return 1.0 - giou


def ltrb_to_xyxy(locations: Tensor, ltrb: Tensor) -> Tensor:
    """Decode (l,t,r,b) distances at ``locations`` into xyxy boxes.

    ``locations``: (N,2) pixel centers (x, y). ``ltrb``: (N,4) non-negative
    distances in pixels. Returns (N,4) xyxy.
    """
    x1 = locations[:, 0] - ltrb[:, 0]
    y1 = locations[:, 1] - ltrb[:, 1]
    x2 = locations[:, 0] + ltrb[:, 2]
    y2 = locations[:, 1] + ltrb[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=1)


def xyxy_to_ltrb(locations: Tensor, boxes: Tensor) -> Tensor:
    """Encode xyxy boxes as (l,t,r,b) distances from ``locations``.

    ``locations``: (N,2) pixel centers. ``boxes``: (N,4) xyxy. Returns (N,4).
    """
    left = locations[:, 0] - boxes[:, 0]
    top = locations[:, 1] - boxes[:, 1]
    right = boxes[:, 2] - locations[:, 0]
    bottom = boxes[:, 3] - locations[:, 1]
    return torch.stack([left, top, right, bottom], dim=1)


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Single-class NMS, hand-written. Returns kept indices sorted by score.

    ``boxes``: (N,4) xyxy. ``scores``: (N,).
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]  # (rest,)
        remaining = (ious <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
        order = order[1:][remaining]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def soft_nms(
    boxes: Tensor, scores: Tensor, sigma: float = 0.5, score_thresh: float = 1e-3
) -> tuple[Tensor, Tensor]:
    """Gaussian Soft-NMS (hand-written). Instead of removing overlapping boxes,
    decay their scores by ``exp(-iou^2 / sigma)``.

    Returns ``(kept_indices, decayed_scores)`` ordered by descending final score.
    For per-class behaviour, offset boxes by class first (see ``soft_nms_per_class``).
    """
    if boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=boxes.device),
            torch.empty((0,), device=boxes.device),
        )
    boxes = boxes.clone().float()
    scores = scores.clone().float()
    idxs = torch.arange(boxes.shape[0], device=boxes.device)

    out_idx, out_score = [], []
    while scores.numel() > 0:
        m = int(torch.argmax(scores))
        out_idx.append(int(idxs[m]))
        out_score.append(float(scores[m]))

        keep = torch.ones(scores.numel(), dtype=torch.bool, device=boxes.device)
        keep[m] = False
        pivot = boxes[m].unsqueeze(0)
        boxes, scores, idxs = boxes[keep], scores[keep], idxs[keep]
        if scores.numel() == 0:
            break
        ious = box_iou(pivot, boxes)[0]
        scores = scores * torch.exp(-(ious ** 2) / sigma)
        survive = scores >= score_thresh
        boxes, scores, idxs = boxes[survive], scores[survive], idxs[survive]

    device = idxs.device
    return (
        torch.tensor(out_idx, dtype=torch.long, device=device),
        torch.tensor(out_score, device=device),
    )


def soft_nms_per_class(
    boxes: Tensor, scores: Tensor, labels: Tensor, sigma: float = 0.5,
    score_thresh: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Per-class Soft-NMS via the coordinate-offset trick (cross-class IoU=0 → no
    decay). Returns ``(kept_indices, decayed_scores)`` into the input arrays."""
    if boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=boxes.device),
            torch.empty((0,), device=boxes.device),
        )
    max_coord = boxes.max()
    shifted = boxes + (labels.to(boxes) * (max_coord + 1))[:, None]
    return soft_nms(shifted, scores, sigma, score_thresh)


def batched_nms(
    boxes: Tensor, scores: Tensor, labels: Tensor, iou_threshold: float
) -> Tensor:
    """Per-class NMS via coordinate offsetting (one NMS call).

    Boxes of different classes never suppress each other because each class is
    shifted into a disjoint coordinate band. Returns kept indices.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    # Offset each class by a stride larger than any coordinate.
    max_coord = boxes.max()
    offsets = labels.to(boxes) * (max_coord + 1)
    shifted = boxes + offsets[:, None]
    return nms(shifted, scores, iou_threshold)
