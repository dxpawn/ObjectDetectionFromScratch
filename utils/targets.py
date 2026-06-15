"""FCOS target assignment (pixel space, with center sampling).

For every feature-map location (across all pyramid levels) decide whether it is a
positive sample for some ground-truth box and, if so, its regression target
(l,t,r,b distances in pixels), class label, and centerness target.

A location is positive for a GT box if:
  1. it lies inside a center-sampling sub-box (gt center ± radius*stride, clipped
     to the box), and
  2. the box's max side-distance falls in that level's regression range.
Ties (a location inside several boxes) go to the smallest-area box.
"""

from __future__ import annotations

import torch
from torch import Tensor

INF = 1e9
# Per-level max-distance ranges (pixels), matching FCOS for strides 8..128.
OBJECT_SIZES = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF))


def build_point_metadata(locations_list, strides, ranges=OBJECT_SIZES, device=None):
    """Concatenate per-level locations and tag each point with its stride and
    regression range. Returns (points (N,2), strides (N,), ranges (N,2))."""
    pts, sp, rg = [], [], []
    for locs, stride, (lo, hi) in zip(locations_list, strides, ranges):
        n = locs.shape[0]
        dev = device or locs.device
        pts.append(locs.to(dev))
        sp.append(torch.full((n,), float(stride), device=dev))
        rg.append(torch.tensor([[lo, hi]], device=dev, dtype=torch.float32).expand(n, 2))
    return torch.cat(pts), torch.cat(sp), torch.cat(rg)


def compute_fcos_targets(
    points: Tensor,
    point_strides: Tensor,
    point_ranges: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    radius: float = 1.5,
    eps: float = 1e-6,
):
    """Single-image assignment.

    Returns ``(labels (N,), reg_targets (N,4), centerness (N,))`` where ``labels``
    is the GT class index for positives and ``-1`` for background.
    """
    n = points.shape[0]
    device = points.device
    if gt_boxes.numel() == 0:
        return (
            torch.full((n,), -1, dtype=torch.long, device=device),
            torch.zeros((n, 4), device=device),
            torch.zeros((n,), device=device),
        )

    xs = points[:, 0:1]  # (N,1)
    ys = points[:, 1:2]
    gx1, gy1, gx2, gy2 = (gt_boxes[:, i][None, :] for i in range(4))  # (1,Ng)

    left = xs - gx1
    top = ys - gy1
    right = gx2 - xs
    bottom = gy2 - ys
    reg = torch.stack([left, top, right, bottom], dim=2)  # (N,Ng,4)
    max_reg = reg.max(dim=2).values  # (N,Ng)

    # Center sampling: sub-box around each GT center, clipped to the box.
    cx = (gx1 + gx2) / 2
    cy = (gy1 + gy2) / 2
    radius_px = point_strides[:, None] * radius  # (N,1)
    xmin = torch.max(cx - radius_px, gx1)
    xmax = torch.min(cx + radius_px, gx2)
    ymin = torch.max(cy - radius_px, gy1)
    ymax = torch.min(cy + radius_px, gy2)
    in_center = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

    lo = point_ranges[:, 0:1]
    hi = point_ranges[:, 1:2]
    in_level = (max_reg >= lo) & (max_reg <= hi)

    is_pos = in_center & in_level  # (N,Ng)

    areas = ((gx2 - gx1) * (gy2 - gy1)).expand(n, -1).clone()  # (N,Ng)
    areas[~is_pos] = INF
    min_area, gt_idx = areas.min(dim=1)  # (N,)
    pos = min_area < INF

    labels = torch.full((n,), -1, dtype=torch.long, device=device)
    labels[pos] = gt_labels[gt_idx[pos]]

    reg_targets = reg[torch.arange(n, device=device), gt_idx]  # (N,4)
    reg_targets[~pos] = 0.0

    l_, t_, r_, b_ = reg_targets.unbind(dim=1)
    lr_min = torch.min(l_, r_)
    lr_max = torch.max(l_, r_).clamp(min=eps)
    tb_min = torch.min(t_, b_)
    tb_max = torch.max(t_, b_).clamp(min=eps)
    centerness = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))
    centerness[~pos] = 0.0

    return labels, reg_targets, centerness
