"""FCOS loss: focal (classification) + GIoU (box) + BCE (centerness).

The detector outputs and the targets are all in pixel space. Predictions are
flattened across pyramid levels and the batch, targets are assigned per image via
``utils.targets``, then the three loss terms are combined.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .boxes import giou_loss, ltrb_to_xyxy
from .targets import OBJECT_SIZES, build_point_metadata, compute_fcos_targets


def sigmoid_focal_loss(
    logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> Tensor:
    """Element-wise sigmoid focal loss; caller reduces (sum / num_pos)."""
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * (1 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * loss


def _flatten(outs, channels: int) -> Tensor:
    """List of (B,C,H,W) over levels -> (B, sum(H*W), C)."""
    flat = [o.permute(0, 2, 3, 1).reshape(o.shape[0], -1, channels) for o in outs]
    return torch.cat(flat, dim=1)


class FCOSLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        strides=(8, 16, 32, 64, 128),
        radius: float = 1.5,
        reg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.radius = radius
        self.reg_weight = reg_weight

    def forward(self, cls_outs, reg_outs, ctr_outs, locations_list, gt_boxes, gt_labels):
        device = cls_outs[0].device
        batch = cls_outs[0].shape[0]
        c = self.num_classes

        points, point_strides, point_ranges = build_point_metadata(
            locations_list, self.strides, OBJECT_SIZES, device
        )
        num_points = points.shape[0]

        # Per-image target assignment.
        labels_b, reg_b, ctr_b = [], [], []
        for i in range(batch):
            lab, reg_t, ctr_t = compute_fcos_targets(
                points, point_strides, point_ranges,
                gt_boxes[i].to(device), gt_labels[i].to(device), self.radius,
            )
            labels_b.append(lab)
            reg_b.append(reg_t)
            ctr_b.append(ctr_t)
        labels = torch.stack(labels_b)        # (B,N)
        reg_targets = torch.stack(reg_b)       # (B,N,4)
        ctr_targets = torch.stack(ctr_b)       # (B,N)

        # Flatten predictions to (B*N, ...).
        cls_pred = _flatten(cls_outs, c).reshape(-1, c)
        reg_pred = _flatten(reg_outs, 4).reshape(-1, 4)
        ctr_pred = _flatten(ctr_outs, 1).reshape(-1)

        flat_labels = labels.reshape(-1)
        pos_inds = (flat_labels >= 0).nonzero(as_tuple=False).squeeze(1)
        num_pos = max(pos_inds.numel(), 1)

        # Classification: focal over all points.
        cls_targets = torch.zeros_like(cls_pred)
        if pos_inds.numel() > 0:
            cls_targets[pos_inds, flat_labels[pos_inds]] = 1.0
        cls_loss = sigmoid_focal_loss(cls_pred, cls_targets).sum() / num_pos

        if pos_inds.numel() == 0:
            reg_loss = reg_pred.sum() * 0.0
            ctr_loss = ctr_pred.sum() * 0.0
            return {
                "loss": cls_loss,
                "cls": cls_loss.detach(),
                "reg": reg_loss.detach(),
                "ctr": ctr_loss.detach(),
                "num_pos": torch.tensor(0.0, device=device),
            }

        loc_rep = points.repeat(batch, 1)             # (B*N,2), image-major
        loc_pos = loc_rep[pos_inds]
        pred_boxes = ltrb_to_xyxy(loc_pos, reg_pred[pos_inds])
        tgt_boxes = ltrb_to_xyxy(loc_pos, reg_targets.reshape(-1, 4)[pos_inds])
        tgt_ctr = ctr_targets.reshape(-1)[pos_inds]

        giou = giou_loss(pred_boxes, tgt_boxes)        # (P,)
        reg_loss = (giou * tgt_ctr).sum() / tgt_ctr.sum().clamp(min=1e-6)
        ctr_loss = F.binary_cross_entropy_with_logits(ctr_pred[pos_inds], tgt_ctr)

        loss = cls_loss + self.reg_weight * reg_loss + ctr_loss
        return {
            "loss": loss,
            "cls": cls_loss.detach(),
            "reg": reg_loss.detach(),
            "ctr": ctr_loss.detach(),
            "num_pos": torch.tensor(float(pos_inds.numel()), device=device),
        }
