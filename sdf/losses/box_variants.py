"""Inner-IoU and MPDIoU bounding-box regression terms for P27 loss screening.

Inner-IoU (Zhang & Ren, 2023): scales auxiliary boxes about their centers by a
ratio r < 1 and evaluates the CIoU penalty on the auxiliary pair, focusing
gradient on low-IoU pairs; documented to help small-object localization.

MPDIoU (Ma et al., 2023): IoU minus the squared minimum-point distances (both
top-left and bottom-right corners) normalized by the enclosing-box diagonal
squared; sensitive to extreme aspect ratios and sub-pixel misalignment.

Both replace only the IoU term consumed by ``BboxLoss``; DFL and classification
losses are untouched, and the inference graph is unchanged.
"""

from __future__ import annotations

import math

import torch


def _scale_about_center(box: torch.Tensor, ratio: float) -> torch.Tensor:
    """Return xyxy boxes shrunk (ratio<1) or expanded (ratio>1) about their centers."""
    center = (box[..., :2] + box[..., 2:]) / 2
    half = (box[..., 2:] - box[..., :2]) / 2 * ratio
    return torch.cat((center - half, center + half), dim=-1)


def inner_bbox_iou(pred: torch.Tensor, target: torch.Tensor, ratio: float, eps: float = 1e-7) -> torch.Tensor:
    """CIoU computed on center-scaled auxiliary boxes (Inner-IoU with ratio<1)."""
    inner_pred = _scale_about_center(pred, ratio)
    inner_target = _scale_about_center(target, ratio)
    from ultralytics.utils.metrics import bbox_iou

    return bbox_iou(inner_pred, inner_target, xywh=False, CIoU=True, eps=eps)


def mpdiou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """MPDIoU: IoU minus squared corner-point distances over enclosing diagonal squared."""
    b1_x1, b1_y1, b1_x2, b1_y2 = pred.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target.chunk(4, -1)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    w1 = (b1_x2 - b1_x1).clamp_(min=0)
    h1 = (b1_y2 - b1_y1).clamp_(min=0)
    w2 = (b2_x2 - b2_x1).clamp_(min=0)
    h2 = (b2_y2 - b2_y1).clamp_(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    diagonal_squared = cw.pow(2) + ch.pow(2) + eps
    d1_squared = (b1_x1 - b2_x1).pow(2) + (b1_y1 - b2_y1).pow(2)
    d2_squared = (b1_x2 - b2_x2).pow(2) + (b1_y2 - b2_y2).pow(2)
    return iou - (d1_squared + d2_squared) / diagonal_squared


def d_interpiou(
    pred: torch.Tensor, target: torch.Tensor, lv: float = 0.6, hv: float = 0.99, eps: float = 1e-7
) -> torch.Tensor:
    """D-InterpIoU similarity (arXiv:2507.12420): IoU + interpolated-box IoU - 1.

    Line-for-line the reference ``d_interpiou`` (github.com/NuayHL/interpiou,
    xywh=False path, verbatim fixture in tests/fixtures): the interpolation
    coefficient adapts to the current IoU (clamped to [lv, hv], reference
    defaults 0.6/0.99), keeping gradients alive in non-overlapping cases. The
    pipeline consumes ``1 - returned`` as the loss, matching the reference
    loss ``2 - IoU - IoU_interp``.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = pred.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target.chunk(4, -1)
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1 + eps
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    interp_coe = torch.clamp((1 - iou.detach()), min=lv, max=hv)

    bi_x1 = (1 - interp_coe) * b1_x1 + interp_coe * b2_x1
    bi_y1 = (1 - interp_coe) * b1_y1 + interp_coe * b2_y1
    bi_x2 = (1 - interp_coe) * b1_x2 + interp_coe * b2_x2
    bi_y2 = (1 - interp_coe) * b1_y2 + interp_coe * b2_y2

    inter_i = (torch.min(bi_x2, b2_x2) - torch.max(bi_x1, b2_x1)).clamp_(0) * (
        torch.min(bi_y2, b2_y2) - torch.max(bi_y1, b2_y1)
    ).clamp_(0)

    wi, hi = bi_x2 - bi_x1 + eps, bi_y2 - bi_y1 + eps
    w2, h2 = b2_x2 - b2_x1 + eps, b2_y2 - b2_y1 + eps

    union_i = wi * hi + w2 * h2 - inter_i + eps
    iou_i = inter_i / union_i

    return iou + iou_i - 1


def piou2(pred: torch.Tensor, target: torch.Tensor, lambda_: float = 1.3, eps: float = 1e-7) -> torch.Tensor:
    """PIoU2 similarity (Powerful-IoU, Neural Networks 2024): ``1 - L_PIoU2``.

    Mathematics line-for-line the reference ``piou(..., PIoU2=True)`` (github
    .com/fppccc/Powerful-IoU, verbatim fixture in tests/fixtures): size-adaptive
    penalty P over corner-distance ratios, L_v1 = 2 - iou - exp(-P^2), and the
    nonmonotonic focusing factor 3x*exp(-x^2) with Lambda=1.3 (reference
    default). The reference returns a loss L; this pipeline consumes
    ``1 - returned`` as the loss, so this returns ``1 - L``. The reference's
    unused cw/ch variables are not reproduced (no effect on values).
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = pred.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target.chunk(4, -1)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1 + eps
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
    dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
    dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
    dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
    p = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
    l_v1 = 1 - iou - torch.exp(-p**2) + 1

    q = torch.exp(-p)
    x = q * lambda_
    loss = 3 * x * torch.exp(-x**2) * l_v1
    return 1.0 - loss
