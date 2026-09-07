"""Size-aware Normalized Wasserstein Distance loss for axis-aligned boxes."""

from __future__ import annotations

import torch
from torch import nn


class SizeAwareNWDLoss(nn.Module):
    """Blend CIoU loss with NWD according to target area at a fixed reference scale."""

    def __init__(
        self,
        constant: float = 32.0,
        area_tau: float = 1024.0,
        alpha_max: float = 0.8,
        area_limit: float = 0.0,
        reference_size: float = 640.0,
    ):
        """Initialize the NWD scale, area decay, blend cap, and reference image size."""
        super().__init__()
        if constant <= 0 or area_tau <= 0 or reference_size <= 0:
            raise ValueError("constant, area_tau, and reference_size must be positive")
        if area_limit < 0:
            raise ValueError("area_limit must be nonnegative")
        if not 0 <= alpha_max <= 1:
            raise ValueError("alpha_max must be in [0, 1]")
        self.constant = constant
        self.area_tau = area_tau
        self.alpha_max = alpha_max
        self.area_limit = area_limit
        self.reference_size = reference_size

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        ciou: torch.Tensor,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-box size-aware NWD-CIoU loss for pixel-space xyxy boxes."""
        pred = pred_boxes.float()
        target = target_boxes.float()
        height, width = image_size.float().unbind()
        scale_x, scale_y = self.reference_size / width, self.reference_size / height
        scale = torch.stack((scale_x, scale_y, scale_x, scale_y)).to(device=target.device)
        pred = pred * scale
        target = target * scale

        pred_xy1, pred_xy2 = pred.chunk(2, dim=-1)
        target_xy1, target_xy2 = target.chunk(2, dim=-1)
        pred_center = (pred_xy1 + pred_xy2) / 2
        target_center = (target_xy1 + target_xy2) / 2
        pred_size = (pred_xy2 - pred_xy1).clamp_min(0)
        target_size = (target_xy2 - target_xy1).clamp_min(0)

        wasserstein = torch.linalg.vector_norm(
            torch.cat((pred_center - target_center, (pred_size - target_size) / 2), dim=-1), dim=-1, keepdim=True
        )
        nwd = torch.exp(-wasserstein / self.constant)
        target_area = target_size.prod(dim=-1, keepdim=True)
        alpha = torch.exp(-target_area / self.area_tau).clamp(max=self.alpha_max)
        if self.area_limit:
            alpha = alpha * (target_area < self.area_limit)
        return (1 - alpha) * (1 - ciou.float()) + alpha * (1 - nwd)
