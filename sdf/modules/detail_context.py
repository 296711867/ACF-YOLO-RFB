"""Residual detail-context injection for the shallow P2 fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualDetailContextConcat(nn.Module):
    """Inject aligned P1 detail and context into P2 before the unchanged top-down concatenation."""

    def __init__(self, channels: list[int], dimension: int = 1):
        """Initialize P1 alignment and an exact-zero P2 residual."""
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"ResidualDetailContextConcat expects [top_down, P2, P1], got {channels}")
        _, p2_channels, p1_channels = channels
        self.dimension = dimension
        self.align = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(p1_channels, p2_channels, 1, bias=False),
            nn.BatchNorm2d(p2_channels),
            nn.SiLU(inplace=True),
        )
        self.residual = nn.Conv2d(2 * p2_channels, p2_channels, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Return the official concat plus a learned zero-initialized detail-context residual on P2."""
        top_down, p2, p1 = inputs
        aligned = self.align(p1)
        if top_down.shape[-2:] != p2.shape[-2:] or aligned.shape[-2:] != p2.shape[-2:]:
            raise ValueError(
                "ResidualDetailContextConcat spatial mismatch: "
                f"top_down={top_down.shape}, P2={p2.shape}, aligned_P1={aligned.shape}"
            )
        context = F.avg_pool2d(aligned, kernel_size=3, stride=1, padding=1)
        detail = aligned - context
        enhanced_p2 = p2 + self.residual(torch.cat((detail, context), dim=1))
        return torch.cat((top_down, enhanced_p2), dim=self.dimension)
