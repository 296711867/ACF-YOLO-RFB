"""Coordinate Attention (Hou et al., CVPR 2021) for the P29 screen.

Factorizes spatial attention into two direction-aware 1D pooling paths,
encodes long-range dependencies along H and W with a shared 1x1 transform,
and applies the resulting sigmoid gates. Faithful to the paper formulation;
the bottleneck width uses max(8, channels // 32) so 64-channel nano features
keep 8 bottleneck channels (pre-registered in the P29 protocol).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CoordAtt(nn.Module):
    """Direction-aware channel-spatial attention with a lightweight bottleneck."""

    def __init__(self, channels: int):
        """Create pooling paths, the shared bottleneck, and the two gate convs."""
        super().__init__()
        mid = max(8, channels // 32)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv_h = nn.Conv2d(mid, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, 1, bias=False)

    @staticmethod
    def _split_by_heights(tensor: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the concatenated (b, c, h + w, 1) descriptor back into H and W parts."""
        return torch.split(tensor, [h, w], dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coordinate attention gates along H and W."""
        b, c, h, w = x.shape
        x_h = self.pool_h(x)  # (b, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b, c, w, 1)
        descriptor = F.silu(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        part_h, part_w = self._split_by_heights(descriptor, h, w)
        gate_h = torch.sigmoid(self.conv_h(part_h))  # (b, c, h, 1)
        gate_w = torch.sigmoid(self.conv_w(part_w))  # (b, c, w, 1)
        return x * gate_h * gate_w.permute(0, 1, 3, 2)
