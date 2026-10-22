"""Efficient Multi-Scale Attention (Ouyang et al., ICASSP 2023) for the P29 screen.

Reference implementation: github.com/YOLOonMe/EMA-attention-module (code for the
ICASSP 2023 paper "Efficient Multi-Scale Attention Module with Cross-Spatial
Learning", arXiv:2305.13563), fetched verbatim on 2026-09-30 once the network
blockage recorded in the P29 protocol §1 cleared. Direction-aware 1D pooling
feeds a shared 1x1 transform; a GroupNorm-gated branch and a 3x3 branch are
combined by cross-spatial matmul attention. factor=32 is the reference default
(frozen, no sweep): at the 64-channel L19 insertion point this means c//g=2,
so the module adds only 48 parameters. The forward pass is line-for-line the
reference one; numerical equivalence is enforced by
tests/test_p29_modules.py::TestEMAAttention against tests/fixtures/ema_reference.py.
"""

from __future__ import annotations

import torch
from torch import nn


class EMAAttention(nn.Module):
    """Grouped cross-spatial multi-scale attention gate (reference factor=32)."""

    def __init__(self, channels: int, factor: int = 32):
        """Create the grouped pooling, normalization, and conv attention structure."""
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped cross-spatial attention (faithful to the reference forward)."""
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)  # b*g, c//g, h, 1
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # b*g, c//g, w, 1
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # b*g, c//g, h+w, 1
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
