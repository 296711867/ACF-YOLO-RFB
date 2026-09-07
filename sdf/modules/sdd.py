"""Small-object detail-preserving downsampling."""

from __future__ import annotations

import torch
from torch import nn


class SDD(nn.Module):
    """Downsample by lossless spatial rearrangement followed by lightweight channel mixing."""

    def __init__(self, c1: int, c2: int):
        """Initialize SDD with input and output channel counts."""
        super().__init__()
        self.space_to_depth = nn.PixelUnshuffle(2)
        self.reduce = nn.Conv2d(4 * c1, c2, 1, bias=False)
        self.depthwise = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
        self.project = nn.Conv2d(c2, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange spatial samples and return a stride-2 output feature map."""
        x = self.space_to_depth(x)
        x = self.reduce(x)
        x = self.depthwise(x)
        return self.act(self.bn(self.project(x)))
