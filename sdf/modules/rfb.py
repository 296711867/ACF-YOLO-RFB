"""Multi-branch dilated context block (RFB-style, Liu et al., ECCV 2018) for P29.

Three parallel branches with dilation rates 1/2/3 enlarge the receptive field
with standard (non-depthwise) convolutions — TRT-safe by construction, unlike
the large-kernel depthwise LSK family. Branch width 16 for 64-channel input;
concatenated branches are fused by 1x1 and added residually.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RFBlock(nn.Module):
    """Residual multi-branch dilated context module."""

    def __init__(self, channels: int, branch_width: int = 16, dilations: tuple[int, ...] = (1, 2, 3)):
        """Create the three dilated branches and the fusion conv."""
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, branch_width, 1, bias=False),
                nn.BatchNorm2d(branch_width),
                nn.Conv2d(branch_width, branch_width, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(branch_width),
            )
            for rate in dilations
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_width * len(dilations), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse multi-scale context branches and add residually."""
        out = torch.cat([F.silu(branch(x)) for branch in self.branches], dim=1)
        return x + self.fuse(out)
