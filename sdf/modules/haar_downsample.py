"""Fixed Haar wavelet downsampling with learned subband fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def haar_dwt(x: torch.Tensor) -> torch.Tensor:
    """Return orthonormal Haar DWT subbands [LL, LH, HL, HH] at half resolution."""
    height, width = x.shape[-2:]
    if height % 2 or width % 2:
        x = F.pad(x, (0, width % 2, 0, height % 2), mode="replicate")
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return torch.cat((ll, lh, hl, hh), dim=1)


def haar_idwt(subbands: torch.Tensor) -> torch.Tensor:
    """Invert :func:`haar_dwt` on [LL, LH, HL, HH] subbands back to full resolution."""
    ll, lh, hl, hh = subbands.chunk(4, dim=1)
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    height, width = subbands.shape[-2:]
    restored = torch.empty(
        subbands.shape[0],
        subbands.shape[1] // 4,
        height * 2,
        width * 2,
        dtype=subbands.dtype,
        device=subbands.device,
    )
    restored[..., 0::2, 0::2] = a
    restored[..., 0::2, 1::2] = b
    restored[..., 1::2, 0::2] = c
    restored[..., 1::2, 1::2] = d
    return restored


class HaarDownsample(nn.Module):
    """Lossless stride-2 downsample: fixed Haar DWT plus a learned subband fusion convolution."""

    def __init__(self, c1: int, c2: int, kernel_size: int = 3):
        """Store channel plan and build the fusion convolution over 4*c1 subband channels."""
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.fusion = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fused subband features at half input resolution."""
        return self.fusion(haar_dwt(x))
