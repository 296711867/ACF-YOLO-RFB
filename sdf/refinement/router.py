"""Learnable perspective-band router for adaptive full-image resolution."""

from __future__ import annotations

import torch
from torch import nn


def perspective_band_descriptor(feature: torch.Tensor, bands: int = 4) -> torch.Tensor:
    """Pool channel means and maxima in fixed vertical bands."""
    if feature.ndim != 4:
        raise ValueError("feature must have BCHW shape")
    if bands <= 0 or feature.shape[2] % bands:
        raise ValueError("bands must evenly divide feature height")
    descriptors = []
    for band in feature.chunk(bands, dim=2):
        descriptors.extend((band.mean(dim=(2, 3)), band.amax(dim=(2, 3))))
    return torch.cat(descriptors, dim=1)


class PerspectiveBandRouter(nn.Module):
    """Score the value of a 960 recheck from frozen P2 feature statistics."""

    def __init__(self, channels: int = 32, bands: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.channels = channels
        self.bands = bands
        descriptor_dim = channels * bands * 2
        self.head = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward_descriptor(self, descriptor: torch.Tensor) -> torch.Tensor:
        """Return one routing logit per precomputed descriptor."""
        return self.head(descriptor).squeeze(1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Return one routing logit per P2 feature map."""
        if feature.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, received {feature.shape[1]}")
        return self.forward_descriptor(perspective_band_descriptor(feature, self.bands))
