"""Detector-free thumbnail router for mutually exclusive input resolutions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .router import perspective_band_descriptor


class DepthwiseSeparableBlock(nn.Module):
    """Small stride-2 block used by the thumbnail encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


def thumbnail_utility_descriptor(feature: torch.Tensor, bands: int = 4) -> torch.Tensor:
    """Combine fixed vertical-band and global channel statistics."""
    band_descriptor = perspective_band_descriptor(feature, bands)
    global_descriptor = torch.cat((feature.mean(dim=(2, 3)), feature.amax(dim=(2, 3))), dim=1)
    return torch.cat((band_descriptor, global_descriptor), dim=1)


class ThumbnailUtilityRouter(nn.Module):
    """Predict high-resolution detection utility before running the detector."""

    def __init__(self, channels: int = 32, bands: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.bands = bands
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            DepthwiseSeparableBlock(16, 24),
            DepthwiseSeparableBlock(24, channels),
        )
        descriptor_dim = channels * (bands * 2 + 2)
        self.head = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("thumbnail input must have BCHW shape with three RGB channels")
        feature = self.encoder(inputs)
        return self.head(thumbnail_utility_descriptor(feature, self.bands)).squeeze(1)


def weighted_pairwise_ranking_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    max_weight: float = 4.0,
) -> torch.Tensor:
    """Rank samples by continuous utility, emphasizing pairs with larger utility gaps."""
    if scores.ndim != 1 or utilities.ndim != 1 or scores.shape != utilities.shape:
        raise ValueError("scores and utilities must be one-dimensional tensors with matching shape")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    utility_difference = utilities[:, None] - utilities[None, :]
    pair_mask = torch.triu(torch.ones_like(utility_difference, dtype=torch.bool), diagonal=1)
    pair_mask &= utility_difference != 0
    if not pair_mask.any():
        return scores.sum() * 0.0
    score_difference = scores[:, None] - scores[None, :]
    absolute_gap = utility_difference.abs()
    gap_scale = absolute_gap[pair_mask].mean().clamp_min(torch.finfo(scores.dtype).eps)
    weights = (absolute_gap / gap_scale).clamp(max=max_weight)
    losses = F.softplus(-utility_difference.sign() * score_difference)
    return (losses[pair_mask] * weights[pair_mask]).sum() / weights[pair_mask].sum()


def standardized_utility_regression_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    utility_mean: float | torch.Tensor,
    utility_std: float | torch.Tensor,
) -> torch.Tensor:
    """Anchor router scores to globally standardized continuous utilities."""
    mean = torch.as_tensor(utility_mean, dtype=utilities.dtype, device=utilities.device)
    std = torch.as_tensor(utility_std, dtype=utilities.dtype, device=utilities.device)
    if torch.any(std <= 0):
        raise ValueError("utility_std must be positive")
    targets = (utilities - mean) / std
    return F.smooth_l1_loss(scores, targets)


def utility_routing_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    utility_mean: float | torch.Tensor,
    utility_std: float | torch.Tensor,
    regression_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the frozen ranking-plus-regression objective and its components."""
    if regression_weight < 0:
        raise ValueError("regression_weight must be non-negative")
    ranking = weighted_pairwise_ranking_loss(scores, utilities)
    regression = standardized_utility_regression_loss(scores, utilities, utility_mean, utility_std)
    total = ranking + regression_weight * regression
    return total, {"ranking": ranking, "regression": regression}
