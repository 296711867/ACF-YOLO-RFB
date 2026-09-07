"""Permutation-invariant router over low-resolution scout detections."""

from __future__ import annotations

import torch
from torch import nn

from .pre_router import weighted_pairwise_ranking_loss


class ScoutDetectionValueRouter(nn.Module):
    """Predict marginal high-resolution utility from an unordered detection set."""

    def __init__(
        self,
        num_classes: int = 10,
        token_features: int = 7,
        class_embedding: int = 8,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or token_features <= 0 or class_embedding <= 0 or hidden <= 0:
            raise ValueError("model dimensions must be positive")
        self.num_classes = num_classes
        self.token_features = token_features
        self.class_embedding_dim = class_embedding
        self.hidden = hidden
        self.class_embedding = nn.Embedding(num_classes + 1, class_embedding, padding_idx=num_classes)
        self.token_encoder = nn.Sequential(
            nn.Linear(token_features + class_embedding, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.attention = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + 1),
            nn.Linear(hidden * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        classes: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_features:
            raise ValueError(f"tokens must have BK{self.token_features} shape")
        if classes.shape != tokens.shape[:2] or mask.shape != tokens.shape[:2]:
            raise ValueError("classes and mask must match the first two token dimensions")
        if mask.dtype is not torch.bool:
            raise ValueError("mask must be boolean")
        valid_classes = classes[mask]
        if valid_classes.numel() and (valid_classes.min() < 0 or valid_classes.max() >= self.num_classes):
            raise ValueError("valid class ids are outside the configured range")

        padding_classes = torch.full_like(classes, self.num_classes)
        embedded_classes = self.class_embedding(torch.where(mask, classes, padding_classes))
        encoded = self.token_encoder(torch.cat((tokens, embedded_classes), dim=-1))
        encoded = encoded * mask.unsqueeze(-1)

        attention_logits = self.attention(encoded).squeeze(-1).masked_fill(~mask, -1e4)
        attention_weights = torch.softmax(attention_logits, dim=1) * mask
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attention_pool = (encoded * attention_weights.unsqueeze(-1)).sum(dim=1)
        sum_pool = encoded.sum(dim=1) / max(tokens.shape[1], 1)
        count_fraction = mask.float().mean(dim=1, keepdim=True)
        output = self.head(torch.cat((attention_pool, sum_pool, count_fraction), dim=1))
        return output[:, 0], output[:, 1].clamp(-6.0, 6.0)


def scout_value_loss(
    means: torch.Tensor,
    log_variances: torch.Tensor,
    utilities: torch.Tensor,
    utility_mean: float | torch.Tensor,
    utility_std: float | torch.Tensor,
    regression_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine continuous ranking with standardized heteroscedastic regression."""
    if means.ndim != 1 or log_variances.shape != means.shape or utilities.shape != means.shape:
        raise ValueError("means, log_variances, and utilities must be matching vectors")
    if regression_weight < 0:
        raise ValueError("regression_weight must be non-negative")
    mean = torch.as_tensor(utility_mean, dtype=utilities.dtype, device=utilities.device)
    std = torch.as_tensor(utility_std, dtype=utilities.dtype, device=utilities.device)
    if torch.any(std <= 0):
        raise ValueError("utility_std must be positive")
    targets = (utilities - mean) / std
    ranking = weighted_pairwise_ranking_loss(means, utilities)
    gaussian_nll = 0.5 * (log_variances + (targets - means).square() * torch.exp(-log_variances)).mean()
    total = ranking + regression_weight * gaussian_nll
    return total, {"ranking": ranking, "gaussian_nll": gaussian_nll}


def ensemble_lower_confidence_score(
    means: torch.Tensor,
    log_variances: torch.Tensor,
) -> torch.Tensor:
    """Aggregate seed predictions as mean minus fixed total standard deviation."""
    if means.ndim != 2 or log_variances.shape != means.shape or means.shape[0] < 2:
        raise ValueError("ensemble inputs must have matching SB shape with at least two seeds")
    predictive_mean = means.mean(dim=0)
    total_variance = torch.exp(log_variances).mean(dim=0) + means.var(dim=0, unbiased=False)
    return predictive_mean - total_variance.clamp_min(0.0).sqrt()
