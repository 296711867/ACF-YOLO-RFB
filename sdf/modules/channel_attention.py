"""Near-identity channel attention for shallow P2 feature fusion."""

from __future__ import annotations

import torch
from torch import nn


class IdentityChannelAttentionConcat(nn.Module):
    """Attend the shallow input and concatenate it without shifting downstream layer indices."""

    def __init__(self, channels: list[int], dimension: int = 1):
        """Initialize an exact-identity gate for the second concat input."""
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"IdentityChannelAttentionConcat expects two inputs, got {channels}")
        self.dimension = dimension
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels[1], channels[1], 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Reweight the shallow input with gains centered at one, then concatenate both inputs."""
        high, shallow = inputs
        shallow = shallow * (2.0 * self.fc(self.pool(shallow)).sigmoid())
        return torch.cat((high, shallow), self.dimension)
