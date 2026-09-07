"""Dynamic weighted fusion for high-resolution feature maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DySample(nn.Module):
    """Upsample with zero-initialized, content-dependent sampling offsets."""

    def __init__(self, channels: int, scale: int = 2, max_offset: float = 0.25):
        """Initialize the offset predictor."""
        super().__init__()
        self.scale = scale
        self.max_offset = max_offset
        self.offset = nn.Conv2d(channels, 2 * scale**2, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample a scale-times larger feature map with learned subpixel offsets."""
        _, _, height, width = x.shape
        output_height, output_width = height * self.scale, width * self.scale
        offsets = F.pixel_shuffle(self.offset(x), self.scale)
        offsets = torch.tanh(offsets) * self.max_offset

        ys = (torch.arange(output_height, device=x.device, dtype=x.dtype) + 0.5) * (2.0 / output_height) - 1.0
        xs = (torch.arange(output_width, device=x.device, dtype=x.dtype) + 0.5) * (2.0 / output_width) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        normalization = x.new_tensor((2.0 / width, 2.0 / height))
        grid = base_grid + offsets.permute(0, 2, 3, 1) * normalization
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


class DWF(nn.Module):
    """Fuse a low-resolution semantic feature with a high-resolution spatial feature."""

    def __init__(self, channels: list[int], c2: int, eps: float = 1e-4):
        """Initialize projections, dynamic sampling, global weights, local gate, and lightweight fusion."""
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"DWF expects [high_channels, low_channels], got {channels}")
        high_channels, low_channels = channels
        self.eps = eps
        self.high_projection = nn.Sequential(
            nn.Conv2d(high_channels, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True)
        )
        self.low_projection = nn.Sequential(
            nn.Conv2d(low_channels, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True)
        )
        self.upsample = DySample(c2)
        self.scale_weights = nn.Parameter(torch.ones(2))
        self.gate = nn.Conv2d(2 * c2, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.fusion = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def normalized_weights(self) -> torch.Tensor:
        """Return nonnegative BiFPN-style global fusion weights."""
        weights = F.relu(self.scale_weights)
        return weights / (weights.sum() + self.eps)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Align, dynamically upsample, gate, and fuse `[high, low]` features."""
        high, low = inputs
        high = self.upsample(self.high_projection(high))
        low = self.low_projection(low)
        if high.shape[-2:] != low.shape[-2:]:
            raise ValueError(f"DWF spatial mismatch after 2x upsampling: high={high.shape}, low={low.shape}")
        weights = self.normalized_weights()
        gate = torch.sigmoid(self.gate(torch.cat((low, high), dim=1)))
        fused = gate * weights[0] * low + (1.0 - gate) * weights[1] * high
        return self.fusion(fused)


class DWFNearestGate(DWF):
    """DWF diagnostic that retains global weights and local gating but uses nearest upsampling."""

    def __init__(self, channels: list[int], c2: int, eps: float = 1e-4):
        """Replace dynamic sampling with parameter-free nearest-neighbor upsampling."""
        super().__init__(channels, c2, eps)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")


class DWFNearestWeight(DWFNearestGate):
    """DWF diagnostic that uses nearest upsampling and global weights without a local gate."""

    def __init__(self, channels: list[int], c2: int, eps: float = 1e-4):
        """Remove the gate parameters retained by the nearest-gated diagnostic."""
        super().__init__(channels, c2, eps)
        del self.gate

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Project, nearest-upsample, globally weight, and fuse `[high, low]` features."""
        high, low = inputs
        high = self.upsample(self.high_projection(high))
        low = self.low_projection(low)
        if high.shape[-2:] != low.shape[-2:]:
            raise ValueError(f"DWF spatial mismatch after 2x upsampling: high={high.shape}, low={low.shape}")
        weights = self.normalized_weights()
        return self.fusion(weights[0] * low + weights[1] * high)
