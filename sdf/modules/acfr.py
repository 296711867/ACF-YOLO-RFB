"""Adaptive complementary fusion and partial refinement for the P2 neck."""

from __future__ import annotations

import torch
from torch import nn


class _PartialResidualRefine(nn.Module):
    """Refine a fixed prefix of channels through an exact-zero residual."""

    def __init__(self, channels: int, ratio: float) -> None:
        super().__init__()
        if channels <= 0 or not 0.0 < ratio <= 1.0:
            raise ValueError(f"Invalid partial refinement: channels={channels}, ratio={ratio}")
        self.channels = channels
        self.active_channels = int(round(channels * ratio))
        if not 0 < self.active_channels <= channels:
            raise ValueError(
                f"Active channel count must be within [1, {channels}], got {self.active_channels}"
            )

        self.residual = nn.Sequential(
            nn.Conv2d(
                self.active_channels,
                self.active_channels,
                kernel_size=3,
                padding=1,
                groups=self.active_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.active_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.active_channels, self.active_channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        active = x[:, : self.active_channels]
        bypass = x[:, self.active_channels :]
        active = active + self.residual(active)
        return torch.cat((active, bypass), dim=1) if bypass.shape[1] else active


class ACFRConcat(nn.Module):
    """Replace one P2 concat with identity-initialized complementary grouped fusion.

    Component-ablation switches (both default True reproduce the frozen P19A
    module bit-for-bit): ``use_refine=False`` keeps only the complementary
    gate; ``use_gate=False`` keeps only the partial residual refinement. Both
    variants remain identity-initialized because each component is zero- or
    one-initialized independently.
    """

    def __init__(
        self,
        channels: list[int],
        groups: int = 4,
        refine_ratio: float = 0.5,
        reduction: int = 4,
        use_gate: bool = True,
        use_refine: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"ACFRConcat expects exactly two inputs, got {channels}")
        high_channels, low_channels = channels
        if high_channels != low_channels:
            raise ValueError(
                "ACFRConcat requires equal channels at the frozen P2 insertion: "
                f"high={high_channels}, low={low_channels}"
            )
        if groups <= 0 or high_channels % groups:
            raise ValueError(f"channels={high_channels} must be divisible by groups={groups}")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")
        if not use_gate and not use_refine:
            raise ValueError("ACFRConcat requires at least one of use_gate/use_refine to be enabled")

        self.channels = high_channels
        self.groups = groups
        self.use_gate = use_gate
        self.use_refine = use_refine

        if use_refine:
            self.high_refine = _PartialResidualRefine(high_channels, refine_ratio)
            self.low_refine = _PartialResidualRefine(low_channels, refine_ratio)

        if use_gate:
            hidden_channels = max(groups, (high_channels + low_channels) // reduction)
            self.gate = nn.Sequential(
                nn.Conv2d(high_channels + low_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, groups, 3, padding=1, bias=True),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

        # Checkpoints saved before the ablation switches lack these flags in
        # their pickled state; those modules always had both components.
        if not hasattr(self, "use_gate"):
            self.use_gate = True
        if not hasattr(self, "use_refine"):
            self.use_refine = True

    def __setstate__(self, state: dict) -> None:
        """Restore pre-ablation checkpoints that carry no component flags."""
        self.__dict__.update(state)
        self.use_gate = bool(state.get("use_gate", True))
        self.use_refine = bool(state.get("use_refine", True))

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError(f"ACFRConcat expects two tensors, got {len(inputs)}")
        high, low = inputs
        if high.shape[1] != self.channels or low.shape[1] != self.channels:
            raise ValueError(f"ACFRConcat channel mismatch: high={high.shape}, low={low.shape}")
        if high.shape[-2:] != low.shape[-2:]:
            raise ValueError(f"ACFRConcat spatial mismatch: high={high.shape}, low={low.shape}")

        if self.use_refine:
            high = self.high_refine(high)
            low = self.low_refine(low)
        if self.use_gate:
            weights = torch.sigmoid(self.gate(torch.cat((high, low), dim=1)))
            weights = weights.repeat_interleave(self.channels // self.groups, dim=1)
            high = high * (2.0 * weights)
            low = low * (2.0 * (1.0 - weights))
        return torch.cat((high, low), dim=1)
