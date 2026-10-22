"""SimAM: a parameter-free attention module (Yang et al., ICML 2021).

Computes 3D attention weights from the neuro-scientific energy function per
neuron and rescales activations; no learnable parameters, inference-cost-only
elementwise and reduction operators (TRT/ONNX friendly). Used by the P28
screen as a background-suppression gate on the P2 detail branch.
"""

from __future__ import annotations

import torch
from torch import nn


class SimAM(nn.Module):
    """Parameter-free attention: x * sigmoid(energy-based weight)."""

    def __init__(self, e_lambda: float = 1e-4):
        """Store the numerical-stability constant; no parameters are created."""
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input activations by sigmoid of normalized neuronal energy."""
        spatial = x.shape[2] * x.shape[3]
        n = spatial - 1
        x_minus_mu_square = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        denominator = 4 * (x_minus_mu_square.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)
        energy = x_minus_mu_square / denominator + 0.5
        return x * torch.sigmoid(energy)

    def extra_repr(self) -> str:
        """Return the module's constant for repr."""
        return f"e_lambda={self.e_lambda}"
