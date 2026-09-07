"""Integration tests for the widen-only capacity control (P18C R2)."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/models/yolo26n-p2-widen.yaml"
OFFICIAL_YAML = ROOT / "upstream/ultralytics/cfg/models/26/yolo26-p2.yaml"
ACR_YAML = ROOT / "configs/models/yolo26n-p2-acr.yaml"


class WidenControlIntegrationTest(unittest.TestCase):
    """Verify widen-only changes exactly L19's width and sits between official and ACR on cost."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.input = torch.randn(1, 3, 640, 640)
        cls.official = DetectionModel(OFFICIAL_YAML, ch=3, nc=10, verbose=False).eval()
        cls.widen = DetectionModel(MODEL_YAML, ch=3, nc=10, verbose=False).eval()
        cls.acr = DetectionModel(ACR_YAML, ch=3, nc=10, verbose=False).eval()

    @staticmethod
    def width(model: DetectionModel, index: int, tensor: torch.Tensor) -> int:
        captured: dict[str, int] = {}

        def hook(_m, _i, output) -> None:
            t = output if torch.is_tensor(output) else output[0]
            captured["w"] = int(t.shape[1])

        h = model.model[index].register_forward_hook(hook)
        with torch.no_grad():
            model(tensor)
        h.remove()
        return captured["w"]

    def test_l19_widened_l28_official(self) -> None:
        """Widen-only must match ACR at L19 (64ch) and official at L28 (256ch)."""
        for model in (self.widen, self.acr):
            self.assertEqual(self.width(model, 19, self.input.clone()), 64)
        for model in (self.widen, self.official):
            self.assertEqual(self.width(model, 28, self.input.clone()), 256)
        self.assertEqual(self.widen.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

    def test_param_ordering(self) -> None:
        """ACR (net-negative) < official < widen-only on parameters (widen-only pays full width cost)."""
        counts = {
            name: sum(p.numel() for p in model.parameters())
            for name, model in (("official", self.official), ("acr", self.acr), ("widen", self.widen))
        }
        self.assertLess(counts["acr"], counts["official"])
        self.assertLess(counts["official"], counts["widen"])

    def test_pretrained_boundary(self) -> None:
        """Official weights load everywhere except layers 19/20/29 (L28 is compatible here)."""
        official_state = self.official.state_dict()
        candidate_state = self.widen.state_dict()
        compatible = {
            k: v for k, v in official_state.items() if k in candidate_state and v.shape == candidate_state[k].shape
        }
        skipped = set(official_state) - set(compatible)
        changed = ("model.19.", "model.20.", "model.29.")
        self.assertTrue(all(k.startswith(changed) for k in skipped))
        self.assertTrue(any(k.startswith("model.28.") for k in compatible))

    def test_eval_forward_finite(self) -> None:
        with torch.no_grad():
            outputs = self.widen(self.input.clone())
        tensor = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[0]
        if tensor.ndim == 3:
            tensor = tensor[0]
        self.assertTrue(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
