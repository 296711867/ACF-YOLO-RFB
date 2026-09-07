"""Unit and integration tests for asymmetric capacity reallocation (P18B ACR)."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/models/yolo26n-p2-acr.yaml"
OFFICIAL_YAML = ROOT / "upstream/ultralytics/cfg/models/26/yolo26-p2.yaml"


def layer_width(model: DetectionModel, index: int, input_tensor: torch.Tensor) -> int:
    """Return the channel width of one layer's output for a fixed input."""
    captured: dict[str, int] = {}

    def hook(_module, _inputs, output) -> None:
        tensor = output if torch.is_tensor(output) else output[0]
        captured["width"] = int(tensor.shape[1])

    handle = model.model[index].register_forward_hook(hook)
    with torch.no_grad():
        model(input_tensor)
    handle.remove()
    return captured["width"]


class AcrModelIntegrationTest(unittest.TestCase):
    """Verify ACR changes exactly two widths and keeps the backbone loadable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.input = torch.randn(1, 3, 640, 640)
        cls.official = DetectionModel(OFFICIAL_YAML, ch=3, nc=10, verbose=False).eval()
        cls.candidate = DetectionModel(MODEL_YAML, ch=3, nc=10, verbose=False).eval()

    def test_only_two_widths_change(self) -> None:
        """L19 widens 32->64, L28 slims 256->192; every probed control width matches."""
        for index in (2, 16, 20, 22, 25):
            self.assertEqual(
                layer_width(self.candidate, index, self.input.clone()),
                layer_width(self.official, index, self.input.clone()),
                f"layer {index} width must match official",
            )
        self.assertEqual(layer_width(self.candidate, 19, self.input.clone()), 64)
        self.assertEqual(layer_width(self.official, 19, self.input.clone()), 32)
        self.assertEqual(layer_width(self.candidate, 28, self.input.clone()), 192)
        self.assertEqual(layer_width(self.official, 28, self.input.clone()), 256)
        self.assertEqual(self.candidate.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

    def test_param_reallocation_budget(self) -> None:
        """Reallocation must be net-negative on parameters within the frozen envelope."""
        official_params = sum(p.numel() for p in self.official.parameters())
        candidate_params = sum(p.numel() for p in self.candidate.parameters())
        delta = candidate_params - official_params
        self.assertLess(delta, -100_000)
        self.assertGreater(delta, -115_000)

    def test_backbone_pretrained_compatibility(self) -> None:
        """All official weights outside layers 19/20/28/29 must load name- and shape-matched."""
        official_state = self.official.state_dict()
        candidate_state = self.candidate.state_dict()
        compatible = {
            key: value
            for key, value in official_state.items()
            if key in candidate_state and value.shape == candidate_state[key].shape
        }
        skipped = set(official_state) - set(compatible)
        changed_prefixes = ("model.19.", "model.20.", "model.28.", "model.29.")
        self.assertTrue(all(key.startswith(changed_prefixes) for key in skipped))
        incompatible = self.candidate.load_state_dict(compatible, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(key.startswith(changed_prefixes) for key in incompatible.missing_keys))

    def test_eval_forward_finite(self) -> None:
        """A full 640 forward must produce finite end-to-end detections."""
        with torch.no_grad():
            outputs = self.candidate(self.input.clone())
        predictions = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        if predictions.ndim == 3:
            predictions = predictions[0]
        self.assertEqual(predictions.shape[1], 6)
        self.assertTrue(torch.isfinite(predictions).all())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA device required")
    def test_cuda_amp_forward(self) -> None:
        """FP16 autocast forward must stay finite on CUDA."""
        model = DetectionModel(MODEL_YAML, ch=3, nc=10, verbose=False).cuda().eval()
        x = torch.randn(2, 3, 640, 640, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            outputs = model(x)
        tensor = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        self.assertTrue(torch.isfinite(tensor.float()).all())


if __name__ == "__main__":
    unittest.main()
