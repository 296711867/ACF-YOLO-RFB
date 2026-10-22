"""Unit and integration tests for the P20 high-level lightweight unit candidates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn
from ultralytics.nn.modules.conv import DWConv, GhostConv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, get_num_params


ROOT = Path(__file__).resolve().parents[1]
L0_YAML = ROOT / "configs/models/yolo26n-p2-acr-acfr.yaml"
U1_YAML = ROOT / "configs/models/yolo26n-p2-acr-acfr-dw-high.yaml"
U2_YAML = ROOT / "configs/models/yolo26n-p2-acr-acfr-ghost-high.yaml"
PB_YAML = ROOT / "configs/models/yolo26n-p2-acr-acfr-dw-backbone.yaml"
PN_YAML = ROOT / "configs/models/yolo26n-p2-acr-acfr-dw-neck.yaml"
U1_CONFIG = ROOT / "configs/train/bdd100k-light-dw-stage20.yaml"
U2_CONFIG = ROOT / "configs/train/bdd100k-light-ghost-stage20.yaml"
PB_CONFIG = ROOT / "configs/train/bdd100k-light-dw-pb-stage20.yaml"
PN_CONFIG = ROOT / "configs/train/bdd100k-light-dw-pn-stage20.yaml"
SWAPPED_LAYERS = (5, 7, 23, 26)
BACKBONE_LAYERS = (5, 7)
NECK_LAYERS = (23, 26)


def build(path: Path) -> DetectionModel:
    return DetectionModel(str(path), ch=3, nc=10, verbose=False)


def tensors_of(obj):
    """Recursively yield tensors from nested list/tuple/dict structures."""
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from tensors_of(item)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from tensors_of(value)


class LightweightStructureTest(unittest.TestCase):
    """Both candidates swap exactly the four frozen high-level plain Conv layers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.l0 = build(L0_YAML).eval()
        cls.u1 = build(U1_YAML).eval()
        cls.u2 = build(U2_YAML).eval()

    def test_swapped_module_types(self) -> None:
        """Layers 5/7/23/26 are DWConv in U1 and GhostConv in U2; no other top-level type changes."""
        for index in range(len(self.l0.model)):
            base_type = type(self.l0.model[index]).__name__
            u1_type = type(self.u1.model[index]).__name__
            u2_type = type(self.u2.model[index]).__name__
            if index in SWAPPED_LAYERS:
                self.assertEqual(u1_type, "DWConv", f"layer {index} must be DWConv in U1")
                self.assertEqual(u2_type, "GhostConv", f"layer {index} must be GhostConv in U2")
                self.assertEqual(base_type, "Conv", f"layer {index} must be plain Conv in L0")
            else:
                self.assertEqual(u1_type, base_type, f"layer {index} changed in U1")
                self.assertEqual(u2_type, base_type, f"layer {index} changed in U2")

    def test_strides_and_detect_interface(self) -> None:
        """Strides stay [4, 8, 16, 32] and Detect inputs keep their widths."""
        for model in (self.u1, self.u2):
            self.assertEqual(model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])
        for index in (19, 22, 25, 28):
            l0_channels = self._out_channels(self.l0, index)
            self.assertEqual(self._out_channels(self.u1, index), l0_channels)
            self.assertEqual(self._out_channels(self.u2, index), l0_channels)
        self.assertEqual(self._out_channels(self.u1, 19), 64)
        self.assertEqual(self._out_channels(self.u1, 28), 192)

    @staticmethod
    def _out_channels(model: DetectionModel, index: int) -> int:
        captured: dict[str, int] = {}

        def hook(_module, _inputs, output) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            captured["channels"] = int(tensor.shape[1])

        handle = model.model[index].register_forward_hook(hook)
        with torch.no_grad():
            model(torch.zeros(1, 3, 320, 320))
        handle.remove()
        return captured["channels"]

    def test_complexity_below_l0(self) -> None:
        """Params and FLOPs strictly below L0 for both candidates; L0 matches the frozen count."""
        l0_params = get_num_params(self.l0)
        l0_flops = get_flops(self.l0, 640)
        for model in (self.u1, self.u2):
            self.assertLess(get_num_params(model), l0_params)
            self.assertLess(get_flops(model, 640), l0_flops)
        # The frozen 2,498,756 is the nc=80 teacher; nc=10 test builds shed Detect head weights.
        teacher = DetectionModel(str(L0_YAML), ch=3, nc=80, verbose=False)
        self.assertEqual(get_num_params(teacher), 2_498_756)

    def test_pretrained_compatibility_outside_swapped_layers(self) -> None:
        """Every L0 weight outside the four swapped layers loads name- and shape-matched."""
        l0_state = self.l0.state_dict()
        swapped_prefixes = tuple(f"model.{index}." for index in SWAPPED_LAYERS)
        for candidate in (self.u1, self.u2):
            candidate_state = candidate.state_dict()
            compatible = {
                key: value
                for key, value in l0_state.items()
                if key in candidate_state and value.shape == candidate_state[key].shape
            }
            skipped = set(l0_state) - set(compatible)
            self.assertTrue(
                all(key.startswith(swapped_prefixes) for key in skipped),
                f"unexpected skipped keys: {sorted(skipped)[:5]}",
            )
            incompatible = candidate.load_state_dict(compatible, strict=False)
            self.assertFalse(incompatible.unexpected_keys)


class LightweightForwardTest(unittest.TestCase):
    """FP32/AMP forwards, backward, and serialization stay healthy."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.u1 = build(U1_YAML).eval()
        cls.u2 = build(U2_YAML).eval()
        cls.image = torch.randn(1, 3, 640, 640)

    def test_eval_forward_finite(self) -> None:
        """640 eval forwards produce finite end-to-end detections with 6 columns."""
        for model in (self.u1, self.u2):
            with torch.no_grad():
                outputs = model(self.image.clone())
            tensor = next(t for t in tensors_of(outputs) if t.ndim == 3)
            self.assertEqual(tensor.shape[-1], 6)
            self.assertLessEqual(tensor.shape[1], 300)
            self.assertTrue(torch.isfinite(tensor).all())

    def test_backward_finite_and_swapped_layers_receive_grads(self) -> None:
        """Train-mode backward leaves finite gradients, including swapped layers."""
        for model, yaml_path in ((self.u1, U1_YAML), (self.u2, U2_YAML)):
            fresh = build(yaml_path).train()
            out = fresh(torch.randn(2, 3, 320, 320))
            differentiable = [t for t in tensors_of(out) if t.requires_grad]
            self.assertTrue(differentiable)
            differentiable[0].float().pow(2).mean().backward()
            grads = [p.grad for p in fresh.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            for index in SWAPPED_LAYERS:
                named = dict(fresh.named_parameters())
                swapped_params = [n for n in named if n.startswith(f"model.{index}.")]
                self.assertTrue(
                    any(named[n].grad is not None for n in swapped_params),
                    f"layer {index} received no gradient",
                )

    def test_state_roundtrip_bitwise(self) -> None:
        """Saving and reloading reproduces outputs exactly."""
        for model in (self.u1, self.u2):
            with torch.no_grad():
                expected = model(self.image.clone())
            expected_tensor = next(t for t in tensors_of(expected) if t.ndim == 3)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pt"
                torch.save(model, path)
                reloaded = torch.load(path, weights_only=False)
                reloaded.eval()
                with torch.no_grad():
                    actual = reloaded(self.image.clone())
            actual_tensor = next(t for t in tensors_of(actual) if t.ndim == 3)
            self.assertTrue(torch.equal(expected_tensor, actual_tensor))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA device required")
    def test_cuda_amp_forward_finite(self) -> None:
        """FP16 autocast forwards stay finite on CUDA."""
        for yaml_path in (U1_YAML, U2_YAML):
            model = build(yaml_path).cuda().eval()
            x = torch.randn(2, 3, 640, 640, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                outputs = model(x)
            tensors = [t for t in tensors_of(outputs) if t.ndim == 3]
            self.assertTrue(tensors)
            self.assertTrue(all(torch.isfinite(t.float()).all() for t in tensors))


class LightweightConfigTest(unittest.TestCase):
    """U1/U2 training configs differ only in model YAML and run identity."""

    FROZEN_FIELDS = (
        "task", "mode", "pretrained", "data", "imgsz", "epochs", "stage_stop_epoch",
        "batch", "nbs", "device", "workers", "optimizer", "seed", "deterministic",
        "amp", "patience", "exist_ok", "save", "plots", "verbose", "project",
    )

    def setUp(self) -> None:
        self.u1 = yaml.safe_load(U1_CONFIG.read_text())
        self.u2 = yaml.safe_load(U2_CONFIG.read_text())

    def test_frozen_fields_identical(self) -> None:
        for field in self.FROZEN_FIELDS:
            self.assertEqual(self.u1[field], self.u2[field], f"field {field} diverged")

    def test_protocol_values(self) -> None:
        shared = {
            "task": "detect", "mode": "train", "imgsz": 640, "epochs": 200,
            "stage_stop_epoch": 20, "batch": 32, "nbs": 64, "seed": 0,
            "deterministic": True, "amp": True, "exist_ok": False,
            "pretrained": "upstream/yolo26n.pt", "data": "configs/data/bdd100k.yaml",
            "device": "1",
        }
        for config in (self.u1, self.u2):
            for key, value in shared.items():
                self.assertEqual(config[key], value, f"{key} must stay frozen")

    def test_model_and_identity_differ(self) -> None:
        self.assertEqual(self.u1["model"], "configs/models/yolo26n-p2-acr-acfr-dw-high.yaml")
        self.assertEqual(self.u2["model"], "configs/models/yolo26n-p2-acr-acfr-ghost-high.yaml")
        self.assertNotEqual(self.u1["name"], self.u2["name"])


class PositionVariantStructureTest(unittest.TestCase):
    """P-B and P-N swap disjoint position subsets; their union equals U1 exactly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.l0 = build(L0_YAML).eval()
        cls.u1 = build(U1_YAML).eval()
        cls.pb = build(PB_YAML).eval()
        cls.pn = build(PN_YAML).eval()

    def test_disjoint_swaps_matching_u1(self) -> None:
        """P-B swaps only L5/L7, P-N only L23/L26; all swapped as DWConv, rest Conv."""
        for index in range(len(self.l0.model)):
            types = {m: type(getattr(model, "model")[index]).__name__ for m, model in
                     (("l0", self.l0), ("u1", self.u1), ("pb", self.pb), ("pn", self.pn))}
            expected_pb = "DWConv" if index in BACKBONE_LAYERS else types["l0"]
            expected_pn = "DWConv" if index in NECK_LAYERS else types["l0"]
            self.assertEqual(types["pb"], expected_pb, f"layer {index} in P-B")
            self.assertEqual(types["pn"], expected_pn, f"layer {index} in P-N")

    def test_param_additivity(self) -> None:
        """P-B + P-N parameter drops together reproduce the U1 drop exactly."""
        l0 = get_num_params(self.l0)
        drop_pb = l0 - get_num_params(self.pb)
        drop_pn = l0 - get_num_params(self.pn)
        drop_u1 = l0 - get_num_params(self.u1)
        self.assertEqual(drop_pb + drop_pn, drop_u1)

    def test_frozen_position_config_protocol(self) -> None:
        """P-B/P-N configs stay frozen and mirror the U1 recipe except model/name.

        The device field is scheduling, not protocol: P-B/P-N were repinned
        between physical GPUs (GPU0/GPU1) by user decisions recorded in the
        position-ablation protocol deviation log, so only its validity is
        checked, not equality with U1.
        """
        u1 = yaml.safe_load(U1_CONFIG.read_text())
        for path, prefix in ((PB_CONFIG, "pb"), (PN_CONFIG, "pn")):
            config = yaml.safe_load(path.read_text())
            for field, value in u1.items():
                if field in ("model", "name", "project", "device"):
                    continue
                self.assertEqual(config[field], value, f"{prefix} field {field} diverged from U1")
            self.assertIn(config["device"], ("0", "1"))
            self.assertTrue(config["name"].startswith("bdd100k-yolo26n-p2-acr-acfr-dw-"))
            self.assertEqual(config["project"], "runs/detect/phase20-light-position-ablation")


if __name__ == "__main__":
    unittest.main()
