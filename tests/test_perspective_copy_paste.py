"""Tests for perspective-aware small-object copy-paste augmentation."""

from __future__ import annotations

import random
import unittest
from copy import deepcopy

import numpy as np

from sdf.augmentations import PerspectiveAwareSmallObjectCopyPaste
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import v8_transforms
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.instance import Instances


class PerspectiveAwareSmallObjectCopyPasteTest(unittest.TestCase):
    """Verify deterministic pixels, labels, geometry, configuration, and pipeline registration."""

    @staticmethod
    def labels() -> dict:
        """Create one pasteable small object and one large exclusion box."""
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[30:38, 10:16] = (30, 120, 240)
        boxes = np.array([[10, 30, 16, 38], [30, 8, 58, 58]], dtype=np.float32)
        return {
            "img": image,
            "cls": np.array([[2], [7]], dtype=np.float32),
            "instances": Instances(
                boxes,
                segments=np.empty((0, 0, 2), dtype=np.float32),
                bbox_format="xyxy",
                normalized=False,
            ),
        }

    @staticmethod
    def transform(**kwargs) -> PerspectiveAwareSmallObjectCopyPaste:
        """Build an always-on transform with a test-sized small-object threshold."""
        return PerspectiveAwareSmallObjectCopyPaste(
            p=1.0,
            max_pastes=1,
            small_area_ratio=0.02,
            vertical_tolerance=0.05,
            max_ioa=0.0,
            attempts=100,
            **kwargs,
        )

    def test_deterministic_paste_preserves_scale_and_vertical_band(self) -> None:
        """The same seed must produce identical pixels and one valid same-scale label."""
        first = self.labels()
        second = deepcopy(first)
        random.seed(17)
        first = self.transform()(first)
        random.seed(17)
        second = self.transform()(second)

        np.testing.assert_array_equal(first["img"], second["img"])
        np.testing.assert_array_equal(first["instances"].bboxes, second["instances"].bboxes)
        np.testing.assert_array_equal(first["cls"], second["cls"])
        self.assertEqual(len(first["instances"]), 3)
        self.assertEqual(first["cls"][-1, 0], 2)

        source, large, pasted = first["instances"].bboxes
        np.testing.assert_allclose(pasted[2:] - pasted[:2], source[2:] - source[:2])
        self.assertLessEqual(abs(float(pasted[3] - source[3])), 0.05 * 64)
        intersection_width = max(0.0, min(pasted[2], large[2]) - max(pasted[0], large[0]))
        intersection_height = max(0.0, min(pasted[3], large[3]) - max(pasted[1], large[1]))
        self.assertEqual(intersection_width * intersection_height, 0.0)
        self.assertGreater(np.count_nonzero(first["img"]), np.count_nonzero(self.labels()["img"]))

    def test_disabled_and_non_detection_inputs_are_unchanged(self) -> None:
        """Default-off and unsupported segmentation inputs must remain untouched."""
        labels = self.labels()
        original_image = labels["img"]
        self.assertIs(PerspectiveAwareSmallObjectCopyPaste()(labels), labels)
        self.assertIs(labels["img"], original_image)

        segmented = self.labels()
        segmented["instances"].segments = np.zeros((2, 4, 2), dtype=np.float32)
        original_boxes = segmented["instances"].bboxes.copy()
        self.transform()(segmented)
        np.testing.assert_array_equal(segmented["instances"].bboxes, original_boxes)

    def test_configuration_validation_and_pipeline_registration(self) -> None:
        """Custom arguments must be typed, bounded, and registered after the affine transform."""
        cfg = get_cfg(overrides={"pascp": 0.5, "pascp_max_pastes": 2})
        self.assertEqual(cfg.pascp, 0.5)
        self.assertEqual(cfg.pascp_max_pastes, 2)
        with self.assertRaises(TypeError):
            get_cfg(overrides={"pascp_max_pastes": 1.5})
        with self.assertRaises(ValueError):
            get_cfg(overrides={"pascp": 1.1})

        class Dataset:
            data = {}
            use_keypoints = False
            cache = "ram"

        hyp = deepcopy(DEFAULT_CFG)
        hyp.mosaic = hyp.copy_paste = hyp.mixup = hyp.cutmix = 0.0
        hyp.pascp = 0.5
        transforms = v8_transforms(Dataset(), imgsz=64, hyp=hyp)
        pre_transform = transforms[0]
        self.assertIsInstance(pre_transform[-1], PerspectiveAwareSmallObjectCopyPaste)
        self.assertEqual(pre_transform[-1].p, 0.5)


if __name__ == "__main__":
    unittest.main()
