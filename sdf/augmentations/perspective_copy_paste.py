"""Perspective-aware copy-paste augmentation for bounding-box small-object detection."""

from __future__ import annotations

import random

import numpy as np


class PerspectiveAwareSmallObjectCopyPaste:
    """Replicate small objects inside a perspective-consistent vertical band.

    The transform is intentionally limited to detection labels. It copies an object with a narrow context border,
    keeps its apparent size, samples a new horizontal position, and constrains the new bottom edge to the source
    object's vertical band. Donors are sampled with inverse within-image class frequency and placements that occlude
    existing boxes beyond ``max_ioa`` are rejected.
    """

    def __init__(
        self,
        p: float = 0.0,
        max_pastes: int = 3,
        small_area_ratio: float = 0.0025,
        vertical_tolerance: float = 0.08,
        max_ioa: float = 0.20,
        attempts: int = 20,
    ) -> None:
        """Initialize the augmentation with auditable geometric constraints."""
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if max_pastes < 1 or attempts < 1:
            raise ValueError("max_pastes and attempts must be positive")
        if not 0.0 < small_area_ratio <= 1.0:
            raise ValueError(f"small_area_ratio must be in (0, 1], got {small_area_ratio}")
        if not 0.0 <= vertical_tolerance <= 1.0:
            raise ValueError(f"vertical_tolerance must be in [0, 1], got {vertical_tolerance}")
        if not 0.0 <= max_ioa <= 1.0:
            raise ValueError(f"max_ioa must be in [0, 1], got {max_ioa}")
        self.p = p
        self.max_pastes = max_pastes
        self.small_area_ratio = small_area_ratio
        self.vertical_tolerance = vertical_tolerance
        self.max_ioa = max_ioa
        self.attempts = attempts

    def __call__(self, labels: dict) -> dict:
        """Apply PASCP to one detection image and update its boxes and classes in place."""
        if self.p == 0.0 or random.random() >= self.p:
            return labels

        image = labels["img"]
        instances = labels["instances"]
        if instances.normalized or instances.keypoints is not None or len(instances.segments):
            return labels

        boxes = instances.bboxes
        classes = labels["cls"]
        height, width = image.shape[:2]
        if len(boxes) == 0 or boxes.shape[0] != classes.shape[0]:
            return labels

        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        small = np.flatnonzero(
            (box_widths >= 2)
            & (box_heights >= 2)
            & (box_widths * box_heights <= self.small_area_ratio * width * height)
        ).tolist()
        if not small:
            return labels

        class_ids = classes.reshape(-1).astype(np.int64)
        counts = {class_id: int(np.sum(class_ids[small] == class_id)) for class_id in np.unique(class_ids[small])}
        available = small.copy()
        new_boxes: list[np.ndarray] = []
        new_classes: list[np.ndarray] = []
        output = image.copy()
        occupied = boxes.copy()

        for _ in range(min(self.max_pastes, len(available))):
            weights = [1.0 / counts[class_ids[index]] for index in available]
            source_index = random.choices(available, weights=weights, k=1)[0]
            available.remove(source_index)
            placement = self._find_placement(boxes[source_index], occupied, width, height)
            if placement is None:
                continue
            source_patch, destination, target_box = placement
            self._paste(
                output,
                image[source_patch[1] : source_patch[3], source_patch[0] : source_patch[2]],
                destination,
            )
            new_boxes.append(target_box.astype(boxes.dtype, copy=False))
            new_classes.append(classes[source_index : source_index + 1])
            occupied = np.concatenate((occupied, target_box[None]), axis=0)

        if not new_boxes:
            return labels
        labels["img"] = output
        instances.update(np.concatenate((boxes, np.stack(new_boxes)), axis=0))
        labels["cls"] = np.concatenate((classes, *new_classes), axis=0)
        return labels

    def _find_placement(
        self,
        source_box: np.ndarray,
        occupied: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], np.ndarray] | None:
        """Find a valid context patch and destination under perspective and overlap constraints."""
        x1, y1, x2, y2 = source_box
        object_width, object_height = x2 - x1, y2 - y1
        margin = max(2, round(0.15 * max(object_width, object_height)))
        patch_x1 = max(0, int(np.floor(x1)) - margin)
        patch_y1 = max(0, int(np.floor(y1)) - margin)
        patch_x2 = min(image_width, int(np.ceil(x2)) + margin)
        patch_y2 = min(image_height, int(np.ceil(y2)) + margin)
        left_offset, top_offset = x1 - patch_x1, y1 - patch_y1
        patch_width, patch_height = patch_x2 - patch_x1, patch_y2 - patch_y1
        if patch_width < 2 or patch_height < 2:
            return None

        min_bottom = max(object_height, y2 - self.vertical_tolerance * image_height)
        max_bottom = min(image_height, y2 + self.vertical_tolerance * image_height)
        for _ in range(self.attempts):
            target_x1 = random.uniform(0, max(0.0, image_width - object_width))
            target_y2 = random.uniform(min_bottom, max_bottom)
            target_y1 = target_y2 - object_height
            target = np.array(
                [target_x1, target_y1, target_x1 + object_width, target_y2],
                dtype=source_box.dtype,
            )
            dest_x1 = round(target_x1 - left_offset)
            dest_y1 = round(target_y1 - top_offset)
            dest_x2, dest_y2 = dest_x1 + patch_width, dest_y1 + patch_height
            if dest_x1 < 0 or dest_y1 < 0 or dest_x2 > image_width or dest_y2 > image_height:
                continue
            if self._max_ioa(target, occupied) <= self.max_ioa:
                return (
                    (patch_x1, patch_y1, patch_x2, patch_y2),
                    (dest_x1, dest_y1, dest_x2, dest_y2),
                    target,
                )
        return None

    @staticmethod
    def _max_ioa(candidate: np.ndarray, occupied: np.ndarray) -> float:
        """Return maximum candidate-area intersection over existing boxes."""
        intersection_width = np.maximum(
            0.0,
            np.minimum(candidate[2], occupied[:, 2]) - np.maximum(candidate[0], occupied[:, 0]),
        )
        intersection_height = np.maximum(
            0.0,
            np.minimum(candidate[3], occupied[:, 3]) - np.maximum(candidate[1], occupied[:, 1]),
        )
        area = max(float((candidate[2] - candidate[0]) * (candidate[3] - candidate[1])), 1e-9)
        return float(np.max(intersection_width * intersection_height / area, initial=0.0))

    @staticmethod
    def _paste(output: np.ndarray, patch: np.ndarray, destination: tuple[int, int, int, int]) -> None:
        """Paste a context patch using a narrow linear border feather."""
        x1, y1, x2, y2 = destination
        patch_height, patch_width = patch.shape[:2]
        feather = min(3, patch_height // 2, patch_width // 2)
        if feather == 0:
            output[y1:y2, x1:x2] = patch
            return
        vertical_distance = np.minimum(
            np.arange(patch_height)[:, None] + 1,
            np.arange(patch_height, 0, -1)[:, None],
        )
        horizontal_distance = np.minimum(np.arange(patch_width)[None, :] + 1, np.arange(patch_width, 0, -1)[None, :])
        edge_distance = np.minimum(vertical_distance, horizontal_distance)
        alpha = np.clip(edge_distance / feather, 0.0, 1.0)[..., None]
        destination_patch = output[y1:y2, x1:x2]
        blended = patch.astype(np.float32) * alpha + destination_patch.astype(np.float32) * (1.0 - alpha)
        output[y1:y2, x1:x2] = np.rint(blended).astype(output.dtype)
