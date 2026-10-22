# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import copy_attr

from .tasks import load_checkpoint


P2P3_DISTILL_MODES = {"p2p3_common", "p2p3_small", "p2p3_consensus"}
# P22 small-object-aware KD modes (docs/protocols/SMALL_OBJECT_KD_PROTOCOL.md):
# small_kd_out = K1 output-only (cls BCE + box L1, teacher-confidence weighted),
# small_kd_feat = K2 P2/P3 same-scale foreground-weighted feature L1,
# small_kd_joint = K3 = K1 + K2. Frozen lambdas from the protocol.
# P23 (docs/protocols/CONFOUND_CONTROL_AND_MGD_KD_PROTOCOL.md):
# small_kd_mgd = M1 = K1 output KD + MGD masked-generative feature distillation
# on P2/P3 (replaces the foreground-L1 path); frozen mask ratio 0.5, alpha 2e-5.
SMALL_KD_MODES = {"small_kd_out", "small_kd_feat", "small_kd_joint", "small_kd_mgd"}
SMALL_KD_LAMBDA_CLS = 1.0
SMALL_KD_LAMBDA_BOX = 0.5
SMALL_KD_LAMBDA_FEAT = 0.1
SMALL_KD_TEMPERATURE = 2.0
SMALL_KD_MGD_MASK_RATIO = 0.5
SMALL_KD_MGD_ALPHA = 2e-5
SMALL_AREA_TAU = 1024.0
BACKGROUND_FLOOR = 0.05
REFERENCE_IMAGE_SIZE = 640


def build_small_object_mask(batch: dict, shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    """Build continuous GT-area weights on one feature grid."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Feature-map shape must be positive, got {shape}")
    images = batch["img"].shape[0]
    device = batch["img"].device
    mask = torch.zeros(images, 1, height, width, dtype=dtype, device=device)
    boxes = batch["bboxes"]
    if not len(boxes):
        return mask

    boxes = boxes.to(device=device, dtype=torch.float32)
    batch_idx = batch["batch_idx"].to(device=device, dtype=torch.long).view(-1)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(batch_idx) != len(boxes):
        raise ValueError("Expected bboxes as Nx4 normalized xywh with one batch_idx per box")
    if (batch_idx < 0).any() or (batch_idx >= images).any():
        raise ValueError(f"batch_idx must be in [0, {images})")
    center_x, center_y, box_width, box_height = boxes.unbind(1)
    area_weight = torch.exp(-(box_width * box_height * REFERENCE_IMAGE_SIZE**2) / SMALL_AREA_TAU)
    grid_x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    grid_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    x1 = center_x - box_width / 2 - 1 / width
    x2 = center_x + box_width / 2 + 1 / width
    y1 = center_y - box_height / 2 - 1 / height
    y2 = center_y + box_height / 2 + 1 / height
    inside = (
        (grid_x[None, None, :] >= x1[:, None, None])
        & (grid_x[None, None, :] <= x2[:, None, None])
        & (grid_y[None, :, None] >= y1[:, None, None])
        & (grid_y[None, :, None] <= y2[:, None, None])
    )
    weighted = inside * area_weight[:, None, None]
    for image_idx in range(images):
        image_weights = weighted[batch_idx == image_idx]
        if len(image_weights):
            mask[image_idx, 0] = image_weights.amax(0).to(dtype)
    return mask


class FeatureHook:
    """Picklable forward hook that stores layer output into a shared dict."""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        """Initialize the hook with the shared feature dict and the layer index to store outputs under."""
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        """Store the layer's forward output into the shared feature dict under its index.

        The output is a tensor for neck layers but a tuple/dict for the Detect head, so it is left untyped.
        """
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """YOLO knowledge distillation model.

    This class wraps a teacher-student pair for knowledge distillation training. Features are extracted from both models
    via forward hooks for distillation.

    Attributes:
        teacher_model (nn.Module): Frozen teacher model providing features.
        student_model (nn.Module): Trainable student model being distilled.
        feats_idx (list): Layer indices for feature extraction.
        projector (nn.ModuleList): MLP projector aligning student features to teacher dimensions.
        dis (float): Distillation loss weight factor.

    Methods:
        get_distill_layers: Auto-detect distillation feature layers from the Detect head.
        forward: Run the student model, or compute the combined loss when given a training batch.
        loss: Compute combined detection and distillation loss.
        loss_sl2: Compute score-weighted L2 distillation loss for a feature pair.
        decouple_outputs: Normalize teacher/student head outputs across train/val formats.
        train: Set training mode while keeping teacher frozen.

    Examples:
        Train a student model with knowledge distillation from a larger teacher (the trainer builds the
        DistillationModel internally when the ``distill_model`` argument is set)
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="coco8.yaml", distill_model="yolo26s.pt")
    """

    def __init__(self, teacher_model: str | Path | nn.Module, student_model: nn.Module):
        """Initialize the distillation model with teacher, student, and feature extraction hooks.

        Args:
            teacher_model (str | Path | nn.Module): Teacher model checkpoint path or module.
            student_model (nn.Module): Student model module to be trained.
        """
        super().__init__()
        ch = student_model.yaml.get("channels", 3)
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]
            if teacher_model.yaml.get("channels", 3) != ch:
                weights = teacher_model
                teacher_model = type(weights)(weights.yaml.copy(), ch=ch, nc=weights.yaml["nc"], verbose=False)
                teacher_model.load(weights)
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model
        self.distill_mode = getattr(student_model.args, "distill_mode", "feature")
        if self.distill_mode not in {"feature", *P2P3_DISTILL_MODES, *SMALL_KD_MODES}:
            raise ValueError(f"Unknown distill_mode={self.distill_mode!r}")
        self.student_feats_idx = self.get_distill_layers(student_model)
        self.teacher_feats_idx = self.get_distill_layers(teacher_model)
        self.feats_idx = self.student_feats_idx  # Backward-compatible public attribute.
        self._validate_distill_layers()

        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self._register_feature_hooks()

        # Get feature dimensions via dummy forward pass (hooks capture outputs)
        imgsz = student_model.args.imgsz
        student_model.eval()
        with torch.no_grad():
            im = torch.zeros(2, ch, imgsz, imgsz, device=device)
            teacher_model(im)
            student_model(im)
        student_model.train()
        teacher_output = [self._teacher_feats[idx] for idx in self.teacher_feats_idx[:-1]]
        student_output = [self._student_feats[idx] for idx in self.student_feats_idx[:-1]]
        teacher_targets = self._teacher_targets(teacher_output)

        copy_attr(self, student_model)
        self.dis = self.student_model.args.dis
        projectors = []
        for student_out, teacher_out in zip(student_output, teacher_targets):
            student_dim = self.decouple_outputs(student_out).shape[1]
            teacher_dim = self.decouple_outputs(teacher_out).shape[1]
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                )
            )
        self.projector = nn.ModuleList(projectors).to(device)

        # P23 M1: MGD generation heads on P2/P3 (train-time only, never exported
        # with the student): 1x1 channel align + 2x(3x3+BN+ReLU) + 3x3, official
        # Masked Generative Distillation (ECCV 2022) structure.
        mgd_heads = []
        if self.distill_mode == "small_kd_mgd":
            for student_out, teacher_out in zip(student_output[:2], teacher_targets[:2]):  # P2/P3 only
                student_dim = self.decouple_outputs(student_out).shape[1]
                teacher_dim = self.decouple_outputs(teacher_out).shape[1]
                mgd_heads.append(
                    nn.Sequential(
                        nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                        nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(teacher_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(teacher_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, stride=1, padding=1),
                    )
                )
        self.mgd_generation = nn.ModuleList(mgd_heads).to(device)

    def __getstate__(self):
        """Return a copy of state for pickling without captured features or hook handles.

        Clears the feature dicts in place (rather than replacing the attributes) because the registered
        FeatureHooks share these exact dict objects; otherwise a deepcopy/pickle of a mid-training model would
        still reach the hook-held tensors (which carry grad_fn and cannot be deep-copied).
        """
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []
        return state

    def __setstate__(self, state):
        """Clear stale features and hooks, and re-register forward hooks after unpickling."""
        self.__dict__.update(state)
        if not hasattr(self, "student_feats_idx"):
            self.student_feats_idx = self.feats_idx
            self.teacher_feats_idx = self.feats_idx
            self.distill_mode = "feature"
        self._teacher_feats = {}
        self._student_feats = {}
        self._register_feature_hooks()

    def _remove_feature_hooks(self) -> None:
        """Remove any previously registered feature-capture hooks."""
        for handle in self._student_hooks:
            handle.remove()
        self._student_hooks.clear()
        if self.teacher_model is not None:
            for handle in self._teacher_hooks:
                handle.remove()
            self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module: nn.Module) -> None:
        """Remove any FeatureHook instances from a module's forward hooks."""
        for handle_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[handle_id]

    def _register_feature_hooks(self) -> None:
        """Register feature-capture hooks, removing stale FeatureHook instances first."""
        self._remove_feature_hooks()
        for idx in self.student_feats_idx:
            self._clear_feature_hooks(self.student_model.model[idx])
            self._student_hooks.append(
                self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))
            )
        if self.teacher_model is not None:
            for idx in self.teacher_feats_idx:
                self._clear_feature_hooks(self.teacher_model.model[idx])
                self._teacher_hooks.append(
                    self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
                )

    def _validate_distill_layers(self) -> None:
        """Require compatible feature pyramids for the selected distillation mode."""
        student_levels = len(self.student_feats_idx) - 1
        teacher_levels = len(self.teacher_feats_idx) - 1
        if self.distill_mode == "feature" and student_levels != teacher_levels:
            raise ValueError(
                "feature distillation requires equal teacher/student pyramid levels; "
                "use a p2p3_* mode for a four-level P2 teacher and three-level student"
            )
        if self.distill_mode in P2P3_DISTILL_MODES and (teacher_levels, student_levels) != (4, 3):
            raise ValueError(
                f"{self.distill_mode} requires four teacher levels and three student levels, "
                f"got teacher={teacher_levels}, student={student_levels}"
            )

    def _teacher_targets(self, teacher_feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return mapped teacher targets for homogeneous or P2-to-P3 distillation."""
        if self.distill_mode == "feature" or self.distill_mode in SMALL_KD_MODES:
            return teacher_feats
        common = teacher_feats[1:]
        if self.distill_mode == "p2p3_common":
            return common
        pooled_p2 = F.avg_pool2d(teacher_feats[0], 2)
        if pooled_p2.shape[0] != teacher_feats[1].shape[0] or pooled_p2.shape[-2:] != teacher_feats[1].shape[-2:]:
            raise ValueError(
                "P2-to-P3 bridge requires pooled P2 and P3 to align, "
                f"got {pooled_p2.shape} and {teacher_feats[1].shape}"
            )
        bridge = torch.cat((pooled_p2, teacher_feats[1]), 1)
        return [bridge, *teacher_feats[2:]]

    @staticmethod
    def get_distill_layers(model: nn.Module) -> list[int]:
        """Auto-detect distillation feature layers from the model's Detect head.

        Returns the Detect head's input layer indices plus the head layer index itself.
        E.g. YOLO26 -> [16, 19, 22, 23], YOLOv8 -> [15, 18, 21, 22].
        """
        for m in model.model:
            if isinstance(m, Detect):
                return [*list(m.f), m.i]
        raise ValueError("No Detect head found in model")

    def _freeze_teacher(self):
        """Keep teacher fixed for distillation."""
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        """Set model train mode while keeping teacher frozen in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        """Forward pass through the student model."""
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:  # for loss calculation during validation while training
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            loss_items["dis_loss"] = loss_distill.detach()
            return torch.cat([regular_loss, loss_distill]), loss_items

        # Clear feature dicts before forward passes
        self._teacher_feats.clear()
        self._student_feats.clear()

        with torch.no_grad():
            teacher_preds = self.teacher_model(batch["img"])  # hooks capture teacher features
        preds = self.student_model(batch["img"])  # hooks capture student features

        regular_loss, loss_items = self.student_model.loss(batch, preds)
        if self.distill_mode in SMALL_KD_MODES:
            # P22 small-object-aware KD: K1 output distillation on the dense
            # one2many branch, K2 foreground-weighted P2/P3 feature L1; the
            # frozen protocol lambdas live in module constants.
            student_out = self.decouple_outputs(preds, branch="one2many")
            teacher_out = self.decouple_outputs(teacher_preds, branch="one2many")
            teacher_conf = teacher_out["scores"].sigmoid().amax(dim=1, keepdim=True)  # N,1,A
            if self.distill_mode in {"small_kd_out", "small_kd_joint", "small_kd_mgd"}:
                loss_distill += SMALL_KD_LAMBDA_BOX * self._kd_dense_l1(
                    student_out["boxes"], teacher_out["boxes"], teacher_conf
                )
                loss_distill += SMALL_KD_LAMBDA_CLS * self._kd_score_bce(
                    student_out["scores"], teacher_out["scores"], teacher_conf
                )
            if self.distill_mode in {"small_kd_feat", "small_kd_joint"}:
                teacher_head_feat = self._teacher_feats[self.teacher_feats_idx[-1]]
                teacher_feats = [self.decouple_outputs(self._teacher_feats[idx]) for idx in self.teacher_feats_idx[:-1]]
                student_feats = [self.decouple_outputs(self._student_feats[idx]) for idx in self.student_feats_idx[:-1]]
                teacher_targets = self._teacher_targets(teacher_feats)
                teacher_scores = self._teacher_level_scores(teacher_head_feat, teacher_feats, consensus=False)
                for level in (0, 1):  # P2 and P3 only - the small-object-sensitive path
                    weight = (
                        BACKGROUND_FLOOR + (1 - BACKGROUND_FLOOR) * build_small_object_mask(
                            batch, teacher_targets[level].shape[-2:], teacher_targets[level].dtype
                        )
                    ) * teacher_scores[level]
                    loss_distill += SMALL_KD_LAMBDA_FEAT * self.loss_weighted_l1(
                        self.projector[level](student_feats[level]), teacher_targets[level], weight
                    )
            if self.distill_mode == "small_kd_mgd":
                teacher_feats = [self.decouple_outputs(self._teacher_feats[idx]) for idx in self.teacher_feats_idx[:-1]]
                student_feats = [self.decouple_outputs(self._student_feats[idx]) for idx in self.student_feats_idx[:-1]]
                teacher_targets = self._teacher_targets(teacher_feats)
                for level in (0, 1):  # P2 and P3 only - the small-object-sensitive path
                    loss_distill += self._mgd_loss(student_feats[level], teacher_targets[level], level)
            loss_distill *= self.dis
            loss_items["dis_loss"] = loss_distill.detach()
            loss_distill = loss_distill * batch["img"].shape[0]
            return torch.cat([regular_loss, loss_distill]), loss_items
        teacher_head_feat = self._teacher_feats[self.teacher_feats_idx[-1]]
        teacher_feats = [self.decouple_outputs(self._teacher_feats[idx]) for idx in self.teacher_feats_idx[:-1]]
        student_feats = [self.decouple_outputs(self._student_feats[idx]) for idx in self.student_feats_idx[:-1]]
        teacher_targets = self._teacher_targets(teacher_feats)
        consensus = self.distill_mode == "p2p3_consensus"
        teacher_scores = self._teacher_level_scores(teacher_head_feat, teacher_feats, consensus)

        if self.distill_mode in {"feature", "p2p3_common"}:
            scores = teacher_scores if self.distill_mode == "feature" else teacher_scores[1:]
            for i, (student_feat, teacher_feat) in enumerate(zip(student_feats, teacher_targets)):
                loss_distill += self.loss_weighted_features(
                    self.projector[i](student_feat), teacher_feat, scores[i], normalize=False
                )
        else:
            bridge_score = torch.maximum(F.avg_pool2d(teacher_scores[0], 2), teacher_scores[1])
            small_mask = build_small_object_mask(batch, teacher_targets[0].shape[-2:], teacher_targets[0].dtype)
            p3_weight = BACKGROUND_FLOOR + (1 - BACKGROUND_FLOOR) * small_mask * bridge_score
            weights = (p3_weight, teacher_scores[2], teacher_scores[3])
            level_gains = (1.0, 0.25, 0.25)
            for i, (student_feat, teacher_feat) in enumerate(zip(student_feats, teacher_targets)):
                loss_distill += level_gains[i] * self.loss_weighted_features(
                    self.projector[i](student_feat), teacher_feat, weights[i], normalize=True
                )
        loss_distill *= self.dis

        loss_items["dis_loss"] = loss_distill.detach()
        loss_distill = loss_distill * batch["img"].shape[0]
        return torch.cat([regular_loss, loss_distill]), loss_items

    @staticmethod
    def _kd_dense_l1(student_boxes: torch.Tensor, teacher_boxes: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Confidence-weighted L1 between dense box maps (N,4,A)."""
        l1 = (student_boxes - teacher_boxes).abs().sum(dim=1, keepdim=True)  # N,1,A
        return (l1 * weight).sum() / (weight.sum() + 1e-9)

    @staticmethod
    def _kd_score_bce(student_scores: torch.Tensor, teacher_scores: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Temperature-softened BCE of dense class maps against sigmoid teacher targets."""
        softened = teacher_scores.sigmoid() ** (1.0 / SMALL_KD_TEMPERATURE)
        # Logits form is autocast-safe; student_scores are raw pre-sigmoid logits.
        bce = F.binary_cross_entropy_with_logits(student_scores, softened.detach(), reduction="none").mean(dim=1, keepdim=True)
        return (bce * weight).sum() / (weight.sum() + 1e-9)

    @staticmethod
    def loss_weighted_l1(student_feat: torch.Tensor, teacher_feat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Spatially weighted L1 feature loss (P22 protocol specifies L1, not L2)."""
        if student_feat.shape != teacher_feat.shape:
            raise ValueError(
                f"Projected student and teacher features must match, got {student_feat.shape} and {teacher_feat.shape}"
            )
        if weight.shape != (student_feat.shape[0], 1, *student_feat.shape[-2:]):
            raise ValueError(f"Weight shape must be N1HW for feature {student_feat.shape}, got {weight.shape}")
        channels = student_feat.shape[1]
        l1 = (student_feat - teacher_feat).abs().sum(dim=1, keepdim=True) / channels
        return (l1 * weight).sum() / (weight.sum() + 1e-9)

    @staticmethod
    def _mgd_masked_sq_sum(generated: torch.Tensor, teacher_feat: torch.Tensor, dropped: torch.Tensor) -> torch.Tensor:
        """Sum over dropped pixels of the channel-mean squared error (MGD sum form).

        The sum-over-pixels (not mean) normalization with alpha=2e-5 reproduces the
        official MGD detection-config loss scale; see P23 protocol section 6.
        """
        sq = (generated - teacher_feat.detach()).pow(2).mean(dim=1, keepdim=True)  # N,1,H,W
        return (sq * dropped).sum()

    def _mgd_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, level: int) -> torch.Tensor:
        """Masked generative distillation (ECCV 2022) on one P2/P3 level.

        P23 frozen: mask ratio 0.5 (random spatial mask shared across channels),
        alpha 2e-5, generation head lives in self.mgd_generation and never enters
        the student deployment graph.
        """
        n, _, h, w = student_feat.shape
        dropped = (torch.rand(n, 1, h, w, device=student_feat.device) < SMALL_KD_MGD_MASK_RATIO).to(student_feat.dtype)
        keep = 1.0 - dropped
        generated = self.mgd_generation[level](student_feat * keep)
        return SMALL_KD_MGD_ALPHA * self._mgd_masked_sq_sum(generated, teacher_feat, dropped)

    def loss_sl2(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, feat_idx: int, teacher_scores: tuple
    ) -> torch.Tensor:
        """Compute score-weighted L2 distillation loss for a feature pair.

        Args:
            student_feat (torch.Tensor): Student feature tensor of shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature tensor of shape (N, C, H, W).
            feat_idx (int): Index of the feature level for selecting teacher scores.
            teacher_scores (tuple): Tuple of score tensors for each feature level.

        Returns:
            (torch.Tensor): The computed score-weighted L2 loss.
        """
        return self.loss_weighted_features(student_feat, teacher_feat, teacher_scores[feat_idx], normalize=False)

    @staticmethod
    def loss_weighted_features(
        student_feat: torch.Tensor, teacher_feat: torch.Tensor, weight: torch.Tensor, normalize: bool
    ) -> torch.Tensor:
        """Compute a spatially weighted feature-matching loss."""
        if student_feat.shape != teacher_feat.shape:
            raise ValueError(
                f"Projected student and teacher features must match, got {student_feat.shape} and {teacher_feat.shape}"
            )
        if weight.shape != (student_feat.shape[0], 1, *student_feat.shape[-2:]):
            raise ValueError(f"Weight shape must be N1HW for feature {student_feat.shape}, got {weight.shape}")
        if normalize:
            student_feat = F.normalize(student_feat, dim=1)
            teacher_feat = F.normalize(teacher_feat, dim=1)
        channels = student_feat.shape[1]
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        return (mse * weight).sum() / (weight.sum() * channels + 1e-9)

    def _teacher_level_scores(
        self, teacher_head_feat, teacher_feats: list[torch.Tensor], consensus: bool
    ) -> tuple[torch.Tensor, ...]:
        """Return per-level confidence or dual-branch consensus maps."""
        one2many = self.decouple_outputs(teacher_head_feat, branch="one2many")["scores"]
        one2one = self.decouple_outputs(teacher_head_feat, branch="one2one")["scores"]
        if consensus:
            scores = torch.sqrt(one2many.sigmoid() * one2one.sigmoid())
        else:
            scores = ((one2many + one2one) / 2).sigmoid()
        parts = torch.split(scores, [feat.shape[-2] * feat.shape[-1] for feat in teacher_feats], dim=-1)
        return tuple(
            part.max(dim=1, keepdim=True).values.view(part.shape[0], 1, *feat.shape[-2:])
            for part, feat in zip(parts, teacher_feats)
        )

    @property
    def criterion(self):
        """Get the criterion from the student model."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Set value for student criterion."""
        self.student_model.criterion = value

    def init_criterion(self):
        """Initialize the loss criterion via the student model."""
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        """Expose student end-to-end mode for validator/predictor control."""
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Forward end-to-end mode update to the student model."""
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        """Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model."""
        self.student_model.set_head_attr(**kwargs)

    def decouple_outputs(self, preds, branch: str = "one2one"):
        """Decouple outputs for teacher/student models.

        This method handles different output formats from YOLO models, including
        tuple outputs (train/val mode), dict outputs with branches (one2one/one2many),
        and direct tensor outputs.

        Args:
            preds (torch.Tensor | tuple | dict): Model predictions in various formats.
            branch (str): Which branch to extract from dict outputs ("one2one" or "one2many").

        Returns:
            (torch.Tensor | dict): The decoupled predictions.
        """
        if isinstance(preds, tuple):  # decouple for val mode
            preds = preds[1]
        if isinstance(preds, dict):
            if branch in preds:
                preds = preds[branch]
        return preds
