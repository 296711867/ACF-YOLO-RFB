"""Sparse high-resolution refinement helpers."""

from .sparse import (
    greedy_oracle_tiles,
    merge_object_predictions,
    merge_scale_aware_predictions,
    object_predictions_to_arrays,
    shift_and_filter_local_predictions,
)
from .router import PerspectiveBandRouter, perspective_band_descriptor
from .pre_router import (
    ThumbnailUtilityRouter,
    thumbnail_utility_descriptor,
    utility_routing_loss,
    weighted_pairwise_ranking_loss,
)
from .scout_router import ScoutDetectionValueRouter, ensemble_lower_confidence_score, scout_value_loss

__all__ = [
    "greedy_oracle_tiles",
    "merge_object_predictions",
    "merge_scale_aware_predictions",
    "object_predictions_to_arrays",
    "shift_and_filter_local_predictions",
    "PerspectiveBandRouter",
    "perspective_band_descriptor",
    "ThumbnailUtilityRouter",
    "thumbnail_utility_descriptor",
    "utility_routing_loss",
    "weighted_pairwise_ranking_loss",
    "ScoutDetectionValueRouter",
    "ensemble_lower_confidence_score",
    "scout_value_loss",
]
