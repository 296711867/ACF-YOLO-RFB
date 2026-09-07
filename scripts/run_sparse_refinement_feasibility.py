"""Run frozen full-frame, SAHI, or oracle sparse-refinement inference on BDD."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.slicing import get_slice_bboxes

from scripts.evaluate_object_sizes import evaluate, load_ground_truth, load_predictions
from sdf.refinement import (
    greedy_oracle_tiles,
    merge_scale_aware_predictions,
    object_predictions_to_arrays,
    shift_and_filter_local_predictions,
)


BDD_NAMES = {
    0: "person",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motor",
    7: "bike",
    8: "traffic light",
    9: "traffic sign",
}


def load_manifest(manifest: Path, images_dir: Path) -> list[Path]:
    """Resolve a filename-only manifest and reject missing or duplicate entries."""
    names = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError("manifest contains duplicate image names")
    paths = [(images_dir / name).resolve() for name in names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest image does not exist: {missing[0]}")
    return paths


def predict_oracle(
    image: np.ndarray,
    image_path: Path,
    model,
    slice_size: int,
    overlap: float,
    top_k: int,
    match_threshold: float,
    local_max_area: float,
) -> tuple[list, dict]:
    """Run a full pass plus GT-selected Top-K tiles as an explicit ceiling test."""
    height, width = image.shape[:2]
    gt_boxes, _, gt_areas = load_ground_truth(image_path, width, height)
    small_boxes = gt_boxes[gt_areas < 32**2]
    slice_boxes = get_slice_bboxes(
        height,
        width,
        slice_height=slice_size,
        slice_width=slice_size,
        auto_slice_resolution=False,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )
    selected, coverage = greedy_oracle_tiles(slice_boxes, small_boxes, top_k)
    full_predictions = list(get_prediction(image, model, full_shape=[height, width], verbose=0).object_prediction_list)
    local_predictions = []
    for index in selected:
        x1, y1, x2, y2 = slice_boxes[index]
        crop = image[y1:y2, x1:x2]
        result = get_prediction(crop, model, shift_amount=[x1, y1], full_shape=[height, width], verbose=0)
        local_predictions.extend(shift_and_filter_local_predictions(result.object_prediction_list, local_max_area))
    predictions = merge_scale_aware_predictions(
        full_predictions,
        local_predictions,
        max_local_area=local_max_area,
        match_threshold=match_threshold,
    )
    return predictions, {
        "candidate_tiles": len(slice_boxes),
        "selected_tiles": [slice_boxes[index] for index in selected],
        "small_gt": len(small_boxes),
        "small_center_coverage": coverage,
        "detector_passes": 1 + len(selected),
    }


def run(args: argparse.Namespace) -> dict:
    """Execute inference, write resumable predictions, and evaluate size metrics."""
    image_paths = load_manifest(args.manifest, args.images_dir)
    image_size = args.full_imgsz if args.mode == "full" else args.imgsz
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(args.model.resolve()),
        confidence_threshold=args.conf,
        device=args.device,
        image_size=image_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_suffix(".predictions.jsonl")
    oracle_details, elapsed = [], 0.0
    with prediction_path.open("w", encoding="utf-8") as output_file:
        for image_index, image_path in enumerate(image_paths, 1):
            with Image.open(image_path) as opened:
                image = np.asarray(opened.convert("RGB"))
            started = time.perf_counter()
            if args.mode == "full":
                predictions = get_prediction(image, model, verbose=0).object_prediction_list
            elif args.mode == "sahi":
                predictions = get_sliced_prediction(
                    image,
                    model,
                    slice_height=args.slice_size,
                    slice_width=args.slice_size,
                    overlap_height_ratio=args.overlap,
                    overlap_width_ratio=args.overlap,
                    perform_standard_pred=True,
                    postprocess_type="GREEDYNMM",
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=args.match_threshold,
                    postprocess_class_agnostic=False,
                    verbose=0,
                ).object_prediction_list
            else:
                predictions, details = predict_oracle(
                    image,
                    image_path,
                    model,
                    args.slice_size,
                    args.overlap,
                    args.top_k,
                    args.match_threshold,
                    args.local_max_area,
                )
                oracle_details.append(details)
            elapsed += time.perf_counter() - started
            boxes, classes, scores = object_predictions_to_arrays(predictions)
            height, width = image.shape[:2]
            output_file.write(
                json.dumps(
                    {
                        "image": str(image_path),
                        "width": width,
                        "height": height,
                        "boxes_xyxy": boxes.tolist(),
                        "classes": classes.tolist(),
                        "scores": scores.tolist(),
                    }
                )
                + "\n"
            )
            if image_index % 25 == 0 or image_index == len(image_paths):
                print(f"{args.mode}: {image_index}/{len(image_paths)} images", flush=True)

    records = load_predictions(prediction_path)
    protocol = {
        "mode": args.mode,
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "images_dir": str(args.images_dir.resolve()),
        "images": len(image_paths),
        "device": args.device,
        "confidence": args.conf,
        "max_det": args.max_det,
        "image_size": image_size,
        "slice_size": args.slice_size if args.mode != "full" else None,
        "overlap": args.overlap if args.mode != "full" else None,
        "top_k": args.top_k if args.mode == "oracle" else None,
        "local_max_area": args.local_max_area if args.mode == "oracle" else None,
        "fusion": "class-aware GREEDYNMM/IOS",
        "match_threshold": args.match_threshold,
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_image": round(elapsed / len(image_paths), 6),
        "raw_predictions": str(prediction_path.resolve()),
    }
    if oracle_details:
        protocol["oracle"] = {
            "mean_candidate_tiles": float(np.mean([item["candidate_tiles"] for item in oracle_details])),
            "mean_detector_passes": float(np.mean([item["detector_passes"] for item in oracle_details])),
            "mean_small_center_coverage": float(np.mean([item["small_center_coverage"] for item in oracle_details])),
        }
    report = {
        "status": "development feasibility evidence; not final validation",
        "protocol": protocol,
        "metrics": evaluate(records, BDD_NAMES, args.max_det, progress=True),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": protocol, "metrics": report["metrics"]}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "sahi", "oracle"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--full-imgsz", type=int, default=640)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--local-max-area", type=float, default=float(32**2))
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
