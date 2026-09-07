"""Tests for resumable stage-boundary training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.resume_experiment import resume_weight_check_callback, update_output_metadata
from scripts.run_experiment import StageBoundaryReached, stage_stop_callback, validate_stage_checkpoint


class StageTrainingTest(unittest.TestCase):
    """Verify that stage stopping occurs only after a complete resumable checkpoint exists."""

    def test_callback_stops_at_requested_epoch(self) -> None:
        """The callback must allow earlier epochs and signal at the configured boundary."""
        callback = stage_stop_callback(20)
        callback(SimpleNamespace(epoch=18))
        with self.assertRaises(StageBoundaryReached):
            callback(SimpleNamespace(epoch=19))

    def test_checkpoint_requires_exact_epoch_and_training_state(self) -> None:
        """Validation must reject stripped or wrong-epoch checkpoints."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save({"epoch": 19, "optimizer": {}, "scaler": {}, "ema": {}}, path)
            validate_stage_checkpoint(path, 20)

            torch.save({"epoch": 19, "optimizer": None, "scaler": {}, "ema": {}}, path)
            with self.assertRaises(RuntimeError):
                validate_stage_checkpoint(path, 20)

    def test_resume_weight_check_rejects_mismatch(self) -> None:
        """Resume verification must accept exact EMA weights and reject a reset model."""
        expected = torch.nn.Conv2d(3, 4, 1)
        actual = torch.nn.Conv2d(3, 4, 1)
        actual.load_state_dict(expected.state_dict())
        callback = resume_weight_check_callback({"ema": expected})
        callback(SimpleNamespace(model=actual))

        with torch.no_grad():
            actual.weight.add_(1)
        with self.assertRaises(RuntimeError):
            callback(SimpleNamespace(model=actual))

    def test_resume_completion_updates_final_output_hashes(self) -> None:
        """Completion metadata must not retain hashes from an earlier stage checkpoint."""
        metadata = {"outputs": {"best": {"sha256": "old"}, "last": {"sha256": "old"}}}
        run_dir = Path("/tmp/example-run")
        update_output_metadata(metadata, run_dir, "final-best", "final-last")

        self.assertEqual(metadata["outputs"]["best"]["sha256"], "final-best")
        self.assertEqual(metadata["outputs"]["last"]["sha256"], "final-last")
        self.assertEqual(metadata["outputs"]["best"]["path"], "/tmp/example-run/weights/best.pt")
        self.assertEqual(metadata["outputs"]["last"]["path"], "/tmp/example-run/weights/last.pt")


if __name__ == "__main__":
    unittest.main()
