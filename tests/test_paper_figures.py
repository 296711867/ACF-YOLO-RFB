"""Tests for frozen paper-figure inputs and curve parsing."""

from __future__ import annotations

import unittest

from scripts.generate_paper_figures import EXTERNAL_INPUT_KEYS, INPUTS, load_training_curve


class PaperFigureTest(unittest.TestCase):
    """Verify that all frozen figure inputs exist and training curves are complete."""

    def test_all_inputs_exist(self) -> None:
        missing = [str(path) for key, path in INPUTS.items() if key not in EXTERNAL_INPUT_KEYS and not path.is_file()]
        self.assertEqual(missing, [])

    def test_visdrone_curves_cover_exactly_100_epochs(self) -> None:
        missing = [str(INPUTS[key]) for key in EXTERNAL_INPUT_KEYS if not INPUTS[key].is_file()]
        if missing:
            self.skipTest("external VisDrone training CSVs are not available in this worktree")
        for seed in (0, 1, 2):
            epochs, values = load_training_curve(INPUTS[f"seed{seed}_csv"])
            self.assertEqual(epochs.tolist(), list(range(1, 101)))
            self.assertEqual(len(values), 100)


if __name__ == "__main__":
    unittest.main()
