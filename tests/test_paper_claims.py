"""Tests for paper claim traceability."""

from __future__ import annotations

import unittest

from scripts.audit_paper_claims import audit


class PaperClaimAuditTest(unittest.TestCase):
    """Keep manuscript claims tied to frozen JSON and figure evidence."""

    def test_frozen_claims_and_figures_are_present(self) -> None:
        self.assertEqual(audit(), [])


if __name__ == "__main__":
    unittest.main()
