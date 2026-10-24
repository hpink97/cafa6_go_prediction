from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.evaluate_cafa6 import official_roc_auc
from modeling.go_evaluation import roc_auc_summary

OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: root process
namespace: biological_process

[Term]
id: GO:0000002
name: child process
namespace: biological_process
is_a: GO:0000001 ! root process

[Term]
id: GO:0000003
name: root function
namespace: molecular_function
"""


class RocAucSummaryTests(unittest.TestCase):
    def test_macro_micro_and_aspects(self) -> None:
        targets = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.uint8)
        scores = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.4], [0.8, 0.3, 0.1], [0.1, 0.2, 0.3]], dtype=np.float32)
        aspects = np.array(["P", "P", "F"])

        summary, per_term = roc_auc_summary(targets, scores, aspects)

        np.testing.assert_allclose(per_term, [1.0, 1.0, 0.75])
        self.assertAlmostEqual(summary["macro_roc_auc"], (1.0 + 1.0 + 0.75) / 3)
        self.assertAlmostEqual(summary["macro_roc_auc_P"], 1.0)
        self.assertAlmostEqual(summary["macro_roc_auc_F"], 0.75)
        self.assertTrue(np.isnan(summary["macro_roc_auc_C"]))
        self.assertEqual(summary["n_terms_with_roc_auc_C"], 0)
        self.assertTrue(0.5 < summary["micro_roc_auc"] <= 1.0)

    def test_skips_single_class_terms_and_respects_mask(self) -> None:
        targets = np.array([[1, 1], [0, 1], [1, 1]], dtype=np.uint8)
        scores = np.array([[0.2, 0.5], [0.9, 0.5], [0.1, 0.5]], dtype=np.float32)
        aspects = np.array(["P", "P"])

        _, per_term = roc_auc_summary(targets, scores, aspects)
        self.assertEqual(per_term[0], 0.0)
        self.assertTrue(np.isnan(per_term[1]))

        # Masking the misleading negative leaves only positives, so the term is skipped.
        mask = np.array([[True, True], [False, True], [True, True]])
        summary, per_term = roc_auc_summary(targets, scores, aspects, mask=mask)
        self.assertTrue(np.isnan(per_term[0]))
        self.assertEqual(summary["n_terms_with_roc_auc"], 0)


class OfficialRocAucTests(unittest.TestCase):
    def _write_release(self, root: Path) -> None:
        (root / "go-basic.obo").write_text(OBO)
        (root / "May2026_groundtruth_terms_of_interest.txt").write_text("GO:0000001\nGO:0000002\nGO:0000003\n")
        truth = "EntryID\tterm\taspect\nA\tGO:0000002\tP\nB\tGO:0000003\tF\nC\tGO:0000003\tF\n"
        for regime in ("NK", "PK"):
            (root / f"May2026_groundtruth_{regime}.tsv").write_text(truth)
        # A already had the child process annotation before the release.
        (root / "May2026_groundtruth_PK_known.tsv").write_text("EntryID\tterm\taspect\nA\tGO:0000002\tP\n")

    def test_propagates_truth_and_restricts_to_aspect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_release(root)
            predictions = root / "predictions.tsv"
            predictions.write_text("A\tGO:0000002\t0.9\nA\tGO:0000001\t0.9\nB\tGO:0000003\t0.8\nC\tGO:0000003\t0.2\n")

            summary = official_roc_auc(predictions, root, "NK", ["GO:0000001", "GO:0000002", "GO:0000003"])

            # Only A has process truth, so the process terms have no negatives and are skipped;
            # the function term ranks B and C but both are positives, so it is skipped too.
            self.assertEqual(summary["n_terms_with_roc_auc"], 0)

    def test_pk_excludes_known_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_release(root)
            (root / "May2026_groundtruth_NK.tsv").write_text(
                "EntryID\tterm\taspect\nA\tGO:0000002\tP\nB\tGO:0000001\tP\n"
            )
            (root / "May2026_groundtruth_PK.tsv").write_text(
                "EntryID\tterm\taspect\nA\tGO:0000002\tP\nB\tGO:0000001\tP\n"
            )
            predictions = root / "predictions.tsv"
            predictions.write_text("A\tGO:0000002\t0.9\nB\tGO:0000002\t0.1\n")

            nk = official_roc_auc(predictions, root, "NK", ["GO:0000001", "GO:0000002"])
            pk = official_roc_auc(predictions, root, "PK", ["GO:0000001", "GO:0000002"])

            self.assertEqual(nk["n_terms_with_roc_auc"], 1)
            self.assertAlmostEqual(nk["macro_roc_auc"], 1.0)
            # In PK, A's child annotation (and its propagated root) is known, leaving no positive for GO:0000002.
            self.assertEqual(pk["n_terms_with_roc_auc"], 0)


if __name__ == "__main__":
    unittest.main()
