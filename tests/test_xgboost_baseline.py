from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.experiments.xgboost_baseline import _direct_label_auc, feature_group, summarise


class FeatureGroupTest(unittest.TestCase):
    def test_prefixes(self) -> None:
        self.assertEqual(feature_group("esm2_640_12"), "ESM-2")
        self.assertEqual(feature_group("kingdom_bacteria"), "kingdom")
        self.assertEqual(feature_group("promoter_nt_5"), "NT promoter")
        self.assertEqual(feature_group("promoter_agront_900"), "AgroNT promoter")
        self.assertEqual(feature_group("gravy"), "handcrafted")


class DirectLabelAucTest(unittest.TestCase):
    def test_scores_against_direct_annotations_not_closed_targets(self) -> None:
        # GO:B is the parent of GO:A. The network's closed target marks P1-P3 positive for GO:B, but only
        # P3 is annotated to GO:B directly, so a score that ranks P3 first is perfect on direct labels.
        proteins = np.array(["P1", "P2", "P3", "P4"])
        scores = np.array([[0.9, 0.5], [0.8, 0.6], [0.1, 0.9], [0.2, 0.1]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "test_predictions.npz", protein_ids=proteins, terms=np.array(["GO:A", "GO:B"]), propagated_probabilities=scores)
            pd.DataFrame({"EntryID": ["P1", "P2", "P3"], "term": ["GO:A", "GO:A", "GO:B"], "aspect": "P"}).to_csv(root / "terms.tsv", sep="\t", index=False)
            aucs = _direct_label_auc(root, ["GO:B", "GO:A"], root / "terms.tsv")
        np.testing.assert_allclose(aucs, [1.0, 1.0])


class SummariseTest(unittest.TestCase):
    def test_gap_and_win_share(self) -> None:
        n = 60
        rng = np.random.default_rng(0)
        xgb = rng.uniform(0.7, 0.9, n)
        table = pd.DataFrame(
            {
                "aspect": np.repeat(["P", "F", "C"], n // 3),
                "response_term": np.arange(n) % 4 == 0,
                "frequency_band": np.tile(["rare", "intermediate", "common"], n // 3),
                "xgb_auc": xgb,
            }
        )
        gap = 0.05 + rng.normal(0, 0.01, n)
        gap -= gap.mean() - 0.05
        for key in ("wider_all", "default_all", "final"):
            table[f"{key}_auc"] = xgb + gap
            table[f"{key}_gap"] = gap
        summary = summarise(table).set_index(["network", "category"])
        row = summary.loc[("wider_all", "all terms")]
        self.assertAlmostEqual(row["gap_mean"], 0.05)
        self.assertLess(row["gap_ci_low"], 0.05)
        self.assertGreater(row["gap_ci_high"], 0.05)
        self.assertEqual(row["network_win_share"], 1.0)
        self.assertEqual(row["n_terms"], n)


if __name__ == "__main__":
    unittest.main()
