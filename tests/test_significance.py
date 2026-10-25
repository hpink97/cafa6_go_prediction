from __future__ import annotations

import unittest

import numpy as np

from modeling.go_evaluation import GeneOntology, cafa_fmax, protein_fmax_curves, weighted_fmax
from modeling.significance import EvaluationCurves, paired_bootstrap


def _ontology() -> GeneOntology:
    terms = ["GO:A", "GO:B", "GO:C", "GO:D"]
    return GeneOntology(
        parents={"GO:A": set(), "GO:B": {"GO:A"}, "GO:C": {"GO:A"}, "GO:D": {"GO:B"}},
        namespaces={term: "biological_process" for term in terms},
    )


TERMS = ["GO:A", "GO:B", "GO:C", "GO:D"]
IA = {"GO:A": 0.0, "GO:B": 1.0, "GO:C": 1.5, "GO:D": 2.0}


def _evaluation(targets: np.ndarray, scores: np.ndarray) -> EvaluationCurves:
    ontology = _ontology()
    propagated = ontology.propagate_scores(scores, TERMS)
    curves = protein_fmax_curves(targets, propagated, TERMS, ontology, IA, threshold_step=0.05)
    return EvaluationCurves({"block": np.arange(len(targets)).astype(str)}, {"block": curves})


class WeightedFmaxTests(unittest.TestCase):
    def test_unit_weights_reproduce_cafa_fmax(self) -> None:
        rng = np.random.default_rng(0)
        targets = (rng.random((60, 4)) < 0.4).astype(np.uint8)
        scores = rng.random((60, 4)).astype(np.float32)
        ontology = _ontology()
        expected, _ = cafa_fmax(targets, scores, TERMS, ontology, IA, threshold_step=0.05, max_predictions=None)
        curves = protein_fmax_curves(targets, ontology.propagate_scores(scores, TERMS), TERMS, ontology, IA, threshold_step=0.05)
        self.assertAlmostEqual(weighted_fmax(curves["P"], np.ones((1, 60)))[0], expected["cafa_fmax_P"])


class PairedBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.targets = (rng.random((300, 4)) < 0.4).astype(np.uint8)
        noise = rng.random((300, 4)).astype(np.float32)
        self.good = np.clip(0.6 * self.targets + 0.4 * noise, 0, 1).astype(np.float32)
        self.bad = noise

    def test_identical_models_have_no_difference(self) -> None:
        evaluation = _evaluation(self.targets, self.good)
        result = paired_bootstrap({0: (evaluation, evaluation)}, n_bootstrap=200)
        self.assertEqual(result["delta"], 0.0)
        self.assertEqual(result["p_bootstrap"], 1.0)

    def test_detects_a_better_model_and_its_sign(self) -> None:
        good, bad = _evaluation(self.targets, self.good), _evaluation(self.targets, self.bad)
        result = paired_bootstrap({0: (good, bad), 1: (good, bad)}, n_bootstrap=200)
        self.assertGreater(result["delta"], 0.05)
        self.assertGreater(result["ci_low"], 0.0)
        self.assertLessEqual(result["p_bootstrap"], 0.01)
        self.assertEqual(paired_bootstrap({0: (bad, good)}, n_bootstrap=200)["delta"], -result["delta"])


if __name__ == "__main__":
    unittest.main()
