from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.go_evaluation import (
    GeneOntology,
    build_closed_label_matrix,
    cafa_fmax,
    limit_top_predictions,
)


class GOEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        namespace = "biological_process"
        self.ontology = GeneOntology(
            parents={
                "GO:ROOT": set(),
                "GO:PARENT": {"GO:ROOT"},
                "GO:CHILD": {"GO:PARENT"},
                "GO:SIBLING": {"GO:ROOT"},
            },
            namespaces={
                term: namespace
                for term in ("GO:ROOT", "GO:PARENT", "GO:CHILD", "GO:SIBLING")
            },
        )
        self.terms = ["GO:ROOT", "GO:PARENT", "GO:CHILD", "GO:SIBLING"]
        self.ia = {
            "GO:ROOT": 0.0,
            "GO:PARENT": 1.0,
            "GO:CHILD": 2.0,
            "GO:SIBLING": 2.0,
        }

    def test_max_propagation_and_cafa_normalization(self) -> None:
        targets = np.array([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
        probabilities = np.array([[0, 0, 0.9, 0], [0, 0, 0, 0.9]], dtype=np.float32)
        propagated = self.ontology.propagate_scores(probabilities, self.terms)
        np.testing.assert_allclose(propagated[0], [0.9, 0.9, 0.9, 0])
        np.testing.assert_allclose(propagated[1], [0.9, 0, 0, 0.9])

        summary, curve = cafa_fmax(
            targets,
            probabilities,
            self.terms,
            self.ontology,
            self.ia,
            threshold_step=0.5,
        )
        self.assertEqual(summary["cafa_fmax_P"], 0.5)
        self.assertEqual(summary["cafa_precision_P"], 0.5)
        self.assertEqual(summary["cafa_recall_P"], 0.5)
        self.assertEqual(len(curve), 1)

    def test_full_truth_weight_penalizes_terms_outside_output(self) -> None:
        output_terms = ["GO:ROOT", "GO:PARENT", "GO:CHILD"]
        targets = np.array([[0, 0, 1]], dtype=np.uint8)
        probabilities = np.array([[0, 0, 0.9]], dtype=np.float32)
        truth_totals = {"biological_process": np.array([6.0], dtype=np.float32)}
        summary, _ = cafa_fmax(
            targets,
            probabilities,
            output_terms,
            self.ontology,
            self.ia,
            truth_ia_totals=truth_totals,
            threshold_step=0.5,
        )
        self.assertAlmostEqual(summary["cafa_precision_P"], 1.0)
        self.assertAlmostEqual(summary["cafa_recall_P"], 0.5)
        self.assertAlmostEqual(summary["cafa_fmax_P"], 2.0 / 3.0)

    def test_prediction_limit_precedes_ancestor_propagation(self) -> None:
        probabilities = np.array([[0.1, 0.2, 0.9, 0.8]], dtype=np.float32)
        limited = limit_top_predictions(probabilities, max_predictions=2)
        np.testing.assert_allclose(limited, [[0, 0, 0.9, 0.8]])
        propagated = self.ontology.propagate_scores(limited, self.terms)
        np.testing.assert_allclose(propagated, [[0.9, 0.9, 0.9, 0.8]])

    def test_binned_thresholds_match_direct_reference_loop(self) -> None:
        rng = np.random.default_rng(7)
        targets = rng.integers(0, 2, size=(8, len(self.terms)), dtype=np.uint8)
        targets[:, 1] = 1
        probabilities = rng.random(size=targets.shape, dtype=np.float32)
        _, curve = cafa_fmax(
            targets,
            probabilities,
            self.terms,
            self.ontology,
            self.ia,
            threshold_step=0.1,
            max_predictions=None,
        )

        closed_targets = self.ontology.propagate_scores(targets, self.terms) > 0
        closed_probabilities = self.ontology.propagate_scores(probabilities, self.terms)
        weights = np.asarray([self.ia[term] for term in self.terms])
        truth_weight = closed_targets @ weights
        for row in curve.itertuples(index=False):
            predicted = closed_probabilities >= row.threshold
            predicted_weight = predicted @ weights
            true_positive_weight = np.logical_and(predicted, closed_targets) @ weights
            has_predictions = predicted_weight > 0
            precision = np.divide(
                true_positive_weight,
                predicted_weight,
                out=np.zeros_like(true_positive_weight),
                where=has_predictions,
            ).sum() / has_predictions.sum()
            recall = (true_positive_weight / truth_weight).mean()
            f_score = 2 * precision * recall / (precision + recall)
            self.assertAlmostEqual(row.weighted_precision, precision)
            self.assertAlmostEqual(row.weighted_recall, recall)
            self.assertAlmostEqual(row.weighted_f, f_score)

    def test_labels_include_ancestors_of_unmodeled_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations_path = Path(directory) / "annotations.tsv"
            pd.DataFrame(
                {"EntryID": ["protein-1"], "term": ["GO:CHILD"], "aspect": ["P"]}
            ).to_csv(annotations_path, sep="\t", index=False)
            labels = build_closed_label_matrix(
                annotations_path,
                pd.Index(["protein-1"]),
                ["GO:ROOT", "GO:PARENT"],
                self.ontology,
            )
        np.testing.assert_array_equal(labels.to_numpy(), [[1, 1]])

    def test_obo_parser_supports_part_of_alt_ids_and_nonterm_stanzas(self) -> None:
        obo = """format-version: 1.2

[Term]
id: GO:ROOT
name: root
namespace: biological_process

[Term]
id: GO:PARENT
name: parent
namespace: biological_process
is_a: GO:ROOT ! root

[Term]
id: GO:CHILD
name: child
namespace: biological_process
alt_id: GO:ALTCHILD
relationship: part_of GO:PARENT ! parent

[Typedef]
id: part_of
name: part of
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "go.obo"
            path.write_text(obo)
            ontology = GeneOntology.from_obo(path)
        self.assertEqual(ontology.canonical_id("GO:ALTCHILD"), "GO:CHILD")
        self.assertEqual(
            ontology.ancestors("GO:ALTCHILD"),
            frozenset(("GO:CHILD", "GO:PARENT", "GO:ROOT")),
        )


if __name__ == "__main__":
    unittest.main()