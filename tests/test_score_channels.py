from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from modeling.nn.score_channels import (
    blend_scores_by_aspect,
    exact_cosine_neighbors,
    fit_taxon_prior,
    knn_annotation_scores,
    make_taxon_stratified_validation_split,
)


class ScoreChannelTests(unittest.TestCase):
    def test_taxon_split_is_disjoint_and_preserves_common_taxa(self) -> None:
        ids = np.asarray([f"p{index}" for index in range(12)])
        taxonomy = pd.Series(
            ["a"] * 6 + ["b"] * 4 + ["singleton-1", "singleton-2"],
            index=ids,
        )
        train_ids, validation_ids = make_taxon_stratified_validation_split(
            ids,
            taxonomy,
            validation_size=0.25,
            random_state=3,
        )
        self.assertFalse(set(train_ids) & set(validation_ids))
        self.assertEqual(set(train_ids) | set(validation_ids), set(ids))
        validation_taxa = set(taxonomy.loc[validation_ids])
        self.assertIn("a", validation_taxa)
        self.assertIn("b", validation_taxa)

    def test_taxon_prior_uses_only_supplied_donors_and_falls_back_global(self) -> None:
        ids = np.asarray(["a1", "a2", "b1", "b2"])
        taxonomy = pd.Series(["a", "a", "b", "b", "unseen"], index=[*ids, "query"])
        labels = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.uint8)
        prior = fit_taxon_prior(ids, labels, taxonomy, np.asarray(["P", "P"]), strength=1.0)
        predictions = prior.predict(np.asarray(["a1", "b1", "query"]), taxonomy)
        self.assertGreater(predictions[0, 0], predictions[0, 1])
        self.assertGreater(predictions[1, 1], predictions[1, 0])
        np.testing.assert_allclose(predictions.max(axis=1), 1.0)
        np.testing.assert_allclose(predictions[2], prior.global_scores)

    def test_exact_cosine_neighbors_and_weighted_votes(self) -> None:
        donors = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        queries = np.asarray([[0.9, 0.1], [0, 1]], dtype=np.float32)
        similarities, indices = exact_cosine_neighbors(donors, queries, k=2, device="cpu")
        self.assertEqual(indices[0, 0], 0)
        self.assertEqual(indices[1, 0], 1)
        self.assertGreaterEqual(similarities[0, 0], similarities[0, 1])

        labels = np.asarray([[1, 0], [0, 1], [0, 0]], dtype=np.uint8)
        scores = knn_annotation_scores(
            indices,
            similarities,
            labels,
            np.asarray(["P", "P"]),
            normalize_by_aspect=False,
        )
        self.assertGreater(scores[0, 0], scores[0, 1])
        self.assertGreater(scores[1, 1], scores[1, 0])

        normalized_scores = knn_annotation_scores(
            indices,
            similarities,
            labels,
            np.asarray(["P", "P"]),
        )
        np.testing.assert_allclose(normalized_scores.max(axis=1), 1.0)

    def test_blend_scores_by_aspect(self) -> None:
        base = np.zeros((1, 3), dtype=np.float32)
        channel = np.ones((1, 3), dtype=np.float32)
        blended = blend_scores_by_aspect(
            base,
            channel,
            np.asarray(["P", "F", "C"]),
            {"P": 0.25, "F": 0.5, "C": 0.75},
        )
        np.testing.assert_allclose(blended, [[0.25, 0.5, 0.75]])


if __name__ == "__main__":
    unittest.main()