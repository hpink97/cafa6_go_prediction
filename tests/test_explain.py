from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

from modeling.explain.attribution import ExplainData, TermLogit, feature_groups, permutation_importance
from modeling.explain.report import _auc, _known_feature, pretty_feature
from modeling.nn.predict import TrainedModel
from modeling.nn.model import MultilabelGOModel

OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: first process
namespace: biological_process

[Term]
id: GO:0000002
name: second process
namespace: biological_process
"""


class FeatureGroupTests(unittest.TestCase):
    def test_kingdom_is_split_out_of_the_handcrafted_branch(self) -> None:
        names = ["esm2_640_0", "esm2_640_1", "gravy", "aa_fraction_A", "kingdom_bacteria"]
        self.assertEqual(feature_groups(names).tolist(), ["esm2", "esm2", "handcrafted", "handcrafted", "kingdom"])

    def test_pretty_feature_names(self) -> None:
        self.assertEqual(pretty_feature("prosite_bipartite_nuclear_localization_signal_present"), "PROSITE: bipartite NLS")
        self.assertEqual(pretty_feature("aa_fraction_V"), "V content")
        self.assertEqual(pretty_feature("esm2_640_412"), "dim 412")


class SignatureHelperTests(unittest.TestCase):
    def test_auc_is_orientation_free(self) -> None:
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])), 1.0)
        self.assertAlmostEqual(_auc(labels, np.array([0.5, np.nan, 0.5, 0.5])), 0.5)

    def test_kinase_motif_combines_both_prosite_patterns(self) -> None:
        raw = pd.DataFrame({"prosite_protein_kinase_atp_binding_present": [1, 0, 0], "prosite_st_kinase_active_site_present": [0, 1, 0]})
        self.assertEqual(_known_feature(raw, "kinase_motif").tolist(), [1.0, 1.0, 0.0])


class ShapAdditivityTests(unittest.TestCase):
    def test_shap_values_sum_to_logit_minus_background_mean(self) -> None:
        torch.manual_seed(0)
        model = MultilabelGOModel({"esm2": [0, 1, 2, 3], "manual_taxonomy": [4, 5]}, n_terms=3, trunk_dim=16, bottleneck_dim=8).eval()
        background = torch.randn(64, 6)
        x = torch.randn(40, 6)
        wrapper = TermLogit(model, 1)
        values = np.asarray(shap.GradientExplainer(wrapper, background).shap_values(x, nsamples=400, rseed=0)).reshape(40, 6)
        with torch.no_grad():
            expected = (wrapper(x) - wrapper(background).mean()).numpy().ravel()
        self.assertGreater(np.corrcoef(values.sum(axis=1), expected)[0, 1], 0.95)
        self.assertAlmostEqual(float(np.median(values.sum(axis=1) / expected)), 1.0, delta=0.2)


class _UsesOnlyFirstColumns(nn.Module):
    """Term i is predicted from feature i alone, so only the ESM-2 group carries signal."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return 5.0 * features[:, :2]


class PermutationImportanceTests(unittest.TestCase):
    def test_only_the_group_the_model_uses_matters(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 4)).astype(np.float32)
        targets = (x[:, :2] > 0).astype(np.uint8)
        names = ["esm2_640_0", "esm2_640_1", "gravy", "kingdom_bacteria"]
        with tempfile.TemporaryDirectory() as directory:
            obo = Path(directory) / "go.obo"
            obo.write_text(OBO)
            trained = TrainedModel(_UsesOnlyFirstColumns(), ["GO:0000001", "GO:0000002"], names, np.zeros(4), np.zeros(4), np.ones(4))
            data = ExplainData(
                trained=trained,
                groups=feature_groups(names),
                test_ids=np.arange(400).astype(str),
                x_test=x,
                raw_test=pd.DataFrame(x, columns=names),
                targets=targets,
                background=torch.from_numpy(x[:32]),
                terms=pd.DataFrame({"term": trained.terms, "aspect": ["P", "P"]}),
                config={"go_obo_path": str(obo), "max_predictions_per_protein": 10},
            )
            result = permutation_importance(data, repeats=2)

        self.assertTrue((result["baseline_auc"] > 0.99).all())
        self.assertTrue((result["esm2_auc_drop"] > 0.3).all())
        self.assertTrue((result["handcrafted_auc_drop"].abs() < 1e-9).all())
        self.assertTrue((result["kingdom_auc_drop"].abs() < 1e-9).all())


if __name__ == "__main__":
    unittest.main()
