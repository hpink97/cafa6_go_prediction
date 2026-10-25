from __future__ import annotations

import unittest

import pandas as pd
import torch

from modeling.hparam_search import DEFAULT_TRIAL, SEARCH_SPACE, best_trial, sample_trials, trial_flags
from modeling.train_multilabel_nn import MultilabelGOModel


class SampleTrialsTests(unittest.TestCase):
    def test_default_first_then_distinct_draws_from_the_space(self) -> None:
        trials = sample_trials(10, seed=3)
        self.assertEqual(trials[0], DEFAULT_TRIAL)
        self.assertEqual(len({tuple(trial.values()) for trial in trials}), 10)
        for trial in trials:
            for name, value in trial.items():
                self.assertIn(value, SEARCH_SPACE[name])
        self.assertEqual(sample_trials(10, seed=3), trials)

    def test_flags_set_promoter_width_for_both_promoter_branches(self) -> None:
        flags = trial_flags({**DEFAULT_TRIAL, "promoter_projection": 64, "trunk_layers": 4})
        projections = flags[flags.index("--projection-dims") + 1 : flags.index("--promoter-fusion-dim")]
        self.assertIn("promoter_nt=64", projections)
        self.assertIn("promoter_agront=64", projections)
        self.assertEqual(flags[flags.index("--trunk-layers") + 1], "4")


class SelectionTests(unittest.TestCase):
    def test_selects_on_seed0_validation_not_test(self) -> None:
        trials = pd.DataFrame(
            {
                "trial": ["trial00", "trial01", "trial01"],
                "seed": [0, 0, 1],
                "validation_nn_cafa_fmax_mean": [0.45, 0.46, 0.40],
                "test_nn_cafa_fmax_mean": [0.50, 0.40, 0.40],
            }
        )
        self.assertEqual(best_trial(trials), "trial01")


class TrunkDepthTests(unittest.TestCase):
    def test_depth_and_projection_widths(self) -> None:
        groups = {"esm2": [0, 1, 2], "promoter_nt": [3, 4]}
        model = MultilabelGOModel(groups, n_terms=5, trunk_dim=16, bottleneck_dim=8, projection_dims={"promoter_nt": 4}, promoter_fusion_dim=2, trunk_layers=4)
        linears = [layer for layer in model.trunk if isinstance(layer, torch.nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linears], [(256 + 2, 16), (16, 16), (16, 16), (16, 8)])
        self.assertEqual(model(torch.zeros(2, 5)).shape, (2, 5))
        with self.assertRaises(ValueError):
            MultilabelGOModel(groups, n_terms=5, trunk_layers=1)


if __name__ == "__main__":
    unittest.main()
