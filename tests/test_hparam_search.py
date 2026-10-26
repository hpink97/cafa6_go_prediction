from __future__ import annotations

import unittest

import json
import sys
import tempfile
import time
from pathlib import Path

import optuna
import pandas as pd
import torch

from modeling.experiments.hparam_search import (
    DEFAULT_TRIAL,
    SEARCH_SPACE,
    TPE_SPACE,
    _run_pruned,
    best_trial,
    complete_trial,
    sample_trials,
    suggest_trial,
    tpe_params,
    trial_flags,
)
from modeling.nn.model import MultilabelGOModel, ResidualBlock
from modeling.nn.train import _learning_rate_factor


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


class TpeSpaceTests(unittest.TestCase):
    def test_conditional_parameters_and_feature_choice(self) -> None:
        params = {name: values[0] for name, values in TPE_SPACE.items()}  # promoter_fusion 0, 2 layers, weighted BCE
        drawn = suggest_trial(optuna.trial.FixedTrial(params))
        self.assertNotIn("promoter_projection", drawn)
        self.assertNotIn("trunk_residual", drawn)
        self.assertIn("max_pos_weight", drawn)
        features, trial = complete_trial(drawn)
        self.assertEqual(features, "esm2_prot_tax")
        self.assertFalse(trial["trunk_residual"])

    def test_seed_run_round_trip(self) -> None:
        # The previous winner on all features, as its trial.json records it (no TPE-only settings).
        winner = {"features": "all", **DEFAULT_TRIAL, "promoter_fusion": 32, "trunk_dim": 1024, "bottleneck_dim": 512, "weight_decay": 1e-3}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "trial.json").write_text(json.dumps(winner))
            params = tpe_params(Path(directory))
        features, trial = complete_trial(suggest_trial(optuna.trial.FixedTrial(params)))
        self.assertEqual(features, "all")
        self.assertEqual(trial_flags(trial), trial_flags({key: value for key, value in winner.items() if key != "features"}))

    def test_cosine_trials_get_a_short_schedule(self) -> None:
        flags = trial_flags({**DEFAULT_TRIAL, "lr_schedule": "cosine", "trunk_residual": True})
        self.assertEqual(flags[flags.index("--epochs") + 1], "40")
        self.assertIn("--trunk-residual", flags)
        self.assertNotIn("--epochs", trial_flags(DEFAULT_TRIAL))

    def test_learning_rate_factor(self) -> None:
        self.assertEqual([_learning_rate_factor(epoch, "constant", 0, 10) for epoch in (0, 9)], [1.0, 1.0])
        self.assertEqual(_learning_rate_factor(0, "cosine", 2, 10), 0.5)
        self.assertAlmostEqual(_learning_rate_factor(2, "cosine", 2, 10), 1.0)
        self.assertAlmostEqual(_learning_rate_factor(6, "cosine", 2, 10), 0.5)


class PruningTests(unittest.TestCase):
    def _trainer(self, progress: Path, scores: list[float]) -> list[str]:
        """A stand-in trainer: writes one progress line per 'epoch', then sleeps as if still training."""
        script = (
            "import json, sys, time\n"
            f"for epoch, score in enumerate({scores}, 1):\n"
            f"    open({str(progress)!r}, 'a').write(json.dumps({{'epoch': epoch, 'selection_score': score}}) + '\\n')\n"
            "time.sleep(60)\n"
        )
        return [sys.executable, "-c", script]

    def test_losing_run_is_stopped_and_marked_pruned(self) -> None:
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.ThresholdPruner(lower=0.5, n_warmup_steps=1))
        trial = study.ask()
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            started = time.monotonic()
            with self.assertRaises(optuna.TrialPruned):
                _run_pruned(self._trainer(progress, [0.4, 0.45, 0.3]), Path(directory) / "train.log", progress, trial)
            self.assertLess(time.monotonic() - started, 30)  # the sleeping trainer was killed, not waited for
        # Stopped at the first report below the threshold; later epochs were never reported.
        self.assertEqual(study.trials[0].intermediate_values, {1: 0.4})

    def test_failed_trainer_raises(self) -> None:
        study = optuna.create_study(direction="maximize")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Exception):
                _run_pruned([sys.executable, "-c", "raise SystemExit(3)"], Path(directory) / "train.log", Path(directory) / "p.jsonl", study.ask())


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

    def test_subset_restricts_to_feature_set(self) -> None:
        trials = pd.DataFrame(
            {"trial": ["trial00", "trial01"], "seed": [0, 0], "features": ["all", "esm2_prot_tax"], "validation_nn_cafa_fmax_mean": [0.47, 0.46]}
        )
        self.assertEqual(best_trial(trials), "trial00")
        self.assertEqual(best_trial(trials, "no-promoters"), "trial01")


class TrunkDepthTests(unittest.TestCase):
    def test_depth_and_projection_widths(self) -> None:
        groups = {"esm2": [0, 1, 2], "promoter_nt": [3, 4]}
        model = MultilabelGOModel(groups, n_terms=5, trunk_dim=16, bottleneck_dim=8, projection_dims={"promoter_nt": 4}, promoter_fusion_dim=2, trunk_layers=4)
        linears = [layer for layer in model.trunk if isinstance(layer, torch.nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linears], [(256 + 2, 16), (16, 16), (16, 16), (16, 8)])
        self.assertEqual(model(torch.zeros(2, 5)).shape, (2, 5))
        with self.assertRaises(ValueError):
            MultilabelGOModel(groups, n_terms=5, trunk_layers=1)

    def test_residual_blocks_replace_middle_layers_only(self) -> None:
        groups = {"esm2": [0, 1, 2]}
        model = MultilabelGOModel(groups, n_terms=5, trunk_dim=16, bottleneck_dim=8, trunk_layers=4, trunk_residual=True)
        self.assertEqual(sum(isinstance(layer, ResidualBlock) for layer in model.trunk), 2)
        self.assertEqual(model(torch.zeros(2, 3)).shape, (2, 5))
        plain = MultilabelGOModel(groups, n_terms=5, trunk_layers=2, trunk_residual=True)
        self.assertFalse(any(isinstance(layer, ResidualBlock) for layer in plain.trunk))


if __name__ == "__main__":
    unittest.main()
