from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.experiments.ablation import CONFIGS, select_runs, train_command
from modeling.experiments.ablation_report import write_report
from modeling.evaluation.official import METRICS, REGIMES


def _write_fake_run(directory: Path, quality: float, cafaeval: bool = True) -> None:
    """The files a finished run leaves behind, with every metric shifted by `quality`."""
    model = directory / "model"
    model.mkdir(parents=True)
    summary = {
        "test_nn_macro_roc_auc": 0.80 + quality,
        "test_nn_micro_roc_auc": 0.90 + quality,
        **{f"test_nn_cafa_fmax_{aspect}": 0.40 + quality for aspect in ("P", "F", "C")},
        "test_nn_cafa_fmax_mean": 0.40 + quality,
    }
    (model / "summary.json").write_text(json.dumps(summary))
    (model / "run_config.json").write_text(json.dumps({"n_parameters": 1000, "best_epoch": 5}))
    pd.DataFrame({"epoch": [1, 2], "validation_loss": [0.3, 0.2], "cafa_fmax_mean": [0.3, 0.4]}).to_csv(model / "history.tsv", sep="\t", index=False)
    pd.DataFrame(
        [{"aspect": aspect, "weighted_precision": p, "weighted_recall": 1 - p} for aspect in ("P", "F", "C") for p in (0.2, 0.6)]
    ).to_csv(model / "test_nn_cafa_curve.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"term": ["GO:1", "GO:2"], "n_positive": [5, 7], "roc_auc": [0.7, 0.8], "average_precision": [0.2, 0.3]}
    ).to_csv(model / "test_nn_metrics.tsv", sep="\t", index=False)
    for regime in REGIMES if cafaeval else ():
        regime_dir = directory / "cafaeval" / regime
        regime_dir.mkdir(parents=True)
        for file_name, column in METRICS.values():
            pd.DataFrame(
                {"filename": "run.tsv", "ns": ["biological_process", "molecular_function", "cellular_component"], column: 0.3 + quality}
            ).to_csv(regime_dir / file_name, sep="\t", index=False)
    auc = {regime: {"macro_roc_auc": 0.8 + quality, "micro_roc_auc": 0.9 + quality} for regime in REGIMES}
    (directory / "official_auc.json").write_text(json.dumps(auc))


class TrainCommandTests(unittest.TestCase):
    def test_configs_toggle_the_intended_feature_groups(self) -> None:
        command = train_command(CONFIGS["prot_tax"], seed=2, directory=Path("run"), smoke=False)
        self.assertIn("--no-esm2-features", command)
        self.assertIn("--no-promoter-features", command)
        self.assertIn("--no-knn", command)
        self.assertEqual(command[command.index("--random-state") + 1], "2")

        self.assertFalse(any(flag.startswith("--no-") and flag.endswith("-features") for flag in train_command(CONFIGS["all"], 0, Path("run"), False)))

    def test_smoke_caps_override_epochs(self) -> None:
        command = train_command(CONFIGS["esm2"], seed=0, directory=Path("run"), smoke=True)
        last_epochs = len(command) - 1 - command[::-1].index("--epochs")
        self.assertEqual(command[last_epochs + 1], "2")


class ReportTests(unittest.TestCase):
    def test_report_writes_summary_tables_figures_and_config_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep = root / "sweep"
            for seed, noise in enumerate((0.0, 0.002)):
                _write_fake_run(sweep / "esm2" / f"seed{seed}", quality=noise)
                _write_fake_run(sweep / "esm2_prot_tax" / f"seed{seed}", quality=0.01 + noise)
            (sweep / "all" / "seed0").mkdir(parents=True)  # unfinished run: skipped

            write_report(sweep, root / "results", root / "release")

            results = root / "results"
            runs = pd.read_csv(results / "summary" / "runs.csv")
            self.assertEqual(sorted(runs["config"].unique()), ["esm2", "esm2_prot_tax"])
            deltas = pd.read_csv(results / "summary" / "deltas.csv")
            auc = deltas[deltas["metric"] == "internal_macro_roc_auc"].iloc[0]
            self.assertAlmostEqual(auc["delta_mean"], 0.01)
            self.assertEqual((auc["n_seeds_improved"], auc["n_seeds"]), (2, 2))
            self.assertIn("p_paired_t", deltas.columns)
            # S-min is lower-is-better, so a higher value counts as getting worse.
            s_min = deltas[deltas["metric"] == "official_s_w_overall_mean"].iloc[0]
            self.assertEqual(s_min["n_seeds_improved"], 0)
            for name in ("metric_by_config.png", "delta_heatmap.png", "summary.csv"):
                self.assertTrue((results / "summary" / name).exists(), name)
            for name in ("config.json", "metrics.csv", "per_term_auc.csv", "training_curves.png", "pr_curves.png"):
                self.assertTrue((results / "esm2" / name).exists(), name)
            self.assertFalse((results / "summary" / "promoter_strata.csv").exists())
            np.testing.assert_allclose(pd.read_csv(results / "esm2" / "metrics.csv")["official_macro_roc_auc_overall"], [0.8, 0.802])


class PartialCafaevalTests(unittest.TestCase):
    def test_best_config_uses_internal_auc_and_report_handles_missing_cafaeval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep = root / "sweep"
            for seed in (0, 1):
                _write_fake_run(sweep / "esm2" / f"seed{seed}", quality=0.0, cafaeval=False)
                _write_fake_run(sweep / "esm2_prot" / f"seed{seed}", quality=0.01, cafaeval=seed == 0)

            best = select_runs(sweep, list(CONFIGS), None, best=True)
            self.assertEqual([run.parent.name for run in best], ["esm2_prot", "esm2_prot"])

            write_report(sweep, root / "results", root / "release")

            official = pd.read_csv(root / "results" / "summary" / "cafaeval.csv")
            self.assertEqual(official[["config", "seed"]].values.tolist(), [["esm2_prot", 0]])
            deltas = pd.read_csv(root / "results" / "summary" / "deltas.csv")
            self.assertNotIn("official_f_w_overall_mean", set(deltas["metric"]))
            self.assertIn("official_macro_roc_auc_overall", set(deltas["metric"]))


if __name__ == "__main__":
    unittest.main()
