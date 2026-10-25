"""Random architecture and optimiser search for the multi-label network on one feature set.

Each trial trains once (seed 0) and is ranked on the *validation* split, never on the test split
or the official truth that the winner is then reported on. The winner is retrained on seeds
1 and 2, and compared with the default architecture on the same feature set (the matching
ablation runs), seed by seed.

    cafa6-hparam-search run --sweep arch --features all --trials 24
    cafa6-hparam-search finalize --sweep arch            # winner on seeds 1, 2
    cafa6-hparam-search finalize --sweep arch --features esm2_prot_tax --seeds 0 1 2   # winner, other features
    cafa6-hparam-search report --sweep arch              # tables and figures -> results/hparam-search/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling.experiments.ablation import COMMON_TRAIN_FLAGS, CONFIGS, DEFAULT_RELEASE_DIR, _run_logged, score_auc

LOGGER = logging.getLogger("cafa6.modeling.experiments.hparam_search")
DEFAULT_ROOT = Path("data/hparams")
SELECTION_METRIC = "validation_nn_cafa_fmax_mean"
SEARCH_SPACE = {
    "esm2_projection": (256, 512),
    "promoter_projection": (64, 128),
    "promoter_fusion": (32, 64, 128),
    "trunk_layers": (2, 3, 4),
    "trunk_dim": (512, 1024),
    "bottleneck_dim": (256, 512),
    "branch_dropout": (0.1, 0.2, 0.3),
    "trunk_dropout": (0.2, 0.35, 0.5),
    "learning_rate": (1e-4, 3e-4, 1e-3),
    "weight_decay": (1e-5, 1e-4, 1e-3),
}
# The trainer's defaults: trial 0, and the architecture used throughout the feature ablation.
DEFAULT_TRIAL = {
    "esm2_projection": 256,
    "promoter_projection": 128,
    "promoter_fusion": 128,
    "trunk_layers": 2,
    "trunk_dim": 512,
    "bottleneck_dim": 256,
    "branch_dropout": 0.1,
    "trunk_dropout": 0.2,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
}
REPORT_METRICS = {
    "validation_nn_cafa_fmax_mean": "Validation IA-weighted Fmax (selection)",
    "test_nn_cafa_fmax_mean": "Test IA-weighted Fmax",
    "test_nn_macro_roc_auc": "Test macro ROC-AUC",
    "official_macro_roc_auc_overall": "Official macro ROC-AUC",
}


def sample_trials(n_trials: int, seed: int = 0) -> list[dict]:
    """The default architecture, then distinct random draws from SEARCH_SPACE."""
    rng = np.random.default_rng(seed)
    trials = [dict(DEFAULT_TRIAL)]
    seen = {tuple(DEFAULT_TRIAL.values())}
    while len(trials) < n_trials:
        trial = {name: values[rng.integers(len(values))] for name, values in SEARCH_SPACE.items()}
        trial = {name: value.item() if isinstance(value, np.generic) else value for name, value in trial.items()}
        if tuple(trial.values()) not in seen:
            seen.add(tuple(trial.values()))
            trials.append(trial)
    return trials


def trial_flags(trial: dict) -> list[str]:
    projections = (
        f"esm2={trial['esm2_projection']}",
        f"promoter_nt={trial['promoter_projection']}",
        f"promoter_agront={trial['promoter_projection']}",
    )
    return [
        "--projection-dims", *projections,
        "--promoter-fusion-dim", str(trial["promoter_fusion"]),
        "--trunk-layers", str(trial["trunk_layers"]),
        "--trunk-dim", str(trial["trunk_dim"]),
        "--bottleneck-dim", str(trial["bottleneck_dim"]),
        "--branch-dropout", str(trial["branch_dropout"]),
        "--trunk-dropout", str(trial["trunk_dropout"]),
        "--learning-rate", str(trial["learning_rate"]),
        "--weight-decay", str(trial["weight_decay"]),
    ]


def train_trial(
    sweep_dir: Path, index: int, trial: dict, features: str, seed: int, release_dir: Path, name: str | None = None
) -> Path:
    """Train, predict the ground-truth proteins and AUC-score one trial x seed (resumable)."""
    directory = sweep_dir / (name or f"trial{index:02d}") / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory.parent / "trial.json").write_text(json.dumps({"features": features, **trial}, indent=2) + "\n")
    if not (directory / "model" / "summary.json").exists():
        command = [
            sys.executable, "-m", "modeling.nn.train",
            *COMMON_TRAIN_FLAGS, *CONFIGS[features].flags, *trial_flags(trial),
            "--random-state", str(seed),
            "--output-dir", str(directory / "model"),
            "--promoter-feature-cache", str(directory / "promoter_cache.csv"),
        ]
        _run_logged(command, directory / "logs" / "train.log")
    for cache in directory.glob("promoter_cache.csv*"):
        cache.unlink()
    predictions = directory / "predictions" / f"{directory.parent.name}_seed{seed}.tsv"
    if not predictions.exists():
        truth_files = [str(release_dir / f"May2026_groundtruth_{regime}.tsv") for regime in ("NK", "LK", "PK")]
        command = [sys.executable, "-m", "modeling.nn.predict", str(directory / "model"), "--output", str(predictions), "--protein-ids", *truth_files]
        _run_logged(command, directory / "logs" / "predict.log")
    return score_auc(directory, release_dir)


def run_metrics(run: Path) -> dict:
    """Validation, test and official-AUC metrics of one finished run directory (trial or ablation)."""
    summary = json.loads((run / "model" / "summary.json").read_text())
    run_config = json.loads((run / "model" / "run_config.json").read_text())
    auc = json.loads((run / "official_auc.json").read_text())
    return {
        "seed": int(run.name.removeprefix("seed")),
        "n_parameters": run_config["n_parameters"],
        "best_epoch": run_config["best_epoch"],
        **{key: summary[key] for key in summary if key.startswith(("validation_nn_", "test_nn_"))},
        "official_macro_roc_auc_overall": np.mean([auc[regime]["macro_roc_auc"] for regime in auc]),
        "official_micro_roc_auc_overall": np.mean([auc[regime]["micro_roc_auc"] for regime in auc]),
    }


def collect_trials(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    # Search trials only (trialNN); retrained winners on other features (trialNN_<features>) are excluded.
    for trial_dir in sorted(path for path in sweep_dir.glob("trial*") if path.name.removeprefix("trial").isdigit()):
        trial = json.loads((trial_dir / "trial.json").read_text())
        for run in sorted(trial_dir.glob("seed*")):
            if (run / "official_auc.json").exists():
                rows.append({"trial": trial_dir.name, **trial, **run_metrics(run)})
    return pd.DataFrame(rows)


def best_trial(trials: pd.DataFrame) -> str:
    """Highest seed-0 validation Fmax; selection never looks at test or official metrics."""
    seed0 = trials[trials["seed"] == 0]
    return str(seed0.loc[seed0[SELECTION_METRIC].idxmax(), "trial"])


def plot_hyperparameter_effects(trials: pd.DataFrame, winner: str, path: Path) -> None:
    seed0 = trials[trials["seed"] == 0]
    names = list(SEARCH_SPACE)
    columns = 5
    figure, axes = plt.subplots(2, columns, figsize=(3.2 * columns, 6.4), sharey=True, facecolor="#fcfcfb", squeeze=False)
    for ax, name in zip(axes.flat, names):
        values = list(SEARCH_SPACE[name])
        positions = seed0[name].map({value: index for index, value in enumerate(values)})
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(seed0))
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.scatter(positions + jitter, seed0[SELECTION_METRIC], s=22, color="#2a78d6", alpha=0.55, linewidths=0)
        means = seed0.groupby(name)[SELECTION_METRIC].mean()
        ax.plot([values.index(value) for value in means.index], means.to_numpy(), color="#0b0b0b", linewidth=1.5, marker="_", markersize=14)
        is_winner = seed0["trial"] == winner
        ax.scatter(positions[is_winner], seed0.loc[is_winner, SELECTION_METRIC], s=90, color="#eb6834", edgecolors="#fcfcfb", linewidths=2, zorder=3)
        ax.set_xticks(range(len(values)), [f"{value:g}" for value in values], fontsize=8, color="#52514e")
        ax.set_title(name.replace("_", " "), fontsize=10, color="#0b0b0b", loc="left")
    axes[0][0].set_ylabel("Validation IA-weighted Fmax", fontsize=9, color="#52514e")
    axes[1][0].set_ylabel("Validation IA-weighted Fmax", fontsize=9, color="#52514e")
    figure.suptitle("Architecture search: each dot is one trial (seed 0); bar = mean per value; orange = selected", fontsize=11, color="#0b0b0b", x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(figure)


def write_report(sweep_dir: Path, out_dir: Path, ablation_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = collect_trials(sweep_dir)
    features = json.loads(next(sweep_dir.glob("trial*/trial.json")).read_text())["features"]
    winner = best_trial(trials)
    trials.sort_values(["seed", SELECTION_METRIC], ascending=[True, False]).round(6).to_csv(out_dir / "trials.csv", index=False)
    plot_hyperparameter_effects(trials, winner, out_dir / "hyperparameter_effects.png")

    # Winner vs the default architecture on the same features and seeds (the ablation runs).
    default = pd.DataFrame(
        [run_metrics(run) for run in sorted((ablation_dir / features).glob("seed*")) if (run / "official_auc.json").exists()]
    ).set_index("seed")
    chosen = trials[trials["trial"] == winner].set_index("seed")
    seeds = sorted(set(chosen.index) & set(default.index))
    rows = []
    for metric, label in REPORT_METRICS.items():
        deltas = chosen.loc[seeds, metric].to_numpy() - default.loc[seeds, metric].to_numpy()
        rows.append(
            {
                "metric": label,
                "default_mean": default.loc[seeds, metric].mean(),
                "selected_mean": chosen.loc[seeds, metric].mean(),
                "delta_mean": deltas.mean(),
                "delta_sd": deltas.std(ddof=1) if len(deltas) > 1 else np.nan,
                "n_seeds_improved": int((deltas > 0).sum()),
                "n_seeds": len(deltas),
            }
        )
    pd.DataFrame(rows).round(6).to_csv(out_dir / "selected_vs_default.csv", index=False)
    (out_dir / "selected.json").write_text(
        json.dumps({"trial": winner, "selection_metric": SELECTION_METRIC, **json.loads((sweep_dir / winner / "trial.json").read_text())}, indent=2) + "\n"
    )
    LOGGER.info("selected %s; wrote %s", winner, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Train the random trials on seed 0")
    run_parser.add_argument("--features", choices=tuple(CONFIGS), default="all", help="Feature configuration from the ablation")
    run_parser.add_argument("--trials", type=int, default=24)
    run_parser.add_argument("--search-seed", type=int, default=0, help="Seed for drawing the trials")
    finalize_parser = commands.add_parser("finalize", help="Retrain the selected trial on more seeds")
    finalize_parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    finalize_parser.add_argument(
        "--features", choices=tuple(CONFIGS), default=None, help="Retrain the winner on another feature set (default: its own)"
    )
    report_parser = commands.add_parser("report", help="Tables and figures")
    report_parser.add_argument("--out", type=Path, default=Path("results/hparam-search"))
    report_parser.add_argument("--ablation-sweep", type=Path, default=Path("data/ablation/feature-groups"))
    for sub in (run_parser, finalize_parser, report_parser):
        sub.add_argument("--sweep", required=True)
        sub.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        sub.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
        sub.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sweep_dir = args.root / args.sweep

    if args.command == "run":
        for index, trial in enumerate(sample_trials(args.trials, args.search_seed)):
            train_trial(sweep_dir, index, trial, args.features, seed=0, release_dir=args.release_dir)
            LOGGER.info("finished trial %02d: %s", index, trial)
    elif args.command == "finalize":
        trials = collect_trials(sweep_dir)
        winner = best_trial(trials)
        trial = json.loads((sweep_dir / winner / "trial.json").read_text())
        searched_features = trial.pop("features")
        features = args.features or searched_features
        # The winner on other features lives beside the trials, e.g. trial10_esm2_prot_tax/.
        name = None if features == searched_features else f"{winner}_{features}"
        LOGGER.info("selected %s on %s: %s; training on %s", winner, SELECTION_METRIC, trial, features)
        for seed in args.seeds:
            train_trial(sweep_dir, int(winner.removeprefix("trial")), trial, features, seed, args.release_dir, name=name)
    else:
        write_report(sweep_dir, args.out, args.ablation_sweep)


if __name__ == "__main__":
    main()
