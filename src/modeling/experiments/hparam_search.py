"""Architecture and optimiser search for the multi-label network.

Each trial trains once (seed 0) and is ranked on the *validation* split, never on the test split
or the official truth that the winner is then reported on. The winner is retrained on seeds
1 and 2, and compared with the default architecture on the same feature set (the matching
ablation runs), seed by seed.

Two samplers:
- `random` draws from SEARCH_SPACE on one feature set (the first search, sweep `arch`);
- `tpe` is Bayesian optimisation with Optuna's TPE sampler over TPE_SPACE, where the promoter
  fusion width doubles as the feature choice (0 = no promoter features). A Hyperband pruner stops
  trials whose validation Fmax falls behind from epoch 15 on, so more configurations fit in the
  time budget. `--warm-start` adds finished trials (e.g. the random search), with their per-epoch
  validation curves, before the first new trial, so TPE and the pruner start from that evidence.
  `--warm-start-study` does the same from an earlier TPE study (finished and pruned trials), so a
  new study can search a wider TPE_SPACE without repeating the earlier one. The study is stored
  in `<sweep>/study.db`, so a stopped search resumes.

    cafa6-hparam-search run --sweep arch --features all --trials 24
    cafa6-hparam-search run --sweep arch2 --sampler tpe --trials 200 --hours 14.5 \\
        --warm-start data/hparams/arch/trial*
    cafa6-hparam-search finalize --sweep arch            # winner on seeds 1, 2
    cafa6-hparam-search finalize --sweep arch2 --subset no-promoters   # best trial without promoters
    cafa6-hparam-search finalize --sweep arch --features esm2_prot_tax --seeds 0 1 2   # winner, other features
    cafa6-hparam-search report --sweep arch              # tables and figures -> results/hparam-search/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from modeling.evaluation.go import GeneOntology, load_ia_weights
from modeling.evaluation.significance import official_curves, point_estimate
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
# The TPE search: the random search's space (so its trials can warm-start the study) extended with
# wider layers, smaller promoter vectors and the training tweaks the random search held fixed.
# Sweep arch3 widened the edges arch2's best trials sat on (bottleneck, ESM-2 projection, promoter
# fusion, weight decay) and added the asymmetric loss's own settings; it is a superset of arch2's
# space, so arch2's trials warm-start it.
TPE_SPACE = {
    "promoter_fusion": (0, 16, 32, 64, 128, 256),  # 0: no promoter features (esm2_prot_tax)
    "promoter_projection": (64, 128, 256),  # only with promoters
    "esm2_projection": (256, 512, 768, 1024),
    "trunk_dim": (512, 1024, 1536, 2048, 3072),
    "bottleneck_dim": (256, 512, 768, 1024, 1536, 2048),
    "trunk_layers": (2, 3, 4),
    "trunk_residual": (False, True),  # only with 3+ layers; residual blocks between first and bottleneck layer
    "branch_dropout": (0.1, 0.2, 0.3),
    "trunk_dropout": (0.1, 0.2, 0.3, 0.35, 0.4, 0.5),
    "learning_rate": (1e-4, 3e-4, 6e-4, 1e-3, 2e-3),
    "weight_decay": (0.0, 1e-5, 1e-4, 1e-3, 1e-2),
    "lr_schedule": ("constant", "cosine"),  # cosine: 2 warmup epochs, decays to 0 over 40 epochs
    "ema_decay": (0.0, 0.998),
    "batch_size": (256, 512, 1024),
    "loss": ("weighted_bce", "asymmetric"),
    "max_pos_weight": (10.0, 20.0, 50.0),  # only with weighted_bce
    "gamma_negative": (1.0, 2.0, 3.0, 4.0, 6.0),  # only with asymmetric: focusing on easy negatives
    "gamma_positive": (0.0, 1.0),  # only with asymmetric
    "asymmetric_clip": (0.0, 0.02, 0.05, 0.1),  # only with asymmetric: probability margin for negatives
}
ASYMMETRIC_PARAMS = ("gamma_negative", "gamma_positive", "asymmetric_clip")
# Trainer settings the random search held fixed; a trial that doesn't set one uses these.
TRAINING_DEFAULTS = {
    "trunk_residual": False,
    "lr_schedule": "constant",
    "ema_decay": 0.0,
    "batch_size": 512,
    "loss": "weighted_bce",
    "max_pos_weight": 20.0,
    "gamma_negative": 4.0,
    "gamma_positive": 0.0,
    "asymmetric_clip": 0.05,
}
COSINE_EPOCHS, WARMUP_EPOCHS = 40, 2
# Hyperband: no trial is pruned before epoch 15; rungs at 15 and 45 epochs keep the best third.
PRUNE_MIN_EPOCHS, PRUNE_MAX_EPOCHS, PRUNE_REDUCTION = 15, 100, 3
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


def suggest_trial(trial: optuna.Trial) -> dict:
    """One TPE draw from TPE_SPACE; conditional parameters are only drawn where they matter."""
    def pick(name: str):
        return trial.suggest_categorical(name, TPE_SPACE[name])

    params = {"promoter_fusion": pick("promoter_fusion")}
    if params["promoter_fusion"]:
        params["promoter_projection"] = pick("promoter_projection")
    for name in ("esm2_projection", "trunk_dim", "bottleneck_dim", "trunk_layers"):
        params[name] = pick(name)
    if params["trunk_layers"] > 2:
        params["trunk_residual"] = pick("trunk_residual")
    for name in ("branch_dropout", "trunk_dropout", "learning_rate", "weight_decay", "lr_schedule", "ema_decay", "batch_size", "loss"):
        params[name] = pick(name)
    if params["loss"] == "weighted_bce":
        params["max_pos_weight"] = pick("max_pos_weight")
    else:
        params.update({name: pick(name) for name in ASYMMETRIC_PARAMS})
    return params


def complete_trial(params: dict) -> tuple[str, dict]:
    """Feature set and the full trial (defaults filled in) for a TPE parameter draw."""
    features = "all" if params["promoter_fusion"] else "esm2_prot_tax"
    return features, {**DEFAULT_TRIAL, **TRAINING_DEFAULTS, **params}


def tpe_params(trial_dir: Path) -> dict:
    """A finished trial's settings as TPE_SPACE parameters, for adding it to a TPE study."""
    trial = {**TRAINING_DEFAULTS, **json.loads((trial_dir / "trial.json").read_text())}
    trial["promoter_fusion"] = 0 if trial["features"] == "esm2_prot_tax" else trial["promoter_fusion"]
    params = {}
    for name, values in TPE_SPACE.items():
        conditional = (
            (name == "promoter_projection" and not trial["promoter_fusion"])
            or (name == "trunk_residual" and trial["trunk_layers"] <= 2)
            or (name == "max_pos_weight" and trial["loss"] != "weighted_bce")
            or (name in ASYMMETRIC_PARAMS and trial["loss"] != "asymmetric")
        )
        if conditional:
            continue
        if trial[name] not in values:
            raise ValueError(f"{trial_dir}: {name}={trial[name]!r} is outside TPE_SPACE {values}")
        params[name] = trial[name]
    return params


def trial_flags(trial: dict) -> list[str]:
    trial = {**TRAINING_DEFAULTS, **trial}
    schedule = ["--lr-schedule", "cosine", "--epochs", str(COSINE_EPOCHS), "--warmup-epochs", str(WARMUP_EPOCHS)] if trial["lr_schedule"] == "cosine" else []
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
        "--batch-size", str(trial["batch_size"]),
        "--loss", trial["loss"],
        "--max-pos-weight", str(trial["max_pos_weight"]),
        "--gamma-negative", str(trial["gamma_negative"]),
        "--gamma-positive", str(trial["gamma_positive"]),
        "--asymmetric-clip", str(trial["asymmetric_clip"]),
        "--ema-decay", str(trial["ema_decay"]),
        *(["--trunk-residual"] if trial["trunk_residual"] else []),
        *schedule,
    ]


def _run_pruned(command: list[str], log_path: Path, progress_path: Path, trial: optuna.Trial) -> None:
    """Run the trainer, reporting each epoch's best-so-far validation score to Optuna, and stop it
    (raising TrialPruned) as soon as the pruner says the trial is losing."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.unlink(missing_ok=True)
    best, reported = -np.inf, 0
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                done = process.poll() is not None
                # Complete lines only: the trainer may be mid-write.
                lines = progress_path.read_text().split("\n")[:-1] if progress_path.exists() else []
                for line in lines[reported:]:
                    record = json.loads(line)
                    best = max(best, record["selection_score"])
                    trial.report(best, record["epoch"])
                    if trial.should_prune():
                        raise optuna.TrialPruned(f"epoch {record['epoch']}: best validation Fmax {best:.4f}")
                reported = len(lines)
                if done:
                    break
                time.sleep(5)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def train_trial(
    sweep_dir: Path, index: int, trial: dict, features: str, seed: int, release_dir: Path, name: str | None = None,
    pruning_trial: optuna.Trial | None = None,
) -> Path:
    """Train, predict the ground-truth proteins and AUC-score one trial x seed (resumable). With
    `pruning_trial`, training reports to Optuna and may stop early with TrialPruned."""
    directory = sweep_dir / (name or f"trial{index:02d}") / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory.parent / "trial.json").write_text(json.dumps({"features": features, **trial}, indent=2) + "\n")
    try:
        if not (directory / "model" / "summary.json").exists():
            command = [
                sys.executable, "-m", "modeling.nn.train",
                *COMMON_TRAIN_FLAGS, *CONFIGS[features].flags, *trial_flags(trial),
                "--random-state", str(seed),
                "--output-dir", str(directory / "model"),
                "--promoter-feature-cache", str(directory / "promoter_cache.csv"),
            ]
            if pruning_trial is None:
                _run_logged(command, directory / "logs" / "train.log")
            else:
                progress = directory / "logs" / "progress.jsonl"
                _run_pruned([*command, "--progress-file", str(progress)], directory / "logs" / "train.log", progress, pruning_trial)
    finally:
        for cache in directory.glob("promoter_cache.csv*"):
            cache.unlink()
    # A reused run (a TPE seed run) keeps its predictions under its original trial name.
    existing = sorted((directory / "predictions").glob("*.tsv"))
    predictions = existing[0] if existing else directory / "predictions" / f"{directory.parent.name}_seed{seed}.tsv"
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


def official_fmax(run: Path, release_dir: Path) -> float:
    """IA-weighted Fmax on the official NK + LK targets (the bootstrap's reproduction of cafaeval),
    cached in the run. Reported next to each trial; never used to select one."""
    path = run / "official_fmax.json"
    if not path.exists():
        config = json.loads((run / "model" / "run_config.json").read_text())
        predictions = next((run / "predictions").glob("*.tsv"))
        curves = official_curves(predictions, release_dir, GeneOntology.from_obo(config["go_obo_path"]), load_ia_weights(config["ia_path"]))
        path.write_text(json.dumps(point_estimate(curves), indent=2) + "\n")
    return json.loads(path.read_text())["overall"]


def collect_trials(sweep_dir: Path, release_dir: Path | None = None) -> pd.DataFrame:
    """One row per finished trial x seed; with `release_dir`, also its official NK + LK Fmax."""
    rows = []
    # Search trials only (trialNN); retrained winners on other features (trialNN_<features>) are excluded.
    for trial_dir in sorted(path for path in sweep_dir.glob("trial*") if path.name.removeprefix("trial").isdigit()):
        trial = json.loads((trial_dir / "trial.json").read_text())
        for run in sorted(trial_dir.glob("seed*")):
            if (run / "official_auc.json").exists():
                row = {"trial": trial_dir.name, **trial, **run_metrics(run)}
                if release_dir is not None:
                    row["official_f_w_nk_lk"] = official_fmax(run, release_dir)
                rows.append(row)
    return pd.DataFrame(rows)


SUBSETS = {"all": lambda features: True, "no-promoters": lambda features: features == "esm2_prot_tax", "promoters": lambda features: features == "all"}


def best_trial(trials: pd.DataFrame, subset: str = "all") -> str:
    """Highest seed-0 validation Fmax within `subset`; selection never looks at test or official metrics."""
    seed0 = trials[trials["seed"] == 0]
    if subset != "all":
        seed0 = seed0[seed0["features"].map(SUBSETS[subset])]
    return str(seed0.loc[seed0[SELECTION_METRIC].idxmax(), "trial"])


LOSS_COLORS = {"weighted_bce": "#2a78d6", "asymmetric": "#eb6834"}
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _applicable(trials: pd.DataFrame, name: str) -> pd.Series:
    """Whether each trial used `name`: conditional TPE parameters are filled with defaults where they don't apply."""
    if name == "promoter_projection":
        return trials["features"] == "all"
    if name == "trunk_residual":
        return trials["trunk_layers"] > 2
    if name == "max_pos_weight" and "loss" in trials:
        return trials["loss"] == "weighted_bce"
    if name in ASYMMETRIC_PARAMS:
        return trials["loss"] == "asymmetric"
    return pd.Series(True, index=trials.index)


def _style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_hyperparameter_effects(trials: pd.DataFrame, winner: str, path: Path, space: dict[str, tuple] = SEARCH_SPACE) -> None:
    """Validation Fmax by value of each hyperparameter. With more than one loss, points and mean
    lines are split by loss, since the loss shifts every score and would otherwise dominate."""
    seed0 = trials[trials["seed"] == 0]
    names = list(space)
    by_loss = "loss" in seed0 and seed0["loss"].nunique() > 1
    columns = 5 if len(names) <= 10 else 4
    rows = -(-len(names) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows), sharey=True, facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    groups = [(loss, seed0["loss"] == loss, LOSS_COLORS[loss]) for loss in LOSS_COLORS] if by_loss else [(None, pd.Series(True, index=seed0.index), "#2a78d6")]
    for ax, name in zip(axes.flat, names):
        values = list(space[name])
        if name not in seed0:
            continue
        _style(ax)
        used = _applicable(seed0, name)
        positions = seed0[name].map({value: index for index, value in enumerate(values)})
        jitter = pd.Series(np.random.default_rng(0).uniform(-0.12, 0.12, len(seed0)), index=seed0.index)
        for loss, selected, color in groups:
            mask = selected & used
            ax.scatter(positions[mask] + jitter[mask], seed0.loc[mask, SELECTION_METRIC], s=20, color=color, alpha=0.55, linewidths=0)
            means = seed0[mask].groupby(name)[SELECTION_METRIC].mean()
            line_color = color if by_loss else TEXT
            ax.plot([values.index(value) for value in means.index], means.to_numpy(), color=line_color, linewidth=1.5, marker="_", markersize=14)
        is_winner = (seed0["trial"] == winner) & used
        ax.scatter(positions[is_winner], seed0.loc[is_winner, SELECTION_METRIC], s=110, marker="*", color=TEXT if by_loss else "#eb6834", edgecolors=SURFACE, linewidths=1, zorder=3)
        ax.set_xticks(range(len(values)), [f"{value:g}" if isinstance(value, float | int) and not isinstance(value, bool) else str(value) for value in values], fontsize=8, color=TEXT_MUTED)
        ax.set_title(name.replace("_", " "), fontsize=10, color=TEXT, loc="left")
    for row in axes:
        row[0].set_ylabel("Validation IA-weighted Fmax", fontsize=9, color=TEXT_MUTED)
    if by_loss:
        handles = [plt.Line2D([], [], marker="o", linestyle="-", color=color, label=f"{loss} (mean per value)") for loss, color in LOSS_COLORS.items()]
        handles.append(plt.Line2D([], [], marker="*", linestyle="", color=TEXT, markersize=11, label="selected"))
        figure.legend(handles=handles, frameon=False, fontsize=9, loc="upper right", ncols=3)
        title = "Hyperparameter search: each dot is one finished trial (seed 0), split by loss; settings a trial didn't use are hidden"
    else:
        title = "Architecture search: each dot is one trial (seed 0); bar = mean per value; orange = selected"
    figure.suptitle(title, fontsize=11, color=TEXT, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.985) if by_loss else (0, 0, 1, 1))
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def _phase(trial: optuna.trial.FrozenTrial) -> str:
    source = trial.user_attrs.get("warm_start", "")
    if "/arch/" in source:
        return "random search"
    if "/arch2/" in source:
        return "TPE, first space"
    return "TPE, wider space"


def plot_search_history(sweep_dir: Path, path: Path) -> None:
    """Every trial in order (warm-start studies first): finished trials filled, pruned trials hollow at
    their best score before stopping, with the running best."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=sweep_dir.name, storage=f"sqlite:///{sweep_dir / 'study.db'}")
    rows = []
    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)):
        complete = trial.state == optuna.trial.TrialState.COMPLETE
        value = trial.value if complete else max(trial.intermediate_values.values(), default=np.nan)
        rows.append({"number": trial.number, "value": value, "complete": complete, "loss": trial.params.get("loss", "weighted_bce"), "phase": _phase(trial)})
    history = pd.DataFrame(rows).sort_values("number")
    figure, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    _style(ax)
    for phase, shade in zip(("random search", "TPE, first space", "TPE, wider space"), ("#f0efec", SURFACE, "#f0efec")):
        numbers = history.loc[history["phase"] == phase, "number"]
        if len(numbers):
            ax.axvspan(numbers.min() - 0.5, numbers.max() + 0.5, color=shade, zorder=0)
            ax.text((numbers.min() + numbers.max()) / 2, 1.0, f"{phase}\n({len(numbers)} trials)", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=TEXT_MUTED)
    for loss, color in LOSS_COLORS.items():
        for complete in (True, False):
            subset = history[(history["loss"] == loss) & (history["complete"] == complete)]
            ax.scatter(subset["number"], subset["value"], s=26, facecolors=color if complete else "none", edgecolors=color, linewidths=1.2, zorder=3,
                       label=f"{loss}, {'finished' if complete else 'pruned (best before stopping)'}")
    finished = history[history["complete"]]
    ax.step(finished["number"], finished["value"].cummax(), where="post", color=TEXT, linewidth=1.5, zorder=2, label="best so far")
    ax.set_xlabel("Trial (in order)", fontsize=9, color=TEXT_MUTED)
    ax.set_ylabel("Validation IA-weighted Fmax (seed 0)", fontsize=9, color=TEXT_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Search history: Bayesian optimisation finds the asymmetric loss and keeps improving on it", fontsize=10.5, color=TEXT, loc="left", pad=30)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def plot_validation_vs_official(trials: pd.DataFrame, highlight: dict[str, str], path: Path) -> None:
    """Selection metric against the official (NK + LK) Fmax for every finished trial, seed 0."""
    from scipy.stats import spearmanr

    seed0 = trials[trials["seed"] == 0]
    figure, ax = plt.subplots(figsize=(7.5, 5.4), facecolor=SURFACE)
    _style(ax, grid_axis="both")
    titles = []
    for loss, color in LOSS_COLORS.items():
        for promoters, marker in ((True, "o"), (False, "s")):
            subset = seed0[(seed0["loss"] == loss) & ((seed0["features"] == "all") == promoters)]
            ax.scatter(subset[SELECTION_METRIC], subset["official_f_w_nk_lk"], s=30, marker=marker, color=color, alpha=0.7, linewidths=0,
                       label=f"{loss}, {'with' if promoters else 'no'} promoters")
        group = seed0[seed0["loss"] == loss]
        titles.append(f"{loss} ρ = {spearmanr(group[SELECTION_METRIC], group['official_f_w_nk_lk']).statistic:.2f}")
    for trial, label in highlight.items():
        row = seed0[seed0["trial"] == trial]
        if len(row):
            ax.annotate(label, (row[SELECTION_METRIC].iloc[0], row["official_f_w_nk_lk"].iloc[0]), xytext=(6, -10), textcoords="offset points", fontsize=8, color=TEXT,
                        arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "linewidth": 0.6})
    ax.set_xlabel("Validation IA-weighted Fmax (selection metric)", fontsize=9, color=TEXT_MUTED)
    ax.set_ylabel("Official IA-weighted Fmax, NK + LK (CAFA6 targets)", fontsize=9, color=TEXT_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Validation tracks the official score across losses, less well within the best region\n"
                 f"one seed per trial; Spearman: {', '.join(titles)}", fontsize=10, color=TEXT, loc="left")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def write_report(sweep_dir: Path, out_dir: Path, ablation_dir: Path, release_dir: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    is_tpe = (sweep_dir / "study.db").exists()
    trials = collect_trials(sweep_dir, release_dir if is_tpe else None)
    winner = best_trial(trials)
    features = json.loads((sweep_dir / winner / "trial.json").read_text())["features"]
    trials.sort_values(["seed", SELECTION_METRIC], ascending=[True, False]).round(6).to_csv(out_dir / "trials.csv", index=False)
    plot_hyperparameter_effects(trials, winner, out_dir / "hyperparameter_effects.png", TPE_SPACE if is_tpe else SEARCH_SPACE)
    if is_tpe:
        plot_search_history(sweep_dir, out_dir / "search_history.png")
        plot_validation_vs_official(trials, {winner: f"{winner} (selected)"}, out_dir / "validation_vs_official.png")

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


def warm_start(study: optuna.Study, sweep_dir: Path, trial_dirs: list[Path]) -> None:
    """Add finished trials (seed 0) to the study with their per-epoch validation curves, and link
    their runs into the sweep, so TPE, the pruner, the report and `finalize` all see them."""
    for trial_dir in trial_dirs:
        params = tpe_params(trial_dir)
        model_dir = trial_dir / "seed0" / "model"
        history = pd.read_csv(model_dir / "history.tsv", sep="\t")
        curve = history["cafa_fmax_mean"].cummax()
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions={name: optuna.distributions.CategoricalDistribution(TPE_SPACE[name]) for name in params},
                value=json.loads((model_dir / "summary.json").read_text())[SELECTION_METRIC],
                intermediate_values={int(epoch): float(value) for epoch, value in zip(history["epoch"], curve)},
                user_attrs={"warm_start": str(trial_dir)},
            )
        )
        target = sweep_dir / f"trial{study.trials[-1].number:02d}"
        target.mkdir(parents=True, exist_ok=True)
        if not (target / "seed0").exists():
            os.symlink((trial_dir / "seed0").resolve(), target / "seed0")
        features, trial = complete_trial(params)
        (target / "trial.json").write_text(json.dumps({"features": features, **trial, "warm_start": str(trial_dir)}, indent=2) + "\n")


def warm_start_from_study(study: optuna.Study, sweep_dir: Path, source_dir: Path) -> None:
    """Add an earlier TPE study's finished and pruned trials, re-expressed in the current TPE_SPACE
    (a superset), and link the finished trials' runs into the sweep."""
    source = optuna.load_study(study_name=source_dir.name, storage=f"sqlite:///{source_dir / 'study.db'}")
    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    for old in source.get_trials(deepcopy=False, states=states):
        params = dict(old.params)
        if params.get("loss") == "asymmetric":
            # Trained with the trainer's defaults before these were searched.
            params.update({name: params.get(name, TRAINING_DEFAULTS[name]) for name in ASYMMETRIC_PARAMS})
        run = (source_dir / f"trial{old.number:02d}" / "seed0").resolve()
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions={name: optuna.distributions.CategoricalDistribution(TPE_SPACE[name]) for name in params},
                value=old.value if old.state == optuna.trial.TrialState.COMPLETE else None,
                intermediate_values=dict(old.intermediate_values),
                state=old.state,
                user_attrs={"warm_start": str(run.parent)},
            )
        )
        if old.state == optuna.trial.TrialState.COMPLETE and (run / "official_auc.json").exists():
            target = sweep_dir / f"trial{study.trials[-1].number:02d}"
            target.mkdir(parents=True, exist_ok=True)
            if not (target / "seed0").exists():
                os.symlink(run, target / "seed0")
            features, trial = complete_trial(params)
            (target / "trial.json").write_text(json.dumps({"features": features, **trial, "warm_start": str(run.parent)}, indent=2) + "\n")


def run_tpe(
    sweep_dir: Path, n_trials: int, hours: float | None, warm_start_dirs: list[Path], release_dir: Path, search_seed: int,
    warm_start_study: Path | None = None,
) -> None:
    """TPE + Hyperband over TPE_SPACE, stored in `<sweep>/study.db`. Runs until `n_trials` new trials
    have finished or been pruned, or `hours` have passed; reruns resume."""
    sweep_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=sweep_dir.name,
        storage=f"sqlite:///{sweep_dir / 'study.db'}",
        sampler=optuna.samplers.TPESampler(seed=search_seed, n_startup_trials=10, multivariate=True, group=True),
        pruner=optuna.pruners.HyperbandPruner(min_resource=PRUNE_MIN_EPOCHS, max_resource=PRUNE_MAX_EPOCHS, reduction_factor=PRUNE_REDUCTION),
        direction="maximize",
        load_if_exists=True,
    )
    # A trial still marked running was interrupted with the previous process; it is retrained under a new number.
    for stale in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(stale.number, state=optuna.trial.TrialState.FAIL)
    if not study.trials:
        warm_start(study, sweep_dir, warm_start_dirs)
        if warm_start_study is not None:
            warm_start_from_study(study, sweep_dir, warm_start_study)
        LOGGER.info("warm start: %d finished trials, best validation Fmax %.4f", len(study.trials), study.best_value)

    def objective(trial: optuna.Trial) -> float:
        features, params = complete_trial(suggest_trial(trial))
        LOGGER.info("trial %02d on %s: %s", trial.number, features, params)
        run = train_trial(sweep_dir, trial.number, params, features, seed=0, release_dir=release_dir, pruning_trial=trial)
        score = json.loads((run / "model" / "summary.json").read_text())[SELECTION_METRIC]
        LOGGER.info("finished trial %02d: validation Fmax %.4f (best %.4f)", trial.number, score, max(score, study.best_value))
        return score

    def new_trials() -> int:
        states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        return sum("warm_start" not in trial.user_attrs for trial in study.get_trials(deepcopy=False, states=states))

    started = time.monotonic()
    while new_trials() < n_trials and (hours is None or time.monotonic() - started < hours * 3600):
        # One trial per call, so a failed trial (logged, marked FAIL) doesn't count towards n_trials.
        study.optimize(objective, n_trials=1, catch=(Exception,))
    pruned = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,)))
    LOGGER.info("%d new trials (%d pruned); best trial%02d, validation Fmax %.4f", new_trials(), pruned, study.best_trial.number, study.best_value)


STATUS_COLUMNS = {
    "promoter_fusion": "prom", "esm2_projection": "esm", "trunk_dim": "trunk", "bottleneck_dim": "bneck", "trunk_layers": "layers",
    "trunk_residual": "res", "trunk_dropout": "drop", "learning_rate": "lr", "weight_decay": "wd", "lr_schedule": "sched",
    "ema_decay": "ema", "batch_size": "batch", "loss": "loss",
}


def print_status(sweep_dir: Path, top: int) -> None:
    """Leaderboard of a TPE study: trial counts, the running trial's progress, and the best trials."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=sweep_dir.name, storage=f"sqlite:///{sweep_dir / 'study.db'}")
    trials = study.get_trials(deepcopy=False)
    new = [trial for trial in trials if "warm_start" not in trial.user_attrs]
    counts = pd.Series([trial.state.name.lower() for trial in new]).value_counts().to_dict()
    started = min((trial.datetime_start for trial in new if trial.datetime_start), default=None)
    elapsed = f", {(pd.Timestamp.now() - pd.Timestamp(started)).total_seconds() / 3600:.1f} h since the first new trial" if started else ""
    print(f"{sweep_dir.name}: {len(trials) - len(new)} warm-start trials + {len(new)} new ({counts}){elapsed}")

    for trial in (trial for trial in new if trial.state == optuna.trial.TrialState.RUNNING):
        progress = sweep_dir / f"trial{trial.number:02d}" / "seed0" / "logs" / "progress.jsonl"
        lines = progress.read_text().split("\n")[:-1] if progress.exists() else []
        latest = json.loads(lines[-1]) if lines else None
        best = max(json.loads(line)["selection_score"] for line in lines) if lines else float("nan")
        where = f"epoch {latest['epoch']}, best validation Fmax {best:.4f}" if latest else "loading features"
        print(f"running: trial{trial.number:02d} ({where})")

    finished = [trial for trial in trials if trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    rows = []
    for trial in finished:
        # A pruned trial's score is its best validation Fmax before it was stopped.
        value = trial.value if trial.state == optuna.trial.TrialState.COMPLETE else max(trial.intermediate_values.values(), default=np.nan)
        rows.append(
            {
                "trial": f"trial{trial.number:02d}",
                "val_fmax": value,
                "state": "warm" if "warm_start" in trial.user_attrs else {"COMPLETE": "done", "PRUNED": "pruned"}[trial.state.name],
                "epochs": max(trial.intermediate_values, default=0),
                **{label: trial.params.get(name, "") for name, label in STATUS_COLUMNS.items()},
            }
        )
    board = pd.DataFrame(rows).sort_values("val_fmax", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\ntop {top} by validation Fmax (seed 0; pruned trials show their best before stopping):")
        print(board.head(top).to_string(index=False, float_format=lambda value: f"{value:.4g}"))
        no_promoters = board[(board["prom"] == 0) & (board["state"] != "pruned")]
        if len(no_promoters):
            print(f"\nbest without promoters: {no_promoters.iloc[0]['trial']} ({no_promoters.iloc[0]['val_fmax']:.4f})")


def write_comparison(final_dir: Path, models: list[str], pairs: list[tuple[str, str]], out_dir: Path, release_dir: Path) -> None:
    """Official evaluator scores per run of each model (seeds as found), and the paired protein
    bootstrap for each (reference, other) pair, in the format of the first search's comparison."""
    from modeling.evaluation.significance import compare_to_reference
    from modeling.experiments.ablation import collect_run

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        for run in sorted((final_dir / model).glob("seed[0-9]")):
            metrics = collect_run(run.resolve())
            rows.append(
                {
                    "config": model,
                    "seed": int(run.name.removeprefix("seed")),
                    "n_parameters": metrics["n_parameters"],
                    "test_nn_macro_roc_auc": metrics["internal_macro_roc_auc"],
                    "test_nn_cafa_fmax_mean": metrics["internal_cafa_fmax_mean"],
                    "official_macro_roc_auc_overall": metrics["official_macro_roc_auc_overall"],
                    **{f"{metric}_{regime}": metrics.get(f"official_{metric}_{regime}_mean") for metric in ("f_w", "f_micro_w", "s_w") for regime in ("overall", "NK", "LK", "PK")},
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "final_comparison_runs.csv", index=False)
    bootstrap = [compare_to_reference(final_dir, reference, [other], release_dir) for reference, other in pairs]
    pd.concat(bootstrap).round(6).to_csv(out_dir / "final_comparison_bootstrap.csv", index=False)
    LOGGER.info("wrote %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Train the random trials on seed 0")
    run_parser.add_argument("--features", choices=tuple(CONFIGS), default="all", help="Feature configuration from the ablation")
    run_parser.add_argument("--trials", type=int, default=24)
    run_parser.add_argument("--search-seed", type=int, default=0, help="Seed for drawing the trials")
    run_parser.add_argument("--sampler", choices=("random", "tpe"), default="random", help="random: SEARCH_SPACE on --features; tpe: Optuna over TPE_SPACE")
    run_parser.add_argument("--hours", type=float, default=None, help="tpe: start no new trial after this many hours")
    run_parser.add_argument("--warm-start", type=Path, nargs="*", default=[], help="tpe: finished trial directories to add to a new study (not retrained)")
    run_parser.add_argument("--warm-start-study", type=Path, default=None, help="tpe: an earlier TPE sweep directory whose trials seed a new study")
    finalize_parser = commands.add_parser("finalize", help="Retrain the selected trial on more seeds")
    finalize_parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    finalize_parser.add_argument(
        "--features", choices=tuple(CONFIGS), default=None, help="Retrain the winner on another feature set (default: its own)"
    )
    finalize_parser.add_argument("--subset", choices=tuple(SUBSETS), default="all", help="Pick the best trial among these feature sets")
    finalize_parser.add_argument("--trial", default=None, help="Retrain this trial (e.g. trial57) instead of the best one")
    status_parser = commands.add_parser("status", help="tpe: progress and leaderboard (safe while the search runs)")
    status_parser.add_argument("--top", type=int, default=15)
    compare_parser = commands.add_parser("compare", help="Official scores and paired bootstrap for finalized models")
    compare_parser.add_argument("--final-dir", type=Path, default=Path("data/final"), help="One directory (or link) per model, with seed*/ runs")
    compare_parser.add_argument("--models", nargs="+", required=True)
    compare_parser.add_argument("--pairs", nargs="+", required=True, help="REFERENCE:OTHER pairs for the bootstrap (reference minus other)")
    compare_parser.add_argument("--out", type=Path, required=True)
    report_parser = commands.add_parser("report", help="Tables and figures")
    report_parser.add_argument("--out", type=Path, default=Path("results/hparam-search"))
    report_parser.add_argument("--ablation-sweep", type=Path, default=Path("data/ablation/feature-groups"))
    for sub in (run_parser, finalize_parser, status_parser, compare_parser, report_parser):
        sub.add_argument("--sweep", required=sub is not compare_parser)
        sub.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        sub.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
        sub.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sweep_dir = args.root / args.sweep if args.sweep else None

    if args.command == "run" and args.sampler == "tpe":
        run_tpe(sweep_dir, args.trials, args.hours, args.warm_start, args.release_dir, args.search_seed, args.warm_start_study)
    elif args.command == "run":
        for index, trial in enumerate(sample_trials(args.trials, args.search_seed)):
            train_trial(sweep_dir, index, trial, args.features, seed=0, release_dir=args.release_dir)
            LOGGER.info("finished trial %02d: %s", index, trial)
    elif args.command == "finalize":
        trials = collect_trials(sweep_dir)
        winner = args.trial or best_trial(trials, args.subset)
        trial = json.loads((sweep_dir / winner / "trial.json").read_text())
        searched_features = trial.pop("features")
        features = args.features or searched_features
        # The winner on other features lives beside the trials, e.g. trial10_esm2_prot_tax/.
        name = None if features == searched_features else f"{winner}_{features}"
        LOGGER.info("selected %s on %s: %s; training on %s", winner, SELECTION_METRIC, trial, features)
        for seed in args.seeds:
            train_trial(sweep_dir, int(winner.removeprefix("trial")), trial, features, seed, args.release_dir, name=name)
    elif args.command == "status":
        print_status(sweep_dir, args.top)
    elif args.command == "compare":
        pairs = [tuple(pair.split(":", 1)) for pair in args.pairs]
        write_comparison(args.final_dir, args.models, pairs, args.out, args.release_dir)
    else:
        write_report(sweep_dir, args.out, args.ablation_sweep, args.release_dir)


if __name__ == "__main__":
    main()
