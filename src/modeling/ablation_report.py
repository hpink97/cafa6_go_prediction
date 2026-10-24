"""Tables and figures for a finished feature-group ablation sweep (see modeling.ablation).

Writes to `results/`:
    summary/runs.csv             one row per config x seed with every metric
    summary/summary.csv          mean, sd and seed count per config and headline metric
    summary/deltas.csv           paired (same-seed) differences for each feature-group comparison
    summary/promoter_strata.csv  promoter effect by promoter provenance and kingdom
    summary/cafaeval.csv         official CAFA evaluator scores, for runs where it was run
    summary/response_terms.csv   do promoters help "response to" GO terms more? (see modeling.term_groups)
    summary/response_term_deltas.csv  per-term AUC change from adding promoters
    summary/*.png                metric_by_config, delta_heatmap, promoter_strata
    <config>/                    config.json, metrics.csv, per_term_auc.csv, training_curves.png, pr_curves.png
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from modeling.ablation import COMMON_TRAIN_FLAGS, COMPARISONS, CONFIGS, collect_run
from modeling.go_evaluation import GeneOntology, build_truth_ia_totals, cafa_fmax, load_ia_weights
from modeling.term_groups import response_term_analysis

LOGGER = logging.getLogger("cafa6.modeling.ablation_report")
HEADLINE_METRICS = {
    "internal_macro_roc_auc": "Internal macro ROC-AUC",
    "internal_micro_roc_auc": "Internal micro ROC-AUC",
    "internal_cafa_fmax_mean": "Internal IA-weighted Fmax",
    "official_f_w_overall_mean": "Official weighted Fmax (f_w)",
    "official_f_micro_w_overall_mean": "Official weighted micro-F (f_micro_w)",
    "official_s_w_overall_mean": "Official weighted S-min (s_w, lower is better)",
    "official_macro_roc_auc_overall": "Official macro ROC-AUC",
    "official_micro_roc_auc_overall": "Official micro ROC-AUC",
}
HEATMAP_METRICS = {
    "internal_macro_roc_auc": "internal\nmacro AUC",
    "internal_micro_roc_auc": "internal\nmicro AUC",
    "internal_cafa_fmax_P": "internal\nFmax BP",
    "internal_cafa_fmax_F": "internal\nFmax MF",
    "internal_cafa_fmax_C": "internal\nFmax CC",
    "official_macro_roc_auc_NK": "official\nAUC NK",
    "official_macro_roc_auc_LK": "official\nAUC LK",
    "official_macro_roc_auc_PK": "official\nAUC PK",
}
ASPECT_NAMES = {"P": "Biological process", "F": "Molecular function", "C": "Cellular component"}
# Reference palette (dataviz skill): categorical slots 1-3, diverging blue <-> red around a gray midpoint.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("delta", ["#e34948", "#f0efec", "#2a78d6"])


def _git_revision() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "src"], capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_runs(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for name in CONFIGS:
        for directory in sorted((sweep_dir / name).glob("seed*")):
            if not (directory / "official_auc.json").exists():
                LOGGER.warning("skipping unfinished run %s", directory)
                continue
            rows.append({"config": name, "seed": int(directory.name.removeprefix("seed")), **collect_run(directory)})
    if not rows:
        raise FileNotFoundError(f"No finished runs under {sweep_dir}")
    return pd.DataFrame(rows).sort_values(["config", "seed"], key=lambda column: column.map(list(CONFIGS).index) if column.name == "config" else column)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in [*HEADLINE_METRICS, "n_parameters"] if metric in runs.columns]
    long = runs.melt(id_vars=["config", "seed"], value_vars=metrics, var_name="metric").dropna(subset=["value"])
    return long.groupby(["config", "metric"], sort=False)["value"].agg(mean="mean", sd="std", n_seeds="count").reset_index()


def cafaeval_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Official CAFA evaluator scores per regime, for the runs where the evaluator was run."""
    columns = [column for column in runs.columns if column.startswith(("official_f_w_", "official_f_micro_w_", "official_s_w_")) and column.endswith("_mean")]
    if not columns:
        return pd.DataFrame()
    return runs.loc[runs[columns].notna().any(axis=1), ["config", "seed", *columns]]


def paired_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in dict.fromkeys([*HEADLINE_METRICS, *HEATMAP_METRICS]) if metric in runs.columns]
    indexed = runs.set_index(["config", "seed"])
    rows = []
    for with_config, without_config, label in COMPARISONS:
        if with_config not in indexed.index.get_level_values(0) or without_config not in indexed.index.get_level_values(0):
            continue
        seeds = sorted(set(indexed.loc[with_config].index) & set(indexed.loc[without_config].index))
        for metric in metrics:
            deltas = indexed.loc[[(with_config, seed) for seed in seeds], metric].to_numpy() - indexed.loc[[(without_config, seed) for seed in seeds], metric].to_numpy()
            deltas = deltas[np.isfinite(deltas)]
            if not len(deltas):
                continue
            better = -deltas if metric.startswith("official_s_w") else deltas
            rows.append(
                {
                    "comparison": label,
                    "with": with_config,
                    "without": without_config,
                    "metric": metric,
                    "delta_mean": deltas.mean(),
                    "delta_sd": deltas.std(ddof=1) if len(deltas) > 1 else np.nan,
                    "n_seeds_improved": int((better > 0).sum()),
                    "n_seeds": len(deltas),
                }
            )
    return pd.DataFrame(rows)


def promoter_strata(sweep_dir: Path, runs: pd.DataFrame, without: str = "esm2_prot_tax", with_: str = "all") -> pd.DataFrame:
    """Internal test Fmax with vs without promoters, split by promoter provenance and by kingdom."""
    seeds = sorted(set(runs.loc[runs["config"] == without, "seed"]) & set(runs.loc[runs["config"] == with_, "seed"]))
    if not seeds:
        return pd.DataFrame()
    config = json.loads((sweep_dir / with_ / f"seed{seeds[0]}" / "model" / "run_config.json").read_text())
    ontology = GeneOntology.from_obo(config["go_obo_path"])
    ia_weights = load_ia_weights(config["ia_path"])
    sources = pd.read_csv(
        Path(config["promoter_nt_dir"]).parent.parent / "promoters" / "train" / "combined.tsv",
        sep="\t",
        usecols=["protein_id", "promoter_source"],
    ).drop_duplicates("protein_id").set_index("protein_id")["promoter_source"]
    kingdoms = pd.read_csv(config["taxonomy_features_path"], sep="\t").set_index("protein_id")["kingdom_eukaryota_viridiplantae"]

    rows = []
    for seed in seeds:
        arrays = {}
        for name in (without, with_):
            with np.load(sweep_dir / name / f"seed{seed}" / "model" / "test_predictions.npz") as predictions:
                arrays[name] = {key: predictions[key] for key in ("protein_ids", "terms", "targets", "propagated_probabilities")}
        protein_ids = arrays[with_]["protein_ids"]
        if not np.array_equal(protein_ids, arrays[without]["protein_ids"]):
            raise ValueError(f"Seed {seed}: test proteins differ between {without} and {with_}")
        terms = [str(term) for term in arrays[with_]["terms"]]
        truth_ia = build_truth_ia_totals(config["train_terms_path"], protein_ids, ontology, ia_weights)
        provenance = sources.reindex(protein_ids).fillna("none (imputed)").to_numpy()
        is_plant = kingdoms.reindex(protein_ids).fillna(0).to_numpy() == 1
        strata = [("provenance", value, provenance == value) for value in ("xref", "alignment_exact", "alignment_proxy", "none (imputed)")]
        strata += [("kingdom", "plants", is_plant), ("kingdom", "non-plants", ~is_plant), ("all", "all test proteins", np.ones(len(protein_ids), bool))]
        for stratum_type, stratum, selected in strata:
            if selected.sum() < 20:
                continue
            fmax = {}
            for name in (without, with_):
                summary, _ = cafa_fmax(
                    arrays[name]["targets"][selected],
                    arrays[name]["propagated_probabilities"][selected],
                    terms,
                    ontology,
                    ia_weights,
                    truth_ia_totals={namespace: values[selected] for namespace, values in truth_ia.items()},
                    propagate=False,
                )
                fmax[name] = summary["cafa_fmax_mean"]
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "stratum": stratum,
                    "seed": seed,
                    "n_proteins": int(selected.sum()),
                    f"fmax_{without}": fmax[without],
                    f"fmax_{with_}": fmax[with_],
                    "delta": fmax[with_] - fmax[without],
                }
            )
    return pd.DataFrame(rows)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(*args, **kwargs) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(*args, **kwargs, facecolor=SURFACE, squeeze=False)
    return figure, axes


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def plot_metric_by_config(runs: pd.DataFrame, path: Path) -> None:
    """Dot plot per metric; configs with and without ESM-2 get separate rows and x scales, since the
    gap between them would otherwise hide the small differences within each group."""
    panels = ["internal_macro_roc_auc", "internal_cafa_fmax_mean", "official_macro_roc_auc_overall", "official_micro_roc_auc_overall"]
    present = [name for name in CONFIGS if name in set(runs["config"])]
    groups = [
        (label, color, [name for name in present if ("esm2" in CONFIGS[name].groups) == uses_esm2][::-1])
        for label, color, uses_esm2 in (("with ESM-2", SERIES[0], True), ("without ESM-2", SERIES[1], False))
    ]
    groups = [group for group in groups if group[2]]
    heights = [len(configs) for _, _, configs in groups]
    figure, axes = plt.subplots(
        len(groups), len(panels), figsize=(4 * len(panels), 0.5 * sum(heights) + 1.2 * len(groups) + 0.6),
        facecolor=SURFACE, squeeze=False, sharey="row", gridspec_kw={"height_ratios": heights},
    )
    for group_row, (label, color, configs) in enumerate(groups):
        for ax, metric in zip(axes[group_row], panels):
            _style(ax)
            for row, name in enumerate(configs):
                values = runs.loc[runs["config"] == name, metric].to_numpy()
                ax.scatter(values, np.full(len(values), row), s=18, color=color, alpha=0.45, linewidths=0, zorder=2)
                if len(values) > 1:
                    ax.plot([values.mean() - values.std(ddof=1), values.mean() + values.std(ddof=1)], [row, row], color=color, linewidth=2, zorder=3)
                ax.scatter([values.mean()], [row], s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=4)
            ax.set_ylim(-0.6, len(configs) - 0.4)
            if group_row == 0:
                ax.set_title(HEADLINE_METRICS[metric], fontsize=10, color=TEXT, loc="left")
        axes[group_row][0].set_yticks(range(len(configs)), configs, color=TEXT)
        axes[group_row][0].set_ylabel(label, fontsize=9, color=TEXT_MUTED)
    figure.suptitle("Metric by feature configuration (dots: seeds; large dot and bar: mean ± sd; x scales differ by row)", fontsize=11, color=TEXT, x=0.01, ha="left")
    figure.tight_layout()
    _save(figure, path)


def plot_delta_heatmap(deltas: pd.DataFrame, path: Path) -> None:
    table = deltas.pivot(index="comparison", columns="metric", values="delta_mean")
    table = table.reindex(index=[label for _, _, label in COMPARISONS if label in table.index], columns=list(HEATMAP_METRICS))
    # Changes involving ESM-2-free models dwarf the rest; scale colour to comparisons where both
    # models have ESM-2 (the decisions that matter) and let the others saturate.
    with_esm2 = [label for with_, without, label in COMPARISONS if label in table.index and all("esm2" in CONFIGS[name].groups for name in (with_, without))]
    scale_rows = table.loc[with_esm2] if with_esm2 else table
    limit = float(np.nanmax(np.abs(scale_rows.to_numpy()))) or 1.0
    saturated = bool(np.nanmax(np.abs(table.to_numpy())) > limit)
    figure, axes = _figure(figsize=(1.0 * table.shape[1] + 3, 0.55 * table.shape[0] + 1.4))
    ax = axes[0][0]
    ax.imshow(table.to_numpy(), cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    for (row, column), value in np.ndenumerate(table.to_numpy()):
        if np.isfinite(value):
            ax.text(column, row, "0.000" if abs(value) < 0.0005 else f"{value:+.3f}", ha="center", va="center", fontsize=8, color=TEXT)
    ax.set_xticks(range(table.shape[1]), [HEATMAP_METRICS[metric] for metric in table.columns], fontsize=8, color=TEXT_MUTED)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=9, color=TEXT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = "Paired change from adding each feature group (mean over seeds; blue = better)"
    if saturated:
        title += f"\ncolour scale ±{limit:.3f}, set by comparisons between ESM-2 models; larger changes saturate"
    ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    _save(figure, path)


def plot_promoter_strata(strata: pd.DataFrame, path: Path) -> None:
    labels = strata[["stratum_type", "stratum", "n_proteins"]].groupby(["stratum_type", "stratum"], sort=False)["n_proteins"].mean()
    figure, axes = _figure(figsize=(7, 0.45 * len(labels) + 1.4))
    ax = axes[0][0]
    _style(ax)
    ax.axvline(0, color=TEXT_MUTED, linewidth=1)
    for row, ((stratum_type, stratum), n_proteins) in enumerate(labels.items()):
        values = strata.loc[(strata["stratum_type"] == stratum_type) & (strata["stratum"] == stratum), "delta"].to_numpy()
        ax.scatter(values, np.full(len(values), row), s=18, color=SERIES[0], alpha=0.45, linewidths=0)
        if len(values) > 1:
            ax.plot([values.mean() - values.std(ddof=1), values.mean() + values.std(ddof=1)], [row, row], color=SERIES[0], linewidth=2)
        ax.scatter([values.mean()], [row], s=64, color=SERIES[0], edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.set_yticks(range(len(labels)), [f"{stratum} (n≈{n:.0f})" for (_, stratum), n in labels.items()], color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Change in internal IA-weighted Fmax from adding promoters (all − esm2_prot_tax)", fontsize=9, color=TEXT_MUTED)
    ax.set_title("Where do promoters help? Test proteins by promoter provenance and kingdom", fontsize=11, color=TEXT, loc="left")
    _save(figure, path)


def plot_response_terms(per_term: pd.DataFrame, stats: pd.DataFrame, path: Path) -> None:
    """ECDF of per-term AUC change from adding promoters: response terms vs other process terms."""
    comparisons = list(dict.fromkeys(per_term["comparison"]))
    figure, axes = _figure(1, len(comparisons), figsize=(6 * len(comparisons), 3.8), sharey=True)
    for ax, comparison in zip(axes[0], comparisons):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.axvline(0, color=TEXT_MUTED, linewidth=1)
        terms = per_term[per_term["comparison"] == comparison]
        for color, label, selected in ((SERIES[1], "response to ... terms", terms["by_name"]), (SERIES[0], "other process terms", ~terms["by_name"])):
            values = np.sort(terms.loc[selected, "delta"].to_numpy())
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=color, linewidth=2, label=f"{label} (n={len(values)})")
        row = stats[(stats["comparison"] == comparison) & (stats["definition"] == "by_name")].iloc[0]
        ax.set_title(
            f"Adding {comparison}\nmean Δ response {row['mean_delta_response']:+.4f} vs other {row['mean_delta_other']:+.4f} "
            f"(p = {row['p_permutation_prevalence_stratified']:.3g})",
            fontsize=10, color=TEXT, loc="left",
        )
        ax.set_xlabel("Change in held-out ROC-AUC per GO term (mean over seeds)", fontsize=9, color=TEXT_MUTED)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0][0].set_ylabel("Share of terms", fontsize=9, color=TEXT_MUTED)
    _save(figure, path)


def plot_training_curves(run_dirs: list[Path], path: Path) -> None:
    figure, axes = _figure(1, 2, figsize=(10, 3.4))
    for index, directory in enumerate(run_dirs):
        history = pd.read_csv(directory / "model" / "history.tsv", sep="\t")
        color, label = SERIES[index % len(SERIES)], directory.name
        axes[0][0].plot(history["epoch"], history["validation_loss"], color=color, linewidth=2, label=label)
        axes[0][1].plot(history["epoch"], history["cafa_fmax_mean"], color=color, linewidth=2, label=label)
    for ax, title in zip(axes[0], ("Validation loss", "Validation IA-weighted Fmax")):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("epoch", fontsize=9, color=TEXT_MUTED)
    axes[0][1].legend(frameon=False, fontsize=8)
    _save(figure, path)


def plot_pr_curves(run_dirs: list[Path], path: Path) -> None:
    figure, axes = _figure(1, 3, figsize=(12, 3.6), sharey=True)
    for index, directory in enumerate(run_dirs):
        curve = pd.read_csv(directory / "model" / "test_nn_cafa_curve.tsv", sep="\t")
        for ax, aspect in zip(axes[0], ASPECT_NAMES):
            rows = curve[curve["aspect"] == aspect]
            ax.plot(rows["weighted_recall"], rows["weighted_precision"], color=SERIES[index % len(SERIES)], linewidth=2, label=directory.name)
    for ax, aspect in zip(axes[0], ASPECT_NAMES):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_title(ASPECT_NAMES[aspect], fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("IA-weighted recall", fontsize=9, color=TEXT_MUTED)
    axes[0][0].set_ylabel("IA-weighted precision", fontsize=9, color=TEXT_MUTED)
    axes[0][2].legend(frameon=False, fontsize=8)
    _save(figure, path)


def write_config_dir(sweep_dir: Path, runs: pd.DataFrame, name: str, out_dir: Path, revision: str) -> None:
    run_dirs = [sweep_dir / name / f"seed{seed}" for seed in runs.loc[runs["config"] == name, "seed"]]
    config_dir = out_dir / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config = CONFIGS[name]
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "config": name,
                "feature_groups": list(config.groups),
                "question": config.question,
                "trainer_flags": [*COMMON_TRAIN_FLAGS, *config.flags],
                "seeds": runs.loc[runs["config"] == name, "seed"].tolist(),
                "git_revision": revision,
            },
            indent=2,
        )
        + "\n"
    )
    columns = ["seed", "n_parameters", "best_epoch"] + [column for column in runs.columns if column.startswith(("internal_", "official_"))]
    runs.loc[runs["config"] == name, columns].round(6).to_csv(config_dir / "metrics.csv", index=False)
    per_term = pd.concat(
        [pd.read_csv(directory / "model" / "test_nn_metrics.tsv", sep="\t").assign(seed=directory.name) for directory in run_dirs]
    )
    per_term.groupby("term")[["n_positive", "roc_auc", "average_precision"]].mean().round(4).to_csv(config_dir / "per_term_auc.csv")
    plot_training_curves(run_dirs, config_dir / "training_curves.png")
    plot_pr_curves(run_dirs, config_dir / "pr_curves.png")


def write_report(sweep_dir: Path, out_dir: Path, release_dir: Path) -> None:
    del release_dir  # official metrics are read from each run's saved outputs
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    revision = _git_revision()
    runs = collect_runs(sweep_dir)
    runs.round(6).to_csv(summary_dir / "runs.csv", index=False)
    summarise(runs).round(6).to_csv(summary_dir / "summary.csv", index=False)
    deltas = paired_deltas(runs)
    deltas.round(6).to_csv(summary_dir / "deltas.csv", index=False)
    official = cafaeval_table(runs)
    if not official.empty:
        official.round(6).to_csv(summary_dir / "cafaeval.csv", index=False)
    plot_metric_by_config(runs, summary_dir / "metric_by_config.png")
    if not deltas.empty:
        plot_delta_heatmap(deltas, summary_dir / "delta_heatmap.png")
    strata = promoter_strata(sweep_dir, runs)
    if not strata.empty:
        strata.round(6).to_csv(summary_dir / "promoter_strata.csv", index=False)
        plot_promoter_strata(strata, summary_dir / "promoter_strata.png")
    obo_path = json.loads(next(sweep_dir.glob("*/seed*/model/run_config.json")).read_text()).get("go_obo_path")
    if obo_path and Path(obo_path).exists():
        response_stats, response_terms = response_term_analysis(sweep_dir, runs, obo_path)
        if not response_stats.empty:
            response_stats.round(6).to_csv(summary_dir / "response_terms.csv", index=False)
            response_terms.round(5).to_csv(summary_dir / "response_term_deltas.csv", index=False)
            plot_response_terms(response_terms, response_stats, summary_dir / "response_terms.png")
    for name in runs["config"].unique():
        write_config_dir(sweep_dir, runs, name, out_dir, revision)
    LOGGER.info("wrote report for %d runs (%s) to %s", len(runs), revision, out_dir)
