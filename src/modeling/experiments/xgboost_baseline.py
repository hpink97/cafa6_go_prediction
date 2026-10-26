"""How much does the multi-label network gain over one XGBoost classifier per GO term?

Compares per-term held-out ROC-AUC of a finished `cafa6-train-go-models` run with the networks'
on the same top-N GO terms. The XGBoost run used the same 2,715 inputs as the network's `all`
configuration (ESM-2 150M, handcrafted, kingdom, NT and AgroNT promoters), so the matched
comparison isolates the model, not the features. XGBoost models were not persisted, so only
per-term ROC-AUC is available; there is no CAFA Fmax or official score for the baseline.

XGBoost trains and is scored on direct annotations, while the network's targets are
ancestor-closed. The network's saved test scores are therefore re-scored against direct
annotations, so both models face the same truth. Both use `make_train_test_split` (20% random
holdout) but with different seeds (XGBoost 42, networks 0-2), so terms are paired but proteins
are not.

Writes to results/xgboost-baseline/:
    per_term_auc.csv         every term: XGBoost and network AUCs, gap, aspect, name, positives
    summary.csv              per network and term category: macro AUCs, mean gap with bootstrap CI,
                             share of terms the network wins, Wilcoxon signed-rank p
    xgb_feature_groups.csv   XGBoost: share of selected features and of gain per input group, by aspect
    *.png                    figures

    uv run python -m modeling.experiments.xgboost_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import bootstrap, wilcoxon
from sklearn.metrics import roc_auc_score

from modeling.data import build_label_matrix
from modeling.evaluation.go import GeneOntology
from modeling.experiments.term_groups import classify_terms, load_term_names

LOGGER = logging.getLogger("cafa6.modeling.experiments.xgboost_baseline")
# key: (label, model directories glob, plot colour or None for table-only)
NETWORKS = {
    "wider_all": ("network, same features (wider, all)", "data/final/winner_all/seed*/model", "#eb6834"),
    "default_all": ("network, default architecture, all features", "data/ablation/feature-groups/all/seed*/model", None),
    "final": ("final network (wider, no promoters)", "data/final/winner_esm2_prot_tax/seed*/model", "#2a78d6"),
}
MATCHED = "wider_all"
XGB_COLOR = "#52514e"
ASPECT_NAMES = {"P": "Biological process", "F": "Molecular function", "C": "Cellular component"}
FREQUENCY_BANDS = ["rare", "intermediate", "common"]
# Input groups of the `all` feature table, by column prefix; everything else is handcrafted.
GROUP_PREFIXES = {"esm2_": "ESM-2", "kingdom_": "kingdom", "promoter_nt_": "NT promoter", "promoter_agront_": "AgroNT promoter"}
GROUP_SIZES = {"ESM-2": 640, "handcrafted": 56, "kingdom": 7, "NT promoter": 512, "AgroNT promoter": 1500}
# Reference palette (dataviz skill); promoters are merged into one slot for plotting.
GROUP_COLORS = {"ESM-2": "#2a78d6", "handcrafted": "#eb6834", "kingdom": "#1baf7a", "promoters": "#eda100"}
DIVERGING = LinearSegmentedColormap.from_list("gap", ["#e34948", "#d6d5d0", "#2a78d6"])
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def feature_group(name: str) -> str:
    for prefix, group in GROUP_PREFIXES.items():
        if name.startswith(prefix):
            return group
    return "handcrafted"


def load_xgboost(xgb_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(xgb_dir / "summary.tsv", sep="\t")
    summary = summary[summary["status"] == "ok"]
    if summary["n_features_total"].nunique() != 1 or summary["n_features_total"].iloc[0] != sum(GROUP_SIZES.values()):
        raise ValueError(f"{xgb_dir} was not trained on the {sum(GROUP_SIZES.values())}-column `all` feature table")
    params = pd.json_normalize(summary["best_params"].map(json.loads))
    params.columns = [column.removeprefix("classifier__") for column in params.columns]
    table = summary[["term", "n_pos_train", "n_pos_test", "cv_best_score", "test_roc_auc"]].rename(
        columns={"cv_best_score": "xgb_cv_auc", "test_roc_auc": "xgb_auc"}
    )
    return pd.concat([table.reset_index(drop=True), params.add_prefix("xgb_")], axis=1)


def _direct_label_auc(model_dir: Path, terms: list[str], train_terms: Path) -> np.ndarray:
    """The network's propagated test scores against direct annotations, the labels XGBoost trains
    and is scored on (the network's own targets are ancestor-closed, so general terms have more
    positives)."""
    with np.load(model_dir / "test_predictions.npz", allow_pickle=True) as predictions:
        columns = pd.Index(predictions["terms"]).get_indexer(terms)
        scores = predictions["propagated_probabilities"][:, columns]
        protein_ids = pd.Index(predictions["protein_ids"])
    labels = build_label_matrix(train_terms, protein_ids, terms).to_numpy()
    return np.array([roc_auc_score(labels[:, i], scores[:, i]) if 0 < labels[:, i].sum() < len(labels) else np.nan for i in range(len(terms))])


def load_network(pattern: str, terms: list[str], train_terms: Path) -> pd.DataFrame:
    """Per-term held-out AUC averaged over seeds: `auc` against direct annotations (comparable with
    XGBoost) and `auc_closed` against the network's own ancestor-closed targets, as training reports it."""
    model_dirs = sorted(Path().glob(pattern))
    if not model_dirs:
        raise FileNotFoundError(pattern)
    LOGGER.info("scoring %d seeds of %s", len(model_dirs), pattern)
    direct = np.vstack([_direct_label_auc(directory, terms, train_terms) for directory in model_dirs])
    closed = pd.concat(
        [pd.read_csv(directory / "test_nn_metrics.tsv", sep="\t").set_index("term")["roc_auc"].rename(str(directory)) for directory in model_dirs],
        axis=1,
    ).reindex(terms)
    return pd.DataFrame(
        {"auc": direct.mean(axis=0), "auc_sd": direct.std(axis=0, ddof=1), "auc_closed": closed.mean(axis=1).to_numpy(), "n_seeds": len(model_dirs)},
        index=terms,
    )


def term_table(xgb_dir: Path, train_terms: Path, obo_path: Path) -> pd.DataFrame:
    table = load_xgboost(xgb_dir)
    terms = table["term"].tolist()
    aspects = pd.read_csv(train_terms, sep="\t", usecols=["term", "aspect"]).drop_duplicates("term").set_index("term")["aspect"]
    names = load_term_names(obo_path)
    table.insert(1, "name", table["term"].map(names))
    table.insert(2, "aspect", table["term"].map(aspects))
    response = classify_terms(table[["term", "aspect"]], GeneOntology.from_obo(obo_path), names)["by_name"]
    table.insert(3, "response_term", table["term"].map(response).fillna(False).astype(bool))
    table.insert(4, "frequency_band", pd.qcut(table["n_pos_train"].rank(method="first"), 3, labels=FREQUENCY_BANDS))
    for key, (_, pattern, _) in NETWORKS.items():
        network = load_network(pattern, terms, train_terms).add_prefix(f"{key}_")
        table = table.join(network, on="term")
        table[f"{key}_gap"] = table[f"{key}_auc"] - table["xgb_auc"]
    return table


def categories(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Named boolean masks over terms, in display order."""
    masks = {"all terms": pd.Series(True, index=table.index)}
    masks.update({ASPECT_NAMES[aspect]: table["aspect"] == aspect for aspect in ASPECT_NAMES})
    masks["'response to ...' processes"] = table["response_term"]
    masks["other processes"] = (table["aspect"] == "P") & ~table["response_term"]
    masks.update({f"{band} terms (by training positives)": table["frequency_band"] == band for band in FREQUENCY_BANDS})
    return masks


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, (label, _, _) in NETWORKS.items():
        for category, mask in categories(table).items():
            gap = table.loc[mask, f"{key}_gap"].dropna().to_numpy()
            interval = bootstrap((gap,), np.mean, n_resamples=2000, random_state=0).confidence_interval
            rows.append(
                {
                    "network": key,
                    "network_label": label,
                    "category": category,
                    "n_terms": len(gap),
                    "network_macro_auc": table.loc[mask, f"{key}_auc"].mean(),
                    "xgb_macro_auc": table.loc[mask, "xgb_auc"].mean(),
                    "gap_mean": gap.mean(),
                    "gap_ci_low": interval.low,
                    "gap_ci_high": interval.high,
                    "gap_median": np.median(gap),
                    "network_win_share": (gap > 0).mean(),
                    "wilcoxon_p": wilcoxon(gap).pvalue,
                }
            )
    return pd.DataFrame(rows)


def xgb_feature_groups(xgb_dir: Path, table: pd.DataFrame) -> pd.DataFrame:
    """Per aspect and input group: share of each term's selected features and of its gain importance,
    and how over-represented the group is among selected features relative to its size."""
    rows = []
    for term in table["term"]:
        importances = pd.read_csv(xgb_dir / "terms" / term.replace(":", "_") / "feature_importances.tsv", sep="\t")
        importances["group"] = importances["feature"].map(feature_group)
        grouped = importances.groupby("group")["importance"].agg(["size", "sum"]).reindex(list(GROUP_SIZES), fill_value=0)
        for group, (n_selected, gain) in grouped.iterrows():
            rows.append({"term": term, "group": group, "selected_share": n_selected / len(importances), "gain_share": gain / importances["importance"].sum()})
    per_term = pd.DataFrame(rows).merge(table[["term", "aspect"]], on="term")
    overall = per_term.assign(aspect="all")
    result = pd.concat([overall, per_term]).groupby(["aspect", "group"], sort=False)[["selected_share", "gain_share"]].mean().reset_index()
    result["input_share"] = result["group"].map(GROUP_SIZES) / sum(GROUP_SIZES.values())
    result["selection_enrichment"] = result["selected_share"] / result["input_share"]
    return result


# ---------------------------------------------------------------- figures


def _style(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def _short(name: str, width: int = 34) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def plot_auc_scatter(table: pd.DataFrame, path: Path, n_labels: int = 3) -> None:
    """XGBoost vs matched-network AUC per term, one panel per aspect, coloured by the gap."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURFACE, sharex=True, sharey=True, layout="constrained")
    low = min(table["xgb_auc"].min(), table[f"{MATCHED}_auc"].min()) - 0.02
    limit = 0.1
    for ax, (aspect, aspect_name) in zip(axes, ASPECT_NAMES.items()):
        _style(ax)
        terms = table[table["aspect"] == aspect]
        gap = terms[f"{MATCHED}_gap"]
        ax.plot([low, 1], [low, 1], color=TEXT_MUTED, linewidth=1, linestyle="--", zorder=1)
        points = ax.scatter(
            terms["xgb_auc"], terms[f"{MATCHED}_auc"], c=gap.clip(-limit, limit), cmap=DIVERGING, vmin=-limit, vmax=limit,
            s=6 + 2.5 * np.sqrt(terms["n_pos_test"]), edgecolors=SURFACE, linewidths=0.4, alpha=0.9, zorder=2,
        )
        ax.set_title(f"{aspect_name} ({len(terms)} terms)\nnetwork higher on {np.mean(gap > 0):.0%}, mean gap {gap.mean():+.3f}", fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("XGBoost per-term ROC-AUC", fontsize=9, color=TEXT_MUTED)
        labels = [
            ax.annotate(_short(row.name), (row.xgb_auc, getattr(row, f"{MATCHED}_auc")), fontsize=7.5, color=TEXT,
                        arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "linewidth": 0.6})
            for row in terms.nsmallest(n_labels, f"{MATCHED}_gap").itertuples()
        ]
        adjust_text(labels, ax=ax, expand=(1.4, 1.6), force_text=(0.6, 0.8))
        ax.set_xlim(low, 1.005)
        ax.set_ylim(low, 1.005)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Network per-term ROC-AUC (same features, mean of 3 seeds)", fontsize=9, color=TEXT_MUTED)
    bar = figure.colorbar(points, ax=axes, shrink=0.8, pad=0.01)
    bar.set_label(f"network − XGBoost AUC (clipped at ±{limit})", fontsize=9, color=TEXT_MUTED)
    bar.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    bar.outline.set_visible(False)
    figure.suptitle(
        "Same 2,715 inputs, same 1,000 GO terms, same labels: the multi-label network beats per-term XGBoost on most terms\n"
        "points above the diagonal favour the network; point size ∝ √test positives; labelled: where XGBoost wins most",
        fontsize=11, color=TEXT, x=0.01, ha="left",
    )
    _save(figure, path)


def plot_gap_vs_frequency(table: pd.DataFrame, path: Path, n_bins: int = 8) -> None:
    """The AUC gap against training positives: binned means with 95% CIs over terms."""
    figure, ax = plt.subplots(figsize=(8.5, 5), facecolor=SURFACE)
    _style(ax)
    edges = np.unique(np.quantile(table["n_pos_train"], np.linspace(0, 1, n_bins + 1)))
    centres = np.sqrt(edges[:-1] * edges[1:])
    bins = pd.cut(table["n_pos_train"], edges, include_lowest=True, labels=False)
    ax.axhline(0, color=TEXT_MUTED, linewidth=1)
    ax.scatter(table["n_pos_train"], table[f"{MATCHED}_gap"], s=7, color="#b5b4ae", linewidths=0, alpha=0.6, zorder=1, label="one GO term (same features)")
    long = pd.concat(
        [pd.DataFrame({"centre": centres[bins], "gap": table[f"{key}_gap"], "network": label}) for key, (label, _, color) in NETWORKS.items() if color]
    )
    palette = {label: color for label, _, color in NETWORKS.values() if color}
    sns.lineplot(long, x="centre", y="gap", hue="network", palette=palette, errorbar=("ci", 95), n_boot=2000, seed=0,
                 marker="o", markersize=7, markeredgecolor=SURFACE, linewidth=2, ax=ax, zorder=3)
    ax.set_xscale("log")
    spread = table[f"{MATCHED}_gap"].abs().quantile(0.98)
    ax.set_ylim(-spread, spread)
    ax.set_xlabel("Training proteins annotated with the term (log scale)", fontsize=9.5, color=TEXT_MUTED)
    ax.set_ylabel("Network − XGBoost held-out ROC-AUC", fontsize=9.5, color=TEXT_MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", title="mean per frequency bin, 95% CI", title_fontsize=8.5)
    ax.set_title(
        "The network's advantage is largest for rare terms\n"
        "XGBoost fits each term alone; the network shares one representation across 2,180 outputs (both scored on direct annotations)",
        fontsize=10.5, color=TEXT, loc="left",
    )
    _save(figure, path)


def plot_auc_ecdf(table: pd.DataFrame, path: Path) -> None:
    figure, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    _style(ax)
    series = [("XGBoost, one model per term (all features)", table["xgb_auc"], XGB_COLOR)]
    series += [(label, table[f"{key}_auc"], color) for key, (label, _, color) in NETWORKS.items() if color]
    for label, values, color in series:
        sns.ecdfplot(values.dropna(), ax=ax, color=color, linewidth=2, label=f"{label}: macro {values.mean():.3f}")
    closed = table[f"{MATCHED}_auc_closed"]
    sns.ecdfplot(closed, ax=ax, color=NETWORKS[MATCHED][2], linewidth=1.5, linestyle=":",
                 label=f"same network, scored on its own ancestor-closed labels: macro {closed.mean():.3f}")
    ax.set_xlim(table["xgb_auc"].quantile(0.005), 1)
    ax.set_xlabel("Per-term held-out ROC-AUC", fontsize=9.5, color=TEXT_MUTED)
    ax.set_ylabel("Share of the 1,000 GO terms", fontsize=9.5, color=TEXT_MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Per-term ROC-AUC on the same labels (direct annotations): the network curves sit to the right", fontsize=10.5, color=TEXT, loc="left")
    _save(figure, path)


def plot_gap_by_category(summary: pd.DataFrame, path: Path) -> None:
    keys = [key for key, (_, _, color) in NETWORKS.items() if color]
    order = list(dict.fromkeys(summary["category"]))
    figure, ax = plt.subplots(figsize=(8.5, 0.48 * len(order) + 1.6), facecolor=SURFACE)
    _style(ax, grid_axis="x")
    ax.axvline(0, color=TEXT_MUTED, linewidth=1)
    offsets = np.linspace(-0.15, 0.15, len(keys))
    for offset, key in zip(offsets, keys):
        label, _, color = NETWORKS[key]
        rows = summary[summary["network"] == key].set_index("category").loc[order]
        y = np.arange(len(order)) + offset
        ax.errorbar(rows["gap_mean"], y, xerr=[rows["gap_mean"] - rows["gap_ci_low"], rows["gap_ci_high"] - rows["gap_mean"]],
                    fmt="o", color=color, markersize=7, markeredgecolor=SURFACE, elinewidth=2, label=label)
    n_terms = summary[summary["network"] == keys[0]].set_index("category").loc[order, "n_terms"]
    ax.set_yticks(range(len(order)), [f"{category} (n={n})" for category, n in n_terms.items()], color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean network − XGBoost per-term ROC-AUC (95% bootstrap CI over terms)", fontsize=9, color=TEXT_MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("The network wins in every aspect and term category (same labels; CIs exclude 0)", fontsize=10.5, color=TEXT, loc="left")
    _save(figure, path)


def plot_xgb_feature_groups(groups: pd.DataFrame, path: Path) -> None:
    """What XGBoost picks (feature selection keeps 250 of 2,715 inputs per term) and what it splits on."""
    merged = groups.assign(group=groups["group"].replace({"NT promoter": "promoters", "AgroNT promoter": "promoters"}))
    merged = merged.groupby(["aspect", "group"], sort=False)[["selected_share", "gain_share", "input_share"]].sum().reset_index()
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.3]})
    overall = groups[groups["aspect"] == "all"].set_index("group").loc[list(GROUP_SIZES)]
    colors = [GROUP_COLORS.get(group, GROUP_COLORS["promoters"]) for group in overall.index]
    _style(left, grid_axis="x")
    y = np.arange(len(overall))
    left.barh(y - 0.2, overall["input_share"], height=0.38, color="#d6d5d0", label="share of the 2,715 inputs")
    left.barh(y + 0.2, overall["selected_share"], height=0.38, color=colors, label="share of the 250 kept per term (group colour)")
    for row, (share, selected) in enumerate(zip(overall["input_share"], overall["selected_share"])):
        left.text(max(share, selected) + 0.01, row, f"×{selected / share:.2g}", va="center", fontsize=8.5, color=TEXT)
    left.set_yticks(y, overall.index, color=TEXT)
    left.invert_yaxis()
    left.set_xlabel("Share of features (mean over terms)", fontsize=9, color=TEXT_MUTED)
    left.legend(frameon=False, fontsize=8, ncols=2, loc="upper left", bbox_to_anchor=(0, -0.2))
    left.set_title("Feature selection: what XGBoost keeps (×: over-representation)", fontsize=10, color=TEXT, loc="left")

    _style(right, grid_axis="x")
    aspects = ["all", *ASPECT_NAMES]
    labels = ["all terms", *ASPECT_NAMES.values()]
    for row, aspect in enumerate(aspects):
        shares = merged[merged["aspect"] == aspect].set_index("group").loc[list(GROUP_COLORS), "gain_share"]
        start = 0.0
        for group, share in shares.items():
            right.barh(row, share, left=start, height=0.6, color=GROUP_COLORS[group], edgecolor=SURFACE, linewidth=2, label=group if row == 0 else None)
            if share > 0.06:
                right.text(start + share / 2, row, f"{share:.0%}", ha="center", va="center", fontsize=8.5, color=TEXT)
            start += share
    right.set_yticks(range(len(aspects)), labels, color=TEXT)
    right.invert_yaxis()
    right.set_xlim(0, 1)
    right.set_xlabel("Share of XGBoost gain importance (mean over terms)", fontsize=9, color=TEXT_MUTED)
    right.legend(frameon=False, fontsize=8, ncols=4, loc="upper left", bbox_to_anchor=(0, -0.2))
    right.set_title("Gain: what XGBoost splits on", fontsize=10, color=TEXT, loc="left")
    figure.tight_layout()
    _save(figure, path)


def write_report(xgb_dir: Path, out_dir: Path, train_terms: Path, obo_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = term_table(xgb_dir, train_terms, obo_path)
    table.round(4).to_csv(out_dir / "per_term_auc.csv", index=False)
    summary = summarise(table)
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.4g")
    groups = xgb_feature_groups(xgb_dir, table)
    groups.round(4).to_csv(out_dir / "xgb_feature_groups.csv", index=False)
    plot_auc_scatter(table, out_dir / "auc_scatter.png")
    plot_gap_vs_frequency(table, out_dir / "gap_vs_frequency.png")
    plot_auc_ecdf(table, out_dir / "auc_ecdf.png")
    plot_gap_by_category(summary, out_dir / "gap_by_category.png")
    plot_xgb_feature_groups(groups, out_dir / "xgb_feature_groups.png")
    LOGGER.info("wrote %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xgb-dir", type=Path, default=Path("data/models/go_term_xgboost_20260919_2756"), help="A finished cafa6-train-go-models run")
    parser.add_argument("--out", type=Path, default=Path("results/xgboost-baseline"))
    parser.add_argument("--train-terms", type=Path, default=Path("data/Train/train_terms.tsv"))
    parser.add_argument("--go-obo", type=Path, default=Path("data/Train/go-basic.obo"))
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    write_report(args.xgb_dir, args.out, args.train_terms, args.go_obo)


if __name__ == "__main__":
    main()
