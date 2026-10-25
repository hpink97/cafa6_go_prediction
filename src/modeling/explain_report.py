"""Tables and figures for `cafa6-explain` (see modeling.explain).

Per-seed SHAP results are cached under --cache-dir, so figures can be re-drawn without recomputing.
Writes to results/explainability/:
    term_group_attribution.csv      every term: ESM-2 / handcrafted / kingdom share of |SHAP| (mean, sd over seeds)
    permutation_importance.csv      every term: AUC drop when each group is shuffled (mean, sd over seeds)
    term_top_features.csv           every term: its 10 most important input features
    deep_dive_features.csv          deep-dive terms x every input feature: mean |SHAP|, mean signed SHAP, direction
    esm2_dimension_importance.csv   each ESM-2 dimension: importance overall and per aspect, rank, seed agreement
    esm2_dimension_feature_correlation.csv  top ESM-2 dimensions x handcrafted features (Spearman)
    *.png, deep_dives/*.png         figures
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from adjustText import adjust_text
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from modeling.explain import (
    DEEP_DIVE_TERMS,
    GROUPS,
    deep_dive,
    dimension_feature_correlation,
    feature_groups,
    global_attribution,
    load_explain_data,
    permutation_importance,
    term_metadata,
)

LOGGER = logging.getLogger("cafa6.modeling.explain_report")
# Reference palette (dataviz skill): categorical slots 1-3 and neutral inks.
GROUP_COLORS = {"esm2": "#2a78d6", "handcrafted": "#eb6834", "kingdom": "#1baf7a"}
GROUP_LABELS = {"esm2": "ESM-2", "handcrafted": "handcrafted", "kingdom": "kingdom"}
ASPECT_COLORS = {"P": "#2a78d6", "F": "#eb6834", "C": "#1baf7a"}
ASPECT_NAMES = {"P": "Biological process", "F": "Molecular function", "C": "Cellular component"}
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TOP_DIMENSIONS = 40
FEATURE_ABBREVIATIONS = {
    "bipartite nuclear localization signal": "bipartite NLS",
    "protein kinase atp binding": "kinase ATP-binding",
    "st kinase active site": "Ser/Thr kinase active site",
    "ploop ntp binding": "P-loop NTP-binding",
    "transmembrane helix count": "TM helix count",
    "gravy": "GRAVY (hydropathy)",
}


def pretty_feature(name: str) -> str:
    """Readable label for a handcrafted feature or ESM-2 dimension column."""
    if name.startswith("esm2_640_"):
        return f"dim {name.removeprefix('esm2_640_')}"
    if name.startswith("aa_fraction_"):
        return f"{name.removeprefix('aa_fraction_')} content"
    label = name.removesuffix("_present").replace("_", " ")
    for prefix, nicer in (("prosite ", "PROSITE: "), ("motif ", "motif: ")):
        if label.startswith(prefix):
            label = nicer + label.removeprefix(prefix)
    for long, short in FEATURE_ABBREVIATIONS.items():
        label = label.replace(long, short)
    return label


def _style(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(figure: plt.Figure, path: Path, dpi: int = 150, palette: bool = False) -> None:
    """Save a figure; `palette` re-encodes it with a 256-colour palette, which keeps dense scatter
    maps sharp but several times smaller (the repo keeps every file under 1 MB)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    if palette:
        from PIL import Image

        with Image.open(path) as image:
            image.convert("RGB").quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(path, optimize=True)
    plt.close(figure)


# ---------------------------------------------------------------- per-seed computation (cached)


def explain_seed(model_dir: Path, cache_dir: Path, max_positives: int, nsamples: int) -> dict:
    """Global SHAP, permutation importance and deep dives for one seed; reuses cached results."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = load_explain_data(model_dir)
    shares_path, importance_path = cache_dir / "group_shares.csv", cache_dir / "feature_importance.npy"
    if not shares_path.exists():
        LOGGER.info("global SHAP for %s", model_dir)
        shares, importance = global_attribution(data, max_positives=max_positives, nsamples=nsamples)
        np.save(importance_path, importance)
        shares.to_csv(shares_path, index=False)
    permutation_path = cache_dir / "permutation.csv"
    if not permutation_path.exists():
        LOGGER.info("permutation importance for %s", model_dir)
        permutation_importance(data).to_csv(permutation_path, index=False)
    deep_dives = {}
    for term in DEEP_DIVE_TERMS:
        path = cache_dir / f"deep_dive_all_{term.replace(':', '_')}.npz"
        if not path.exists():
            explanation = deep_dive(data, term)
            np.savez_compressed(path, values=explanation.values, data=explanation.data, feature_names=np.asarray(explanation.feature_names))
        with np.load(path) as saved:
            deep_dives[term] = shap.Explanation(values=saved["values"], data=saved["data"], feature_names=[str(name) for name in saved["feature_names"]])
    return {
        "data": data,
        "shares": pd.read_csv(shares_path),
        "importance": np.load(importance_path),
        "permutation": pd.read_csv(permutation_path),
        "deep_dives": deep_dives,
    }


# ---------------------------------------------------------------- aggregation


def aggregate_over_seeds(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    stacked = pd.concat(frames)
    grouped = stacked.groupby("term")[columns]
    return grouped.mean().join(grouped.std().add_suffix("_sd")).reset_index()


def dimension_importance(importance: list[np.ndarray], feature_names: list[str], meta: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Each ESM-2 dimension's share of a term's ESM-2 attribution, averaged over terms (overall and
    per aspect), so frequent terms with large logits don't dominate."""
    names = np.asarray(feature_names)
    esm = np.char.startswith(names.astype(str), "esm2_")
    per_seed = []
    for matrix in importance:
        block = matrix[:, esm]
        per_seed.append(block / np.nansum(block, axis=1, keepdims=True))
    shares = np.nanmean(per_seed, axis=0)
    table = pd.DataFrame({"dimension": names[esm], "importance": np.nanmean(shares, axis=0)})
    for aspect in ASPECT_NAMES:
        table[f"importance_{aspect}"] = np.nanmean(shares[(meta["aspect"] == aspect).to_numpy()], axis=0)
    table["rank"] = table["importance"].rank(ascending=False).astype(int)
    seed_scores = [np.nanmean(seed, axis=0) for seed in per_seed]
    table["importance_seed_sd"] = np.std(seed_scores, axis=0)
    agreement = [spearmanr(seed_scores[i], seed_scores[j]).statistic for i in range(len(seed_scores)) for j in range(i + 1, len(seed_scores))]
    LOGGER.info("ESM-2 dimension importance, Spearman between seeds: %s", np.round(agreement, 3))
    table.attrs["seed_spearman"] = agreement
    return table.sort_values("rank"), list(table.sort_values("rank")["dimension"][:TOP_DIMENSIONS])


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    cumulative = np.cumsum(values)
    return float((len(values) + 1 - 2 * (cumulative / cumulative[-1]).sum()) / len(values))


# ---------------------------------------------------------------- figures


def plot_group_shares_by_category(shares: pd.DataFrame, meta: pd.DataFrame, path: Path) -> None:
    merged = shares.merge(meta, on="term")
    categories = {"All terms": merged}
    for aspect, name in ASPECT_NAMES.items():
        categories[name] = merged[merged["aspect"] == aspect]
    process = merged[merged["aspect"] == "P"]
    categories['"response to" processes'] = process[process["response_term"]]
    categories["other processes"] = process[~process["response_term"]]
    for band in ("general (common)", "intermediate", "specific (rare)"):
        categories[f"{band} terms"] = merged[merged["frequency_band"] == band]
    labels = list(categories)[::-1]
    figure, ax = plt.subplots(figsize=(9, 0.42 * len(labels) + 1.4), facecolor=SURFACE)
    _style(ax)
    left = np.zeros(len(labels))
    for group in GROUPS:
        values = np.asarray([categories[label][group].mean() for label in labels])
        ax.barh(labels, values, left=left, color=GROUP_COLORS[group], edgecolor=SURFACE, linewidth=2, label=GROUP_LABELS[group])
        for y, (start, value) in enumerate(zip(left, values)):
            if value > 0.035:
                ax.text(start + value / 2, y, f"{value:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += values
    ax.set_yticks(range(len(labels)), [f"{label} (n={len(categories[label])})" for label in labels], color=TEXT)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Where does each GO term's prediction come from? Mean share of |SHAP| per feature group", fontsize=10.5, color=TEXT, loc="left", pad=28)
    _save(figure, path)


def plot_deep_dive_group_shares(shares: pd.DataFrame, meta: pd.DataFrame, path: Path) -> None:
    rows = shares.merge(meta, on="term").set_index("term").loc[list(DEEP_DIVE_TERMS)[::-1]]
    labels = [f"{rows.loc[term, 'name']} [{DEEP_DIVE_TERMS[term]}]" for term in rows.index]
    figure, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.2), facecolor=SURFACE)
    _style(ax)
    left = np.zeros(len(rows))
    for group in GROUPS:
        values = rows[group].to_numpy()
        ax.barh(labels, values, left=left, color=GROUP_COLORS[group], edgecolor=SURFACE, linewidth=2, label=GROUP_LABELS[group])
        for y, (start, value) in enumerate(zip(left, values)):
            if value > 0.035:
                ax.text(start + value / 2, y, f"{value:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += values
    ax.set_yticks(range(len(labels)), labels, color=TEXT, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Deep-dive GO terms: share of |SHAP| per feature group (mean over 3 seeds)", fontsize=10.5, color=TEXT, loc="left", pad=28)
    _save(figure, path)


def plot_deep_dive_beeswarms(explanations: dict[str, shap.Explanation], meta: pd.DataFrame, out_dir: Path) -> None:
    """Beeswarm of the top individual features for each deep-dive term: ESM-2 dimensions and
    handcrafted features ranked together."""
    names = meta.set_index("term")["name"]
    for term, explanation in explanations.items():
        labels = [f"ESM-2 {pretty_feature(name)}" if name.startswith("esm2_") else pretty_feature(name) for name in explanation.feature_names]
        explanation = shap.Explanation(values=explanation.values, data=explanation.data, feature_names=labels)
        plt.figure(facecolor=SURFACE)
        shap.plots.beeswarm(explanation, max_display=13, show=False, plot_size=(8.5, 6))
        plt.title(f"{names[term]} ({term}): top features, SHAP on the logit", fontsize=10, loc="left")
        figure = plt.gcf()
        _save(figure, out_dir / f"{term.replace(':', '_')}_beeswarm.png")


def plot_deep_dive_overview(features: pd.DataFrame, meta: pd.DataFrame, path: Path) -> None:
    """Small multiples: top 6 handcrafted features per deep-dive term, bar = mean |SHAP|, colour = direction."""
    names = meta.set_index("term")["name"]
    terms = list(DEEP_DIVE_TERMS)
    columns = 3
    figure, axes = plt.subplots(int(np.ceil(len(terms) / columns)), columns, figsize=(16, 2.9 * np.ceil(len(terms) / columns)), facecolor=SURFACE)
    for ax, term in zip(axes.flat, terms):
        _style(ax)
        rows = features[(features["term"] == term) & (features["group"] == "handcrafted")].nlargest(6, "mean_abs_shap")[::-1]
        colors = np.where(rows["direction"] > 0, "#e34948", "#2a78d6")
        ax.barh([pretty_feature(name) for name in rows["feature"]], rows["mean_abs_shap"], color=colors, edgecolor=SURFACE)
        ax.tick_params(axis="y", labelsize=7.5, colors=TEXT)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        title = names[term] if len(names[term]) < 48 else names[term][:46] + "…"
        ax.set_title(f"{title} ({DEEP_DIVE_TERMS[term]})", fontsize=9, color=TEXT, loc="left")
    for ax in list(axes.flat)[len(terms):]:
        ax.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#e34948"), plt.Rectangle((0, 0), 1, 1, color="#2a78d6")]
    figure.legend(handles, ["higher value → more likely", "higher value → less likely"], frameon=False, loc="lower right", fontsize=9)
    figure.suptitle("Top handcrafted features per deep-dive term (mean |SHAP|; colour = direction of effect)", fontsize=11, color=TEXT, x=0.01, ha="left")
    figure.tight_layout()
    _save(figure, path)


def plot_permutation(permutation: pd.DataFrame, meta: pd.DataFrame, path: Path) -> None:
    merged = permutation.merge(meta, on="term")
    figure, axes = plt.subplots(1, len(GROUPS), figsize=(13, 3.8), facecolor=SURFACE)
    for ax, group in zip(axes, GROUPS):
        _style(ax, grid_axis="y")
        data = [merged.loc[merged["aspect"] == aspect, f"{group}_auc_drop"].dropna() for aspect in ASPECT_NAMES]
        parts = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, medianprops={"color": TEXT})
        for patch in parts["boxes"]:
            patch.set_facecolor(GROUP_COLORS[group])
            patch.set_alpha(0.75)
        ax.axhline(0, color=TEXT_MUTED, linewidth=1)
        ax.set_xticks(range(1, 4), ["BP", "MF", "CC"])
        ax.set_title(f"Shuffle {GROUP_LABELS[group]}: median drop {np.nanmedian(merged[f'{group}_auc_drop']):.3f}", fontsize=10, color=TEXT, loc="left")
    axes[0].set_ylabel("Per-term drop in held-out ROC-AUC", fontsize=9, color=TEXT_MUTED)
    figure.suptitle("Group permutation importance (each feature group shuffled across test proteins; boxes = GO terms)", fontsize=11, color=TEXT, x=0.01, ha="left")
    figure.tight_layout()
    _save(figure, path)


def plot_dimension_concentration(dimensions: pd.DataFrame, path: Path) -> dict[str, float]:
    figure, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    _style(ax, grid_axis="both")
    stats = {}
    for column, label, color in (("importance", "all terms", TEXT), *((f"importance_{a}", ASPECT_NAMES[a], ASPECT_COLORS[a]) for a in ASPECT_NAMES)):
        values = np.sort(dimensions[column].to_numpy())[::-1]
        cumulative = np.cumsum(values) / values.sum()
        ax.plot(np.arange(1, len(values) + 1), cumulative, color=color, linewidth=2 if column == "importance" else 1.5, label=f"{label} (Gini {gini(values):.2f})")
        stats[label] = {"gini": gini(values), "top_10_share": float(cumulative[9]), "top_64_share": float(cumulative[63])}
    ranks = np.arange(1, len(dimensions) + 1)
    ax.plot(ranks, ranks / len(dimensions), color=TEXT_MUTED, linestyle="--", linewidth=1, label="every dimension equal")
    ax.set_xscale("log")
    ax.set_xlabel("ESM-2 dimensions, ranked by importance (log scale)", fontsize=9, color=TEXT_MUTED)
    ax.set_ylabel("Cumulative share of ESM-2 attribution", fontsize=9, color=TEXT_MUTED)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Do a few ESM-2 dimensions do most of the work?", fontsize=10.5, color=TEXT, loc="left")
    _save(figure, path)
    return stats


def plot_dimension_feature_correlation(correlation: pd.DataFrame, path: Path) -> None:
    frame = correlation.copy()
    best = frame.abs().idxmax(axis=1)
    frame.index = [f"{pretty_feature(d)} ↔ {pretty_feature(best[d])} ({frame.loc[d, best[d]]:+.2f})" for d in frame.index]
    frame.columns = [pretty_feature(column) for column in frame.columns]
    grid = sns.clustermap(
        frame.fillna(0), cmap="RdBu_r", center=0, vmin=-1, vmax=1, figsize=(15, 11), dendrogram_ratio=(0.08, 0.08),
        cbar_pos=(0.01, 0.84, 0.015, 0.1), xticklabels=True, yticklabels=True, method="average",
    )
    grid.ax_heatmap.tick_params(axis="x", labelsize=7, rotation=90)
    grid.ax_heatmap.tick_params(axis="y", labelsize=7)
    grid.figure.suptitle("What do the most important ESM-2 dimensions encode? Spearman ρ with handcrafted features (training proteins)", fontsize=11, x=0.02, ha="left", y=1.01)
    grid.figure.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(grid.figure)


ANNOTATION_COLOR, SEQUENCE_FEATURE_COLOR, BACKGROUND_POINT = "#eb6834", "#2a78d6", "#d6d5d1"


def protein_umap(data, cache_path: Path) -> np.ndarray:
    """2-D UMAP of the test proteins' ESM-2 embeddings (cached: shared by both protein maps)."""
    if cache_path.exists():
        return np.load(cache_path)
    import umap

    embedding = data.raw_test.to_numpy(dtype=np.float32)[:, data.groups == "esm2"]
    coords = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="cosine", random_state=0).fit_transform(embedding)
    np.save(cache_path, coords)
    return coords


def _map_panel(figure: plt.Figure, ax: plt.Axes, coords: np.ndarray, values: np.ndarray, kind: str, title: str, note: str = "") -> None:
    """One UMAP panel. kind: 'annotation' / 'feature' (binary, highlighted), 'continuous', 'categorical'."""
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if kind == "categorical":
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
        names, counts = np.unique(values, return_counts=True)
        for color, name in zip(palette, names[np.argsort(-counts)]):
            index = np.flatnonzero(values == name)
            ax.scatter(*coords[index].T, s=1, color=color, label=f"{name} ({len(index):,})", rasterized=True)
        ax.legend(frameon=False, fontsize=7, markerscale=6, loc="center left", bbox_to_anchor=(0.98, 0.5))
    elif kind in ("annotation", "feature"):
        positive = np.asarray(values) > 0
        color = ANNOTATION_COLOR if kind == "annotation" else SEQUENCE_FEATURE_COLOR
        ax.scatter(*coords[~positive].T, s=1, color=BACKGROUND_POINT, rasterized=True)
        ax.scatter(*coords[positive].T, s=1.6, color=color, rasterized=True)
        note = note or f"{positive.sum():,} proteins"
    else:
        finite = np.isfinite(values)
        low, high = np.nanpercentile(values, [2, 98])
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=float(np.nanmedian(values)), vmin=low, vmax=high) if low < np.nanmedian(values) < high else None
        order = np.argsort(np.abs(values - np.nanmedian(values)))  # extreme values drawn on top
        order = order[finite[order]]
        scatter = ax.scatter(*coords[order].T, s=1, c=np.clip(values[order], low, high), cmap="RdBu_r" if norm else "viridis", norm=norm, rasterized=True)
        figure.colorbar(scatter, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(labelsize=7)
    # A few far-out UMAP points would otherwise shrink the map to a corner of the panel.
    (x_low, y_low), (x_high, y_high) = np.percentile(coords, [0.5, 99.5], axis=0)
    pad_x, pad_y = 0.04 * (x_high - x_low), 0.04 * (y_high - y_low)
    ax.set_xlim(x_low - pad_x, x_high + pad_x)
    ax.set_ylim(y_low - pad_y, y_high + pad_y)
    ax.set_title(title, fontsize=9.5, color=TEXT, loc="left")
    if note:
        ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=7.5, color=TEXT_MUTED)


def _enrichment(feature_positive: np.ndarray, annotated: np.ndarray, feature: str, annotation: str) -> str:
    feature_positive, annotated = np.asarray(feature_positive, bool), np.asarray(annotated, bool)
    return f"{annotation}: {annotated[feature_positive].mean():.0%} of {feature} proteins vs {annotated.mean():.0%} overall"


def plot_protein_map_biology(data, coords: np.ndarray, path: Path) -> None:
    """Annotations (orange) next to the sequence features that should explain them (blue)."""
    terms, raw, targets = data.trained.terms, data.raw_test, data.targets
    label = {term: targets[:, terms.index(term)] for term in ("GO:0005634", "GO:0005886", "GO:0005576", "GO:0005739", "GO:0003677", "GO:0003700", "GO:0004672")}
    nls = raw["prosite_bipartite_nuclear_localization_signal_present"].to_numpy() > 0
    tm = raw["transmembrane_helix_count"].to_numpy()
    signal = raw["signal_peptide_like"].to_numpy() > 0
    kinase_motif = (raw["prosite_protein_kinase_atp_binding_present"].to_numpy() > 0) | (raw["prosite_st_kinase_active_site_present"].to_numpy() > 0)
    kingdoms = raw[[c for c in raw.columns if c.startswith("kingdom_")]].idxmax(axis=1).str.removeprefix("kingdom_").str.replace("eukaryota_", "").to_numpy()
    panels = [
        (label["GO:0005634"], "annotation", "nucleus (GO annotation)", ""),
        (nls, "feature", "nuclear localisation signal (sequence motif)", _enrichment(nls, label["GO:0005634"], "NLS+", "nucleus")),
        (label["GO:0005886"], "annotation", "plasma membrane (GO annotation)", ""),
        (tm, "continuous", "transmembrane helix count (sequence)", _enrichment(tm >= 1, label["GO:0005886"], "TM-helix", "plasma membrane")),
        (label["GO:0005576"], "annotation", "extracellular region (GO annotation)", ""),
        (signal, "feature", "signal peptide (sequence heuristic)", _enrichment(signal, label["GO:0005576"], "signal-peptide", "extracellular")),
        (label["GO:0005739"], "annotation", "mitochondrion (GO annotation)", ""),
        (kingdoms, "categorical", "kingdom", ""),
        (label["GO:0003677"], "annotation", "DNA binding (GO annotation)", ""),
        (label["GO:0003700"], "annotation", "DNA-binding transcription factor activity (GO)", ""),
        (label["GO:0004672"], "annotation", "protein kinase activity (GO annotation)", ""),
        (kinase_motif, "feature", "PROSITE kinase motif (sequence motif)", _enrichment(kinase_motif, label["GO:0004672"], "motif+", "kinase activity")),
    ]
    figure, axes = plt.subplots(3, 4, figsize=(19, 13.5), facecolor=SURFACE)
    for ax, (values, kind, title, note) in zip(axes.flat, panels):
        _map_panel(figure, ax, coords, np.asarray(values), kind, title, note)
    for row, heading in enumerate(("Localisation: nucleus and membranes", "Secretion, organelles and taxonomy", "Molecular function: DNA binding and kinases")):
        axes[row, 0].annotate(heading, xy=(0, 1.13), xycoords="axes fraction", fontsize=11, color=TEXT, fontweight="bold")
    figure.suptitle(
        f"One map of {len(coords):,} held-out proteins (UMAP of ESM-2 embeddings), coloured by GO annotations (orange) and the sequence features that should explain them (blue)",
        fontsize=12.5, color=TEXT, x=0.01, ha="left", y=1.025,
    )
    figure.tight_layout(h_pad=3.0)
    _save(figure, path, palette=True)


def plot_protein_map_dimensions(data, coords: np.ndarray, dimensions: pd.DataFrame, correlation: pd.DataFrame, path: Path, n: int = 8) -> None:
    """The same protein map coloured by the most important ESM-2 dimensions."""
    figure, axes = plt.subplots(2, 4, figsize=(19, 9.5), facecolor=SURFACE)
    for ax, row in zip(axes.flat, dimensions.head(n).itertuples()):
        best = correlation.loc[row.dimension].abs().idxmax() if row.dimension in correlation.index else None
        correlate = f"\nstrongest correlate: {pretty_feature(best)} (ρ {correlation.loc[row.dimension, best]:+.2f})" if best else ""
        title = f"{pretty_feature(row.dimension)}: rank {row.rank} ({row.importance * len(dimensions):.1f}× uniform share){correlate}"
        _map_panel(figure, ax, coords, data.raw_test[row.dimension].to_numpy(dtype=float), "continuous", title)
    figure.suptitle(
        "The 8 most important ESM-2 dimensions on the protein map (colour = dimension value; red above, blue below its median)",
        fontsize=12.5, color=TEXT, x=0.01, ha="left",
    )
    figure.tight_layout()
    _save(figure, path, dpi=130, palette=True)


# GO term -> known sequence biology, for linking SHAP's term-specific dimensions to what we know.
KNOWN_BIOLOGY = (
    ("GO:0005634", "prosite_bipartite_nuclear_localization_signal_present", "nuclear localisation signal"),
    ("GO:0005886", "transmembrane_helix_count", "transmembrane helices"),
    ("GO:0005576", "signal_peptide_score", "signal peptide score"),
    ("GO:0003677", "isoelectric_point", "isoelectric point (DNA binders are basic)"),
    ("GO:0003700", "disorder_score", "intrinsic disorder"),
    ("GO:0004672", "kinase_motif", "PROSITE kinase motif"),
)
SIGNATURE_CANDIDATES = 30


def _known_feature(raw: pd.DataFrame, column: str) -> np.ndarray:
    if column == "kinase_motif":
        return ((raw["prosite_protein_kinase_atp_binding_present"] > 0) | (raw["prosite_st_kinase_active_site_present"] > 0)).to_numpy(float)
    return raw[column].to_numpy(dtype=float)


def _auc(labels: np.ndarray, values: np.ndarray) -> float:
    """AUC of one variable alone for a label, oriented so 0.5 = no signal and 1 = perfect."""
    finite = np.isfinite(values)
    auc = roc_auc_score(labels[finite], values[finite])
    return max(auc, 1 - auc)


def term_signature_dimensions(data, importance: list[np.ndarray], meta: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """For each GO term with known biology: the ESM-2 dimensions SHAP says the model uses
    *specifically* for that term, and how they relate to the GO label and the known feature.

    Signature = among the term's top SHAP dimensions, the highest lift (the term's attribution share
    for the dimension over that dimension's mean share across all terms), so dimensions that are
    important for every term don't crowd out term-specific ones.
    """
    names = np.asarray(data.trained.feature_names, dtype=str)
    esm = np.char.startswith(names, "esm2_")
    shares = np.nanmean([seed[:, esm] / np.nansum(seed[:, esm], axis=1, keepdims=True) for seed in importance], axis=0)
    mean_share = np.nanmean(shares, axis=0)
    handcrafted = [column for column in data.raw_test.columns if not column.startswith(("esm2_", "kingdom_"))]
    rows = []
    for term, feature, feature_label in KNOWN_BIOLOGY:
        term_index = data.trained.terms.index(term)
        labels = data.targets[:, term_index].astype(bool)
        known = _known_feature(data.raw_test, feature)
        candidates = np.argsort(shares[term_index])[::-1][:SIGNATURE_CANDIDATES]
        lift = shares[term_index, candidates] / mean_share[candidates]
        for rank, position in enumerate(candidates[np.argsort(lift)[::-1]][:top], start=1):
            dimension = names[esm][position]
            values = data.raw_test[dimension].to_numpy(dtype=float)
            correlations = data.raw_test[handcrafted].corrwith(data.raw_test[dimension], method="spearman")
            rows.append(
                {
                    "term": term,
                    "name": meta.set_index("term").loc[term, "name"],
                    "known_feature": feature_label,
                    "signature_rank": rank,
                    "dimension": dimension,
                    "term_share": shares[term_index, position],
                    "lift": shares[term_index, position] / mean_share[position],
                    "dimension_auc_for_term": _auc(labels, values),
                    "known_feature_auc_for_term": _auc(labels, known),
                    "rho_with_known_feature": spearmanr(values, known, nan_policy="omit").statistic,
                    "best_handcrafted_correlate": correlations.abs().idxmax(),
                    "rho_with_best_correlate": correlations[correlations.abs().idxmax()],
                }
            )
    return pd.DataFrame(rows)


def _annotation_outline(ax: plt.Axes, coords: np.ndarray, annotated: np.ndarray, seed: int = 0) -> None:
    """Contour enclosing the densest half of the annotated proteins (KDE on the map)."""
    from scipy.stats import gaussian_kde

    points = coords[annotated]
    if len(points) > 3000:
        points = points[np.random.default_rng(seed).choice(len(points), 3000, replace=False)]
    kde = gaussian_kde(points.T, bw_method=0.15)
    (x_low, y_low), (x_high, y_high) = np.percentile(coords, [0.5, 99.5], axis=0)
    grid_x, grid_y = np.meshgrid(np.linspace(x_low, x_high, 200), np.linspace(y_low, y_high, 200))
    density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    level = np.percentile(kde(points.T), 50)
    ax.contour(grid_x, grid_y, density, levels=[level], colors=TEXT, linewidths=1.1)


def plot_term_dimension_links(data, coords: np.ndarray, signatures: pd.DataFrame, path: Path) -> None:
    """Per GO term: annotation | known sequence feature | the term's signature ESM-2 dimension | how
    well that dimension alone separates annotated proteins. Black outline = where annotated proteins sit."""
    top = signatures[signatures["signature_rank"] == 1].reset_index(drop=True)
    figure, axes = plt.subplots(len(top), 4, figsize=(21, 4.4 * len(top)), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    for row, link in top.iterrows():
        term_index = data.trained.terms.index(link["term"])
        annotated = data.targets[:, term_index].astype(bool)
        feature = next(column for term, column, _ in KNOWN_BIOLOGY if term == link["term"])
        known = _known_feature(data.raw_test, feature)
        binary = np.isin(np.unique(known[np.isfinite(known)]), [0, 1]).all()
        dimension_values = data.raw_test[link["dimension"]].to_numpy(dtype=float)
        short_name = "kinase motif" if feature == "kinase_motif" else pretty_feature(feature)
        _map_panel(figure, axes[row, 0], coords, annotated, "annotation", f"{link['name']}\n(GO annotation; outline = densest half)")
        _map_panel(
            figure, axes[row, 1], coords, known, "feature" if binary else "continuous",
            f"known biology: {link['known_feature']}\nalone predicts the term with AUC {link['known_feature_auc_for_term']:.2f}",
        )
        _map_panel(
            figure, axes[row, 2], coords, dimension_values, "continuous",
            f"ESM-2 {pretty_feature(link['dimension'])} (embedding value)\n"
            f"{link['lift']:.1f}× its usual SHAP share for this term; ρ with {short_name}: {link['rho_with_known_feature']:+.2f}",
        )
        for column in range(3):
            _annotation_outline(axes[row, column], coords, annotated)

        ax = axes[row, 3]
        _style(ax, grid_axis="")
        low, high = np.nanpercentile(dimension_values, [1, 99])
        bins = np.linspace(low, high, 50)
        ax.hist(dimension_values[~annotated], bins=bins, density=True, histtype="stepfilled", color=BACKGROUND_POINT, label="other proteins")
        ax.hist(dimension_values[annotated], bins=bins, density=True, histtype="step", linewidth=2, color=ANNOTATION_COLOR, label="annotated")
        ax.set_yticks([])
        ax.set_xlabel(f"{pretty_feature(link['dimension'])} value", fontsize=8.5, color=TEXT_MUTED)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        ax.set_title(f"ESM-2 {pretty_feature(link['dimension'])} alone separates the term\nAUC {link['dimension_auc_for_term']:.2f} (known feature: {link['known_feature_auc_for_term']:.2f})", fontsize=9.5, color=TEXT, loc="left")
    figure.suptitle(
        "Linking what we know to what the model uses: GO annotation | known sequence biology | the ESM-2 dimension the model relies on most for that term (by SHAP) | how well it separates",
        fontsize=12.5, color=TEXT, x=0.01, ha="left", y=1.005,
    )
    figure.tight_layout(h_pad=2.5)
    _save(figure, path, dpi=110, palette=True)


def plot_go_term_map(importance: list[np.ndarray], feature_names: list[str], meta: pd.DataFrame, path: Path, labelled_path: Path) -> None:
    """Each GO term placed by UMAP of its ESM-2 attribution profile: terms the model predicts with
    the same embedding dimensions sit together."""
    import umap

    esm = np.char.startswith(np.asarray(feature_names, dtype=str), "esm2_")
    profiles = np.nanmean([seed[:, esm] / np.nansum(seed[:, esm], axis=1, keepdims=True) for seed in importance], axis=0)
    keep = np.isfinite(profiles).all(axis=1)
    profiles, terms = profiles[keep], meta[keep].reset_index(drop=True)
    coords = umap.UMAP(n_neighbors=15, min_dist=0.15, metric="cosine", random_state=0).fit_transform(profiles)

    # Name clusters after a representative, reasonably specific term (skip the most generic ones).
    clusters = KMeans(n_clusters=14, n_init=10, random_state=0).fit_predict(coords)
    specific = terms["train_positive"] < terms["train_positive"].quantile(0.9)
    labels = np.empty(len(terms), dtype=object)
    for cluster in np.unique(clusters):
        members = terms[(clusters == cluster) & specific]
        members = members if len(members) else terms[clusters == cluster]
        labels[clusters == cluster] = members.sort_values("train_positive", ascending=False)["name"].iloc[0]

    figure, ax = plt.subplots(figsize=(11, 8.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for aspect, name in ASPECT_NAMES.items():
        rows = (terms["aspect"] == aspect).to_numpy()
        ax.scatter(*coords[rows].T, s=9, color=ASPECT_COLORS[aspect], alpha=0.55, linewidths=0, label=f"{name} ({rows.sum()})")
    labels_to_place = []
    for term in DEEP_DIVE_TERMS:
        row = np.flatnonzero(terms["term"] == term)
        if len(row):
            x, y = coords[row[0]]
            ax.scatter(x, y, s=70, marker="*", color=TEXT, zorder=4)
            labels_to_place.append(ax.text(x, y, terms.loc[row[0], "name"], fontsize=7.5, color=TEXT, zorder=5))
    adjust_text(
        labels_to_place, ax=ax, expand=(1.3, 1.6),
        arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "lw": 0.6},
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", markerscale=1.5)
    ax.set_title("GO terms mapped by which ESM-2 dimensions the model uses to predict them (UMAP of SHAP profiles; ★ = deep-dive terms)", fontsize=10.5, color=TEXT, loc="left")
    _save(figure, path)

    try:
        import datamapplot

        figure, _ = datamapplot.create_plot(
            coords, labels, title="The model's map of GO terms",
            sub_title="Each point is a GO term, placed by the ESM-2 dimensions the network uses to predict it",
            label_font_size=9, darkmode=False, figsize=(12, 10),
        )
        figure.savefig(labelled_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    except Exception as error:  # the labelled map is a nicety; the plain map above always exists
        LOGGER.warning("datamapplot figure skipped: %s", error)


# ---------------------------------------------------------------- orchestration


def write_explainability_report(model_dirs: list[Path], out_dir: Path, cache_dir: Path, max_positives: int, nsamples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [explain_seed(model_dir, cache_dir / model_dir.parent.name, max_positives, nsamples) for model_dir in model_dirs]
    data = seeds[0]["data"]
    meta = term_metadata(data)

    shares = aggregate_over_seeds([seed["shares"] for seed in seeds], list(GROUPS))
    shares.merge(meta[["term", "name", "aspect", "test_positive", "response_term"]], on="term").round(5).to_csv(out_dir / "term_group_attribution.csv", index=False)
    drop_columns = [f"{group}_auc_drop" for group in GROUPS]
    permutation = aggregate_over_seeds([seed["permutation"] for seed in seeds], ["baseline_auc", *drop_columns])
    permutation.merge(meta[["term", "name", "aspect"]], on="term").round(5).to_csv(out_dir / "permutation_importance.csv", index=False)

    feature_names = data.trained.feature_names
    importance = [seed["importance"] for seed in seeds]
    mean_importance = np.nanmean(importance, axis=0)
    top = []
    for term_index, term in enumerate(data.trained.terms):
        if np.isfinite(mean_importance[term_index]).all():
            order = np.argsort(mean_importance[term_index])[::-1][:10]
            top += [{"term": term, "rank": r + 1, "feature": feature_names[i], "mean_abs_shap": mean_importance[term_index, i]} for r, i in enumerate(order)]
    pd.DataFrame(top).round(5).to_csv(out_dir / "term_top_features.csv", index=False)

    rows = []
    for term in DEEP_DIVE_TERMS:
        values = np.concatenate([seed["deep_dives"][term].values for seed in seeds])
        data_values = np.concatenate([seed["deep_dives"][term].data for seed in seeds])
        for index, feature in enumerate(seeds[0]["deep_dives"][term].feature_names):
            valid = np.isfinite(data_values[:, index])
            direction = spearmanr(data_values[valid, index], values[valid, index]).statistic if valid.sum() > 2 and np.nanstd(data_values[valid, index]) > 0 else 0.0
            rows.append({"term": term, "feature": feature, "group": feature_groups([feature])[0], "mean_abs_shap": np.abs(values[:, index]).mean(), "mean_shap": values[:, index].mean(), "direction": direction})
    features = pd.DataFrame(rows)
    features.merge(meta[["term", "name"]], on="term").round(5).to_csv(out_dir / "deep_dive_features.csv", index=False)

    dimensions, top_dimensions = dimension_importance(importance, feature_names, meta)
    dimensions.round(6).to_csv(out_dir / "esm2_dimension_importance.csv", index=False)
    correlation = dimension_feature_correlation(model_dirs[0], top_dimensions)
    correlation.round(4).to_csv(out_dir / "esm2_dimension_feature_correlation.csv")

    plot_group_shares_by_category(shares, meta, out_dir / "group_attribution_by_category.png")
    plot_deep_dive_group_shares(shares, meta, out_dir / "deep_dive_group_shares.png")
    plot_deep_dive_beeswarms(seeds[0]["deep_dives"], meta, out_dir / "deep_dives")
    plot_deep_dive_overview(features, meta, out_dir / "deep_dive_overview.png")
    plot_permutation(permutation, meta, out_dir / "permutation_importance.png")
    concentration = plot_dimension_concentration(dimensions, out_dir / "esm2_dimension_concentration.png")
    plot_dimension_feature_correlation(correlation, out_dir / "esm2_dimension_feature_correlation.png")
    coords = protein_umap(data, cache_dir / "protein_umap.npy")
    plot_protein_map_biology(data, coords, out_dir / "protein_map.png")
    plot_protein_map_dimensions(data, coords, dimensions, correlation, out_dir / "protein_map_dimensions.png")
    signatures = term_signature_dimensions(data, importance, meta)
    signatures.round(4).to_csv(out_dir / "term_signature_dimensions.csv", index=False)
    plot_term_dimension_links(data, coords, signatures, out_dir / "term_dimension_links.png")
    plot_go_term_map(importance, feature_names, meta, out_dir / "go_term_map.png", out_dir / "go_term_map_labelled.png")
    summary = {"esm2_dimension_seed_spearman": dimensions.attrs.get("seed_spearman"), "concentration": concentration}
    pd.Series(summary).to_json(out_dir / "summary.json", indent=2)
    LOGGER.info("wrote explainability report to %s", out_dir)
