"""Headline figure: internal holdout Fmax vs official CAFA6 weighted Fmax for every ESM-2 model.

Reads only committed results (results/summary/runs.csv, results/hparam-search/final_comparison_runs.csv
and results/hparam-search/tpe/final_comparison_runs.csv), so it can be regenerated without data/.

    uv run python scripts/plot_summary_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

# Reference palette (dataviz skill): slot 1 blue, slot 2 orange; text in neutral inks.
WITHOUT_PROMOTERS, WITH_PROMOTERS = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

MODELS = (
    # label, source, config, has promoters, architecture (default / random search / Bayesian search)
    ("ESM-2", "ablation", "esm2", False, "default"),
    ("ESM-2 + handcrafted", "ablation", "esm2_prot", False, "default"),
    ("ESM-2 + handcrafted + kingdom", "ablation", "esm2_prot_tax", False, "default"),
    ("+ promoters (all)", "ablation", "all", True, "default"),
    ("wider, + promoters", "final", "winner_all", True, "random"),
    ("wider, ESM-2 + handcrafted + kingdom", "final", "winner_esm2_prot_tax", False, "random"),
    ("Bayesian search, + promoters", "tpe", "arch3_trial110", True, "tpe"),
    ("Bayesian search, no promoters", "tpe", "arch3_trial95", False, "tpe"),
)
MARKERS = {"default": ("o", 110), "random": ("D", 80), "tpe": ("*", 220)}

def load(results_dir: Path) -> pd.DataFrame:
    ablation = pd.read_csv(results_dir / "summary" / "runs.csv")
    final = {
        "final": pd.read_csv(results_dir / "hparam-search" / "final_comparison_runs.csv"),
        "tpe": pd.read_csv(results_dir / "hparam-search" / "tpe" / "final_comparison_runs.csv"),
    }
    rows = []
    for label, source, config, promoters, architecture in MODELS:
        if source == "ablation":
            runs = ablation[ablation["config"] == config]
            internal, official = runs["internal_cafa_fmax_mean"], runs["official_f_w_overall_mean"]
        else:
            runs = final[source][final[source]["config"] == config]
            internal, official = runs["test_nn_cafa_fmax_mean"], runs["f_w_overall"]
        rows.append(
            {
                "label": label,
                "promoters": promoters,
                "architecture": architecture,
                "internal": internal.mean(),
                "internal_sd": internal.std(),
                "official": official.mean(),
                "official_sd": official.std(),
            }
        )
    return pd.DataFrame(rows)


def plot(models: pd.DataFrame, path: Path) -> None:
    figure, ax = plt.subplots(figsize=(8.2, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)

    labels = []
    for row in models.itertuples():
        color = WITH_PROMOTERS if row.promoters else WITHOUT_PROMOTERS
        marker, size = MARKERS[row.architecture]
        ax.errorbar(row.internal, row.official, xerr=row.internal_sd, yerr=row.official_sd, fmt="none", ecolor=color, elinewidth=1.5, alpha=0.6)
        ax.scatter(row.internal, row.official, s=size, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        labels.append(ax.text(row.internal, row.official, row.label, fontsize=8.5, color=TEXT))

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=WITHOUT_PROMOTERS, markersize=8, label="without promoters"),
        plt.Line2D([], [], marker="o", linestyle="", color=WITH_PROMOTERS, markersize=8, label="with promoters"),
        plt.Line2D([], [], marker="o", linestyle="", color=TEXT_MUTED, markersize=8, label="default architecture"),
        plt.Line2D([], [], marker="D", linestyle="", color=TEXT_MUTED, markersize=7, label="random search (wider)"),
        plt.Line2D([], [], marker="*", linestyle="", color=TEXT_MUTED, markersize=12, label="Bayesian search (asymmetric loss)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlabel("Internal IA-weighted Fmax (random 20% holdout)", fontsize=9.5, color=TEXT_MUTED)
    ax.set_ylabel("Official weighted Fmax (CAFA6 evaluator, NK/LK/PK)", fontsize=9.5, color=TEXT_MUTED)
    ax.set_title(
        "Internal holdout vs official CAFA6 score: the Bayesian search's models lead on both\n"
        "mean ± sd over 3 seeds; models without ESM-2 (official f_w ≈ 0.25) not shown",
        fontsize=10.5,
        color=TEXT,
        loc="left",
    )
    ax.margins(x=0.08, y=0.12)
    adjust_text(labels, ax=ax, expand=(1.3, 1.6), arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "linewidth": 0.6})
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/summary/internal_vs_official.png"))
    args = parser.parse_args()
    plot(load(args.results_dir), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
