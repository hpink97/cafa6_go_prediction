"""Do promoters help some kinds of GO terms more than others?

Hypothesis: promoters carry regulatory (expression) signal, so they should help most for
"response to <stimulus>" processes, which are largely defined by inducible expression. For every
term, the effect is the paired change in held-out ROC-AUC from adding promoters, averaged over
seeds. Response terms are compared with all other biological-process terms, defined two ways:
by name ("response to ..." / "cellular response to ...") and by ontology (descendants of
GO:0050896, response to stimulus). Rare terms have noisier AUCs, so the main test permutes the
response label only within term-prevalence bins.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from modeling.evaluation.go import GeneOntology

RESPONSE_TO_STIMULUS = "GO:0050896"
RESPONSE_NAME = re.compile(r"^(cellular )?response to ")
# (with, without, label): each isolates adding promoters, with and without ESM-2.
PROMOTER_COMPARISONS = (
    ("all", "esm2_prot_tax", "promoters with ESM-2"),
    ("prot_tax_promoters", "prot_tax", "promoters without ESM-2"),
)
DEFINITIONS = {"by_name": "name starts 'response to'", "by_ontology": "descendant of response to stimulus"}


def load_term_names(obo_path: str | Path) -> dict[str, str]:
    names: dict[str, str] = {}
    term = None
    for line in Path(obo_path).read_text().splitlines():
        if line.startswith("["):
            term = None
        elif line.startswith("id: GO:"):
            term = line[4:].strip()
        elif line.startswith("name: ") and term:
            names[term] = line[6:].strip()
    return names


def classify_terms(terms: pd.DataFrame, ontology: GeneOntology, names: dict[str, str]) -> pd.DataFrame:
    """Biological-process terms flagged as response terms under each definition."""
    process = terms[terms["aspect"] == "P"][["term"]].copy()
    process["name"] = process["term"].map(names)
    process["by_name"] = process["name"].fillna("").str.match(RESPONSE_NAME)
    process["by_ontology"] = [
        term != RESPONSE_TO_STIMULUS and RESPONSE_TO_STIMULUS in ontology.ancestors(term) for term in process["term"]
    ]
    return process.set_index("term")


def per_term_deltas(sweep_dir: Path, seeds: list[int], with_: str, without: str) -> pd.DataFrame:
    """Per-term held-out ROC-AUC change (with - without) for each shared seed, plus test prevalence."""
    frames = []
    for seed in seeds:
        tables = {
            name: pd.read_csv(sweep_dir / name / f"seed{seed}" / "model" / "test_nn_metrics.tsv", sep="\t").set_index("term")
            for name in (with_, without)
        }
        frame = pd.DataFrame(
            {
                "seed": seed,
                "n_positive": tables[with_]["n_positive"],
                "delta": tables[with_]["roc_auc"] - tables[without]["roc_auc"],
            }
        )
        frames.append(frame.dropna(subset=["delta"]))
    return pd.concat(frames).rename_axis("term").reset_index()


def stratified_permutation_test(
    delta: np.ndarray,
    in_group: np.ndarray,
    prevalence: np.ndarray,
    n_bins: int = 5,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean delta difference (group - rest) and its two-sided p-value, permuting group labels only
    within prevalence quantile bins so a prevalence difference cannot create the effect."""
    bins = pd.qcut(np.log10(prevalence), q=n_bins, labels=False, duplicates="drop")
    observed = delta[in_group].mean() - delta[~in_group].mean()
    rng = np.random.default_rng(seed)
    bin_indices = [np.flatnonzero(bins == value) for value in np.unique(bins)]
    exceed = 0
    for _ in range(n_permutations):
        shuffled = in_group.copy()
        for indices in bin_indices:
            shuffled[indices] = rng.permutation(in_group[indices])
        exceed += abs(delta[shuffled].mean() - delta[~shuffled].mean()) >= abs(observed) - 1e-12
    return float(observed), (exceed + 1) / (n_permutations + 1)


def response_term_analysis(sweep_dir: Path, runs: pd.DataFrame, obo_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Statistics per (comparison, definition) and the per-term deltas behind them."""
    ontology = GeneOntology.from_obo(obo_path)
    names = load_term_names(obo_path)
    stats_rows, term_tables = [], []
    for with_, without, label in PROMOTER_COMPARISONS:
        seeds = sorted(set(runs.loc[runs["config"] == with_, "seed"]) & set(runs.loc[runs["config"] == without, "seed"]))
        if not seeds:
            continue
        terms = pd.read_csv(sweep_dir / with_ / f"seed{seeds[0]}" / "model" / "terms.tsv", sep="\t")
        classes = classify_terms(terms, ontology, names)
        deltas = per_term_deltas(sweep_dir, seeds, with_, without)
        deltas = deltas[deltas["term"].isin(classes.index)].join(classes, on="term")
        per_term = deltas.groupby("term").agg(
            name=("name", "first"), by_name=("by_name", "first"), by_ontology=("by_ontology", "first"),
            n_positive=("n_positive", "mean"), delta=("delta", "mean"), n_seeds=("seed", "count"),
        )
        term_tables.append(per_term.assign(comparison=label).reset_index())
        for definition in DEFINITIONS:
            in_group = per_term[definition].to_numpy(dtype=bool)
            delta = per_term["delta"].to_numpy()
            difference, p_permutation = stratified_permutation_test(delta, in_group, per_term["n_positive"].to_numpy())
            seed_differences = [
                group["delta"][group[definition]].mean() - group["delta"][~group[definition]].mean()
                for _, group in deltas.groupby("seed")
            ]
            stats_rows.append(
                {
                    "comparison": label,
                    "definition": definition,
                    "n_response_terms": int(in_group.sum()),
                    "n_other_process_terms": int((~in_group).sum()),
                    "mean_delta_response": delta[in_group].mean(),
                    "mean_delta_other": delta[~in_group].mean(),
                    "median_delta_response": np.median(delta[in_group]),
                    "median_delta_other": np.median(delta[~in_group]),
                    "share_improved_response": (delta[in_group] > 0).mean(),
                    "share_improved_other": (delta[~in_group] > 0).mean(),
                    "difference_in_means": difference,
                    "p_permutation_prevalence_stratified": p_permutation,
                    "p_mann_whitney": mannwhitneyu(delta[in_group], delta[~in_group]).pvalue,
                    "n_seeds_response_larger": int(sum(value > 0 for value in seed_differences)),
                    "n_seeds": len(seed_differences),
                }
            )
    return pd.DataFrame(stats_rows), pd.concat(term_tables, ignore_index=True) if term_tables else pd.DataFrame()
