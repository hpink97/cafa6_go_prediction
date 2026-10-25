"""Is one configuration better than another? Paired bootstrap over proteins.

Seed-level t-tests have little power with three seeds, so this resamples *proteins*: within each
seed, the same bootstrap resample of evaluation proteins is applied to both configurations, the
metric difference is recomputed, and differences are averaged over seeds. This is how CAFA
compares methods, and it answers "would the ranking hold on a different draw of targets?".

Metrics:
    internal_fmax  IA-weighted Fmax on the held-out 20% (identical to the trainer's own metric)
    official_f_w   CAFA weighted Fmax against the CAFA6 release (NK and LK), recomputed from each
                   run's saved predictions with the evaluator's settings (terms of interest, 500
                   terms per protein, propagation). It reproduces cafaeval's f_w to within 3e-4 per
                   aspect. PK is left out: cafaeval's handling of known annotations is not yet
                   reproduced (recall differs), so PK is covered by seed-level tests on cafaeval's
                   own scores instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.go_evaluation import (
    NAMESPACE_TO_ASPECT,
    GeneOntology,
    ProteinCurves,
    build_truth_ia_totals,
    load_ia_weights,
    protein_fmax_curves,
    weighted_fmax,
)

REGIMES = ("NK", "LK", "PK")
BOOTSTRAP_REGIMES = ("NK", "LK")
ASPECTS = tuple(NAMESPACE_TO_ASPECT.values())


@dataclass
class EvaluationCurves:
    """Per-protein curves for one run: `curves[block][aspect]`, proteins in `protein_ids[block]`.

    A block is one set of evaluation proteins (the internal test split, or one official regime).
    """

    protein_ids: dict[str, np.ndarray]
    curves: dict[str, dict[str, ProteinCurves]]


def internal_curves(run_dir: Path) -> EvaluationCurves:
    model_dir = run_dir / "model"
    config = json.loads((model_dir / "run_config.json").read_text())
    ontology = GeneOntology.from_obo(config["go_obo_path"])
    ia_weights = load_ia_weights(config["ia_path"])
    with np.load(model_dir / "test_predictions.npz") as predictions:
        protein_ids = predictions["protein_ids"]
        terms = [str(term) for term in predictions["terms"]]
        targets = predictions["targets"]
        probabilities = predictions["propagated_probabilities"]
    truth = build_truth_ia_totals(config["train_terms_path"], protein_ids, ontology, ia_weights)
    curves = protein_fmax_curves(targets, probabilities, terms, ontology, ia_weights, truth_ia_totals=truth)
    return EvaluationCurves({"internal": protein_ids}, {"internal": curves})


def _propagated_pairs(table: pd.DataFrame, ontology: GeneOntology, keep: set[str]) -> set[tuple[str, str]]:
    return {
        (protein_id, ancestor)
        for protein_id, term in table[["EntryID", "term"]].drop_duplicates().itertuples(index=False, name=None)
        for ancestor in ontology.ancestors(term)
        if ancestor in keep
    }


def official_curves(
    predictions_path: Path,
    release_dir: Path,
    ontology: GeneOntology,
    ia_weights: dict[str, float],
    max_terms: int = 500,
    threshold_step: float = 0.001,
    regimes: tuple[str, ...] = BOOTSTRAP_REGIMES,
) -> EvaluationCurves:
    """Per-protein curves against each regime's ground truth, mirroring cafaeval's settings."""
    terms_of_interest = set(pd.read_csv(release_dir / "May2026_groundtruth_terms_of_interest.txt", header=None)[0])
    predictions = pd.read_csv(predictions_path, sep="\t", header=None, names=["EntryID", "term", "score"])
    # cafaeval keeps the first `max_terms` lines per protein; prediction files are written best-first.
    predictions = predictions.groupby("EntryID", sort=False).head(max_terms)
    predictions["term"] = predictions["term"].map(ontology.canonical_id)
    predictions = predictions[predictions["term"].isin(terms_of_interest)]

    protein_ids, curves = {}, {}
    for regime in regimes:
        truth_table = pd.read_csv(release_dir / f"May2026_groundtruth_{regime}.tsv", sep="\t")
        truth = _propagated_pairs(truth_table, ontology, terms_of_interest)
        excluded: set[tuple[str, str]] = set()
        if regime == "PK":
            excluded = _propagated_pairs(pd.read_csv(release_dir / "May2026_groundtruth_PK_known.tsv", sep="\t"), ontology, terms_of_interest)
            truth -= excluded
        proteins = np.asarray(sorted({protein_id for protein_id, _ in truth}))
        protein_index = {protein_id: index for index, protein_id in enumerate(proteins)}
        regime_predictions = predictions[predictions["EntryID"].isin(protein_index)]
        terms = sorted({term for _, term in truth} | set(regime_predictions["term"]))
        term_index = {term: index for index, term in enumerate(terms)}

        targets = np.zeros((len(proteins), len(terms)), dtype=np.uint8)
        rows, columns = zip(*((protein_index[p], term_index[t]) for p, t in truth))
        targets[list(rows), list(columns)] = 1
        scores = np.zeros(targets.shape, dtype=np.float32)
        scores[regime_predictions["EntryID"].map(protein_index), regime_predictions["term"].map(term_index)] = regime_predictions["score"]
        scores = ontology.propagate_scores(scores, terms, copy=False)
        if excluded:
            pairs = [(protein_index[p], term_index[t]) for p, t in excluded if p in protein_index and t in term_index]
            if pairs:
                excluded_rows, excluded_columns = zip(*pairs)
                scores[list(excluded_rows), list(excluded_columns)] = 0.0
        protein_ids[regime] = proteins
        curves[regime] = protein_fmax_curves(targets, scores, terms, ontology, ia_weights, threshold_step=threshold_step)
    return EvaluationCurves(protein_ids, curves)


def point_estimate(evaluation: EvaluationCurves) -> dict[str, float]:
    """Fmax per block and aspect, with each block's aspect mean and the overall mean of block means."""
    values: dict[str, float] = {}
    for block, block_curves in evaluation.curves.items():
        ones = np.ones((1, len(evaluation.protein_ids[block])))
        for aspect, curves in block_curves.items():
            values[f"{block}_{aspect}"] = float(weighted_fmax(curves, ones)[0])
        values[f"{block}_mean"] = float(np.mean([values[f"{block}_{aspect}"] for aspect in block_curves]))
    values["overall"] = float(np.mean([values[f"{block}_mean"] for block in evaluation.curves]))
    return values


def _bootstrap_overall(evaluation: EvaluationCurves, weights: dict[str, np.ndarray]) -> np.ndarray:
    block_means = []
    for block, block_curves in evaluation.curves.items():
        block_means.append(np.mean([weighted_fmax(curves, weights[block]) for curves in block_curves.values()], axis=0))
    return np.mean(block_means, axis=0)


def paired_bootstrap(
    pairs_by_seed: dict[int, tuple[EvaluationCurves, EvaluationCurves]],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean over seeds of (A - B) overall Fmax, with a percentile CI and two-sided p from resampling proteins.

    Within a seed both configurations share the evaluation proteins and the same resample.
    """
    rng = np.random.default_rng(seed)
    observed, replicates = [], []
    for evaluation_a, evaluation_b in pairs_by_seed.values():
        for block in evaluation_a.protein_ids:
            if not np.array_equal(evaluation_a.protein_ids[block], evaluation_b.protein_ids[block]):
                raise ValueError(f"Evaluation proteins differ between the two runs in block {block}")
        weights = {
            block: rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=n_bootstrap).astype(np.float64)
            for block, ids in evaluation_a.protein_ids.items()
        }
        observed.append(point_estimate(evaluation_a)["overall"] - point_estimate(evaluation_b)["overall"])
        replicates.append(_bootstrap_overall(evaluation_a, weights) - _bootstrap_overall(evaluation_b, weights))
    differences = np.mean(replicates, axis=0)
    low, high = np.percentile(differences, [2.5, 97.5])
    p_value = min(1.0, 2.0 * min((differences <= 0).mean(), (differences >= 0).mean()))
    return {
        "delta": float(np.mean(observed)),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_bootstrap": max(p_value, 1.0 / n_bootstrap),
        "n_seeds": len(pairs_by_seed),
        "n_bootstrap": n_bootstrap,
    }


def compare_to_reference(
    sweep_dir: Path,
    reference: str,
    others: list[str],
    release_dir: Path,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Reference minus each other config: protein bootstrap on internal and official (NK+LK) Fmax."""
    ontology = GeneOntology.from_obo(release_dir / "go-basic.obo")
    ia_weights = load_ia_weights(release_dir / "IA.tsv")
    seeds = sorted(int(path.name.removeprefix("seed")) for path in (sweep_dir / reference).glob("seed*"))

    def load(config: str, seed: int, metric: str) -> EvaluationCurves:
        run = sweep_dir / config / f"seed{seed}"
        if metric == "internal_fmax":
            return internal_curves(run)
        return official_curves(next((run / "predictions").glob("*.tsv")), release_dir, ontology, ia_weights)

    rows = []
    for metric in ("internal_fmax", "official_f_w_nk_lk"):
        reference_curves = {seed: load(reference, seed, metric) for seed in seeds}
        for other in others:
            other_seeds = [seed for seed in seeds if (sweep_dir / other / f"seed{seed}").exists()]
            pairs = {seed: (reference_curves[seed], load(other, seed, metric)) for seed in other_seeds}
            rows.append({"reference": reference, "other": other, "metric": metric, **paired_bootstrap(pairs, n_bootstrap)})
    return pd.DataFrame(rows)
