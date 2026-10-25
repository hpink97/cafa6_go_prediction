"""Gene Ontology propagation and CAFA-style IA-weighted evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

NAMESPACE_TO_ASPECT = {
    "biological_process": "P",
    "molecular_function": "F",
    "cellular_component": "C",
}
ASPECT_TO_NAMESPACE = {aspect: namespace for namespace, aspect in NAMESPACE_TO_ASPECT.items()}


class GeneOntology:
    """Minimal GO DAG containing canonical IDs, namespaces, and valid parents."""

    def __init__(
        self,
        parents: dict[str, set[str]],
        namespaces: dict[str, str],
        alt_id_to_id: dict[str, str] | None = None,
    ) -> None:
        self.namespaces = namespaces
        self.alt_id_to_id = alt_id_to_id or {}
        self.parents = {
            term: frozenset(
                parent
                for parent in term_parents
                if parent in namespaces and namespaces[parent] == namespaces[term]
            )
            for term, term_parents in parents.items()
            if term in namespaces
        }
        self._ancestor_cache: dict[str, frozenset[str]] = {}
        self._depth_cache: dict[str, int] = {}

    @classmethod
    def from_obo(cls, path: str | Path) -> GeneOntology:
        """Parse active GO terms and their ``is_a``/``part_of`` relationships."""
        raw_terms: dict[str, dict[str, object]] = {}
        current: dict[str, object] | None = None

        def save_current() -> None:
            if current is None or "id" not in current or current.get("obsolete", False):
                return
            term = str(current["id"])
            raw_terms[term] = current.copy()

        with Path(path).open() as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line.startswith("[") and line.endswith("]"):
                    save_current()
                    current = (
                        {"parents": set(), "alt_ids": set(), "obsolete": False}
                        if line == "[Term]"
                        else None
                    )
                elif current is None:
                    continue
                elif line.startswith("id: GO:"):
                    current["id"] = line[4:]
                elif line.startswith("namespace: "):
                    current["namespace"] = line[11:]
                elif line.startswith("alt_id: GO:"):
                    current["alt_ids"].add(line[8:])  # type: ignore[union-attr]
                elif line.startswith("is_a: GO:"):
                    current["parents"].add(line.split()[1])  # type: ignore[union-attr]
                elif line.startswith("relationship: part_of GO:"):
                    current["parents"].add(line.split()[2])  # type: ignore[union-attr]
                elif line == "is_obsolete: true":
                    current["obsolete"] = True
        save_current()

        namespaces = {
            term: str(record["namespace"])
            for term, record in raw_terms.items()
            if "namespace" in record and record["namespace"] in NAMESPACE_TO_ASPECT
        }
        alt_id_to_id = {
            alt_id: term
            for term, record in raw_terms.items()
            if term in namespaces
            for alt_id in record["alt_ids"]
        }
        parents: dict[str, set[str]] = {}
        for term, record in raw_terms.items():
            if term not in namespaces:
                continue
            canonical_parents = {
                alt_id_to_id.get(parent, parent)
                for parent in record["parents"]
            }
            parents[term] = canonical_parents
        return cls(parents=parents, namespaces=namespaces, alt_id_to_id=alt_id_to_id)

    def canonical_id(self, term: str) -> str:
        return self.alt_id_to_id.get(term, term)

    def ancestors(self, term: str) -> frozenset[str]:
        """Return ``term`` and all reachable same-namespace ancestors."""
        term = self.canonical_id(term)
        if term not in self.namespaces:
            return frozenset()
        if term in self._ancestor_cache:
            return self._ancestor_cache[term]

        ancestors = {term}
        pending = [term]
        while pending:
            child = pending.pop()
            for parent in self.parents.get(child, ()):
                if parent not in ancestors:
                    ancestors.add(parent)
                    pending.append(parent)
        result = frozenset(ancestors)
        self._ancestor_cache[term] = result
        return result

    def depth(self, term: str, visiting: set[str] | None = None) -> int:
        """Longest distance from a root, used to propagate children before parents."""
        term = self.canonical_id(term)
        if term in self._depth_cache:
            return self._depth_cache[term]
        visiting = set() if visiting is None else visiting
        if term in visiting:
            raise ValueError(f"Cycle detected in GO graph at {term}")
        visiting.add(term)
        parents = self.parents.get(term, ())
        depth = 0 if not parents else 1 + max(self.depth(parent, visiting) for parent in parents)
        visiting.remove(term)
        self._depth_cache[term] = depth
        return depth

    def expand_with_ancestors(self, terms: list[str]) -> list[str]:
        """Canonicalize terms and append every missing ancestor deterministically."""
        canonical_terms = []
        seen = set()
        for term in terms:
            canonical = self.canonical_id(term)
            if canonical in self.namespaces and canonical not in seen:
                canonical_terms.append(canonical)
                seen.add(canonical)
        ancestors = set().union(*(self.ancestors(term) for term in canonical_terms)) if canonical_terms else set()
        appended = sorted(
            ancestors - seen,
            key=lambda term: (NAMESPACE_TO_ASPECT[self.namespaces[term]], self.depth(term), term),
        )
        return [*canonical_terms, *appended]

    def propagate_scores(
        self,
        values: np.ndarray,
        terms: list[str],
        *,
        copy: bool = True,
    ) -> np.ndarray:
        """Propagate each child value to its ancestors using iterative maximum."""
        if values.ndim != 2 or values.shape[1] != len(terms):
            raise ValueError(f"Expected a 2D matrix with {len(terms)} columns, got {values.shape}")
        canonical_terms = [self.canonical_id(term) for term in terms]
        if len(set(canonical_terms)) != len(canonical_terms):
            raise ValueError("Term list contains duplicate canonical GO IDs")
        result = values.copy() if copy else values
        term_index = {term: index for index, term in enumerate(canonical_terms)}
        for child in sorted(canonical_terms, key=self.depth, reverse=True):
            child_index = term_index[child]
            for parent in self.parents.get(child, ()):
                parent_index = term_index.get(parent)
                if parent_index is not None:
                    np.maximum(result[:, parent_index], result[:, child_index], out=result[:, parent_index])
        return result


def load_ia_weights(path: str | Path) -> dict[str, float]:
    table = pd.read_csv(path, sep="\t", header=None, names=["term", "ia"])
    if table["term"].duplicated().any():
        raise ValueError(f"Duplicate GO terms in IA file {path}")
    return dict(zip(table["term"], table["ia"].astype(float)))


def limit_top_predictions(probabilities: np.ndarray, max_predictions: int | None) -> np.ndarray:
    """Keep the highest-scoring explicit terms per protein before GO propagation."""
    if probabilities.ndim != 2:
        raise ValueError(f"Expected a 2D probability matrix, got {probabilities.shape}")
    if max_predictions is None or max_predictions >= probabilities.shape[1]:
        return probabilities.copy()
    if max_predictions < 1:
        raise ValueError("max_predictions must be positive or None")
    top_indices = np.argpartition(probabilities, -max_predictions, axis=1)[:, -max_predictions:]
    limited = np.zeros_like(probabilities)
    np.put_along_axis(limited, top_indices, np.take_along_axis(probabilities, top_indices, axis=1), axis=1)
    return limited


def build_closed_label_matrix(
    train_terms_path: str | Path,
    protein_ids: pd.Index,
    terms: list[str],
    ontology: GeneOntology,
    *,
    ancestor_closure: bool = True,
) -> pd.DataFrame:
    """Build labels from all direct annotations, retaining output terms and ancestors."""
    term_index = {ontology.canonical_id(term): index for index, term in enumerate(terms)}
    protein_index = {protein_id: index for index, protein_id in enumerate(protein_ids)}
    labels = np.zeros((len(protein_ids), len(terms)), dtype=np.uint8)
    annotations = pd.read_csv(train_terms_path, sep="\t", usecols=["EntryID", "term"])
    annotations = annotations[annotations["EntryID"].isin(protein_index)].drop_duplicates()

    output_indices: dict[str, np.ndarray] = {}
    for direct_term in annotations["term"].unique():
        canonical = ontology.canonical_id(direct_term)
        retained_terms = ontology.ancestors(canonical) if ancestor_closure else frozenset((canonical,))
        output_indices[direct_term] = np.fromiter(
            (term_index[term] for term in retained_terms if term in term_index),
            dtype=np.int64,
        )

    for protein_id, direct_term in annotations.itertuples(index=False, name=None):
        indices = output_indices[direct_term]
        if len(indices):
            labels[protein_index[protein_id], indices] = 1
    return pd.DataFrame(labels, index=protein_ids, columns=terms)


def build_truth_ia_totals(
    train_terms_path: str | Path,
    protein_ids: np.ndarray,
    ontology: GeneOntology,
    ia_weights: dict[str, float],
) -> dict[str, np.ndarray]:
    """Total IA of the full ancestor-closed truth, including terms outside model outputs."""
    protein_index = {protein_id: index for index, protein_id in enumerate(protein_ids)}
    totals = {
        namespace: np.zeros(len(protein_ids), dtype=np.float32)
        for namespace in NAMESPACE_TO_ASPECT
    }
    annotations = pd.read_csv(train_terms_path, sep="\t", usecols=["EntryID", "term"])
    annotations = annotations[annotations["EntryID"].isin(protein_index)].drop_duplicates()

    for protein_id, group in annotations.groupby("EntryID", sort=False):
        terms_by_namespace: dict[str, set[str]] = defaultdict(set)
        for direct_term in group["term"]:
            canonical = ontology.canonical_id(direct_term)
            namespace = ontology.namespaces.get(canonical)
            if namespace is not None:
                terms_by_namespace[namespace].update(ontology.ancestors(canonical))
        row_index = protein_index[protein_id]
        for namespace, closed_terms in terms_by_namespace.items():
            totals[namespace][row_index] = sum(ia_weights.get(term, 0.0) for term in closed_terms)
    return totals


@dataclass
class ProteinCurves:
    """Per-protein IA-weighted precision and recall at each threshold, for one GO aspect.

    Rows are the proteins with ground truth in the aspect (`rows` indexes the input protein axis).
    Fmax over any re-weighting of proteins (e.g. a bootstrap resample) follows from these arrays
    without touching the predictions again; see `weighted_fmax`.
    """

    rows: np.ndarray
    precision: np.ndarray
    has_prediction: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray


def protein_fmax_curves(
    targets: np.ndarray,
    propagated_probabilities: np.ndarray,
    terms: list[str],
    ontology: GeneOntology,
    ia_weights: dict[str, float],
    *,
    truth_ia_totals: dict[str, np.ndarray] | None = None,
    threshold_step: float = 0.01,
    include_aspects: set[str] | None = None,
) -> dict[str, ProteinCurves]:
    """Per-protein CAFA curves by aspect, from ancestor-propagated predictions."""
    if targets.shape != propagated_probabilities.shape or targets.shape[1] != len(terms):
        raise ValueError("Targets, probabilities, and terms have incompatible shapes")
    if not 0.0 < threshold_step < 1.0:
        raise ValueError("threshold_step must be between 0 and 1")
    propagated_targets = ontology.propagate_scores(targets, terms) > 0
    thresholds = np.arange(threshold_step, 1.0, threshold_step)
    curves: dict[str, ProteinCurves] = {}
    for namespace, aspect in NAMESPACE_TO_ASPECT.items():
        if include_aspects is not None and aspect not in include_aspects:
            continue
        indices = np.asarray(
            [
                index
                for index, term in enumerate(terms)
                if ontology.namespaces.get(ontology.canonical_id(term)) == namespace and term in ia_weights
            ],
            dtype=np.int64,
        )
        if not len(indices):
            continue
        weights = np.asarray([ia_weights[terms[index]] for index in indices], dtype=np.float64)
        namespace_targets = propagated_targets[:, indices]
        namespace_probabilities = propagated_probabilities[:, indices]
        if truth_ia_totals is None:
            total_truth_weight = namespace_targets @ weights
        else:
            total_truth_weight = np.asarray(truth_ia_totals[namespace], dtype=np.float64)
        has_ground_truth = total_truth_weight > 0
        if not has_ground_truth.any():
            continue

        namespace_targets = namespace_targets[has_ground_truth]
        namespace_probabilities = namespace_probabilities[has_ground_truth]
        total_truth_weight = total_truth_weight[has_ground_truth]
        n_proteins = len(namespace_targets)
        n_bins = len(thresholds) + 1
        protein_indices = np.arange(n_proteins)
        predicted_buckets = np.zeros((n_proteins, n_bins), dtype=np.float64)
        true_positive_buckets = np.zeros_like(predicted_buckets)
        for term_index, weight in enumerate(weights):
            if weight == 0:
                continue
            score_bins = np.searchsorted(
                thresholds,
                namespace_probabilities[:, term_index],
                side="right",
            )
            np.add.at(predicted_buckets, (protein_indices, score_bins), weight)
            true_rows = namespace_targets[:, term_index]
            np.add.at(
                true_positive_buckets,
                (protein_indices[true_rows], score_bins[true_rows]),
                weight,
            )

        predicted_weight = np.cumsum(predicted_buckets[:, ::-1], axis=1)[:, ::-1][:, 1:]
        true_positive_weight = np.cumsum(true_positive_buckets[:, ::-1], axis=1)[:, ::-1][:, 1:]
        has_predictions = predicted_weight > 0
        curves[aspect] = ProteinCurves(
            rows=np.flatnonzero(has_ground_truth),
            precision=np.divide(
                true_positive_weight,
                predicted_weight,
                out=np.zeros_like(true_positive_weight),
                where=has_predictions,
            ),
            has_prediction=has_predictions,
            recall=true_positive_weight / total_truth_weight[:, np.newaxis],
            thresholds=thresholds,
        )
    return curves


def weighted_fmax(curves: ProteinCurves, protein_weights: np.ndarray) -> np.ndarray:
    """CAFA Fmax under protein weights of shape (n_weightings, n_input_proteins), e.g. bootstrap counts.

    Precision averages over weighted proteins with a prediction at each threshold, recall over all
    weighted proteins with ground truth, exactly as in `cafa_fmax`.
    """
    weights = np.atleast_2d(protein_weights)[:, curves.rows].astype(np.float64)
    n_predicted = weights @ curves.has_prediction
    precision = np.divide(weights @ curves.precision, n_predicted, out=np.zeros_like(n_predicted), where=n_predicted > 0)
    recall = (weights @ curves.recall) / weights.sum(axis=1, keepdims=True)
    f_score = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    return f_score.max(axis=1)


def cafa_fmax(
    targets: np.ndarray,
    probabilities: np.ndarray,
    terms: list[str],
    ontology: GeneOntology,
    ia_weights: dict[str, float],
    *,
    truth_ia_totals: dict[str, np.ndarray] | None = None,
    threshold_step: float = 0.01,
    propagate: bool = True,
    max_predictions: int | None = 1500,
    include_aspects: set[str] | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Compute CAFA-normalized, IA-weighted protein-centric Fmax by namespace."""
    if targets.shape != probabilities.shape or targets.shape[1] != len(terms):
        raise ValueError("Targets, probabilities, and terms have incompatible shapes")
    if propagate:
        limited_probabilities = limit_top_predictions(probabilities, max_predictions)
        propagated_probabilities = ontology.propagate_scores(limited_probabilities, terms, copy=False)
    else:
        propagated_probabilities = probabilities
    curves = protein_fmax_curves(
        targets,
        propagated_probabilities,
        terms,
        ontology,
        ia_weights,
        truth_ia_totals=truth_ia_totals,
        threshold_step=threshold_step,
        include_aspects=include_aspects,
    )
    curve_rows = []
    summary: dict[str, float | int] = {}
    aspect_fmax = []
    for aspect, aspect_curves in curves.items():
        namespace = ASPECT_TO_NAMESPACE[aspect]
        n_ground_truth = len(aspect_curves.rows)
        n_predicted_by_threshold = aspect_curves.has_prediction.sum(axis=0)
        precision_by_threshold = np.divide(
            aspect_curves.precision.sum(axis=0),
            n_predicted_by_threshold,
            out=np.zeros(len(aspect_curves.thresholds), dtype=np.float64),
            where=n_predicted_by_threshold > 0,
        )
        recall_by_threshold = aspect_curves.recall.mean(axis=0)
        f_by_threshold = np.divide(
            2.0 * precision_by_threshold * recall_by_threshold,
            precision_by_threshold + recall_by_threshold,
            out=np.zeros(len(aspect_curves.thresholds), dtype=np.float64),
            where=(precision_by_threshold + recall_by_threshold) > 0,
        )
        for threshold_index, threshold in enumerate(aspect_curves.thresholds):
            n_predicted = int(n_predicted_by_threshold[threshold_index])
            curve_rows.append(
                {
                    "namespace": namespace,
                    "aspect": aspect,
                    "threshold": float(threshold),
                    "n_ground_truth": n_ground_truth,
                    "n_predicted": n_predicted,
                    "coverage": n_predicted / n_ground_truth,
                    "weighted_precision": float(precision_by_threshold[threshold_index]),
                    "weighted_recall": float(recall_by_threshold[threshold_index]),
                    "weighted_f": float(f_by_threshold[threshold_index]),
                }
            )

        namespace_curve = curve_rows[-len(aspect_curves.thresholds):]
        best = max(namespace_curve, key=lambda row: row["weighted_f"])
        summary[f"cafa_fmax_{aspect}"] = best["weighted_f"]
        summary[f"cafa_threshold_{aspect}"] = best["threshold"]
        summary[f"cafa_precision_{aspect}"] = best["weighted_precision"]
        summary[f"cafa_recall_{aspect}"] = best["weighted_recall"]
        summary[f"cafa_n_ground_truth_{aspect}"] = n_ground_truth
        aspect_fmax.append(best["weighted_f"])

    summary["cafa_fmax_mean"] = float(np.mean(aspect_fmax)) if aspect_fmax else float("nan")
    return summary, pd.DataFrame(curve_rows)


def roc_auc_summary(
    targets: np.ndarray,
    scores: np.ndarray,
    aspects: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Per-term ROC-AUC, summarised as macro (mean over terms) and micro (pooled pairs), overall and per aspect.

    Terms without both a positive and a negative are skipped. `mask` excludes individual
    protein-term pairs, e.g. proteins without truth in a term's aspect or annotations already
    known in the partial-knowledge setting.
    """
    if targets.shape != scores.shape or targets.shape[1] != len(aspects):
        raise ValueError("Targets, scores, and aspects have incompatible shapes")
    mask = np.ones(targets.shape, dtype=bool) if mask is None else mask.astype(bool, copy=False)
    per_term = np.full(targets.shape[1], np.nan)
    for index in range(targets.shape[1]):
        rows = mask[:, index]
        term_targets = targets[rows, index]
        positives = int(term_targets.sum())
        if 0 < positives < len(term_targets):
            per_term[index] = roc_auc_score(term_targets, scores[rows, index])

    summary: dict[str, float | int] = {}
    for suffix, columns in (("", np.ones(len(aspects), dtype=bool)), *((f"_{a}", aspects == a) for a in ("P", "F", "C"))):
        term_aucs = per_term[columns]
        term_aucs = term_aucs[np.isfinite(term_aucs)]
        summary[f"macro_roc_auc{suffix}"] = float(term_aucs.mean()) if len(term_aucs) else float("nan")
        summary[f"n_terms_with_roc_auc{suffix}"] = len(term_aucs)
        pair_mask = mask[:, columns]
        pooled_targets = targets[:, columns][pair_mask]
        pooled_scores = scores[:, columns][pair_mask]
        has_both = 0 < int(pooled_targets.sum()) < len(pooled_targets)
        summary[f"micro_roc_auc{suffix}"] = float(roc_auc_score(pooled_targets, pooled_scores)) if has_both else float("nan")
    return summary, per_term
