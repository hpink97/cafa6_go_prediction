"""Blending the network with the score channels (taxon prior, ESM-2 KNN): per-aspect weights are
tuned on validation by CAFA Fmax, then applied unchanged to test or prediction scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.evaluation.go import GeneOntology, cafa_fmax, limit_top_predictions
from modeling.nn.score_channels import blend_scores_by_aspect

def evaluate_cafa_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    terms: list[str],
    ontology: GeneOntology,
    ia_weights: dict[str, float],
    truth_ia_totals: dict[str, np.ndarray],
    threshold_step: float,
    max_predictions: int,
    include_aspects: set[str] | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame, np.ndarray]:
    propagated = ontology.propagate_scores(
        limit_top_predictions(probabilities, max_predictions),
        terms,
        copy=False,
    )
    summary, curve = cafa_fmax(
        targets,
        propagated,
        terms,
        ontology,
        ia_weights,
        truth_ia_totals=truth_ia_totals,
        threshold_step=threshold_step,
        propagate=False,
        include_aspects=include_aspects,
    )
    return summary, curve, propagated


def apply_channel_selection(
    base_scores: np.ndarray,
    channel_variants: dict[str, np.ndarray],
    aspects: np.ndarray,
    selection: dict[str, dict[str, float | str]],
) -> tuple[np.ndarray, np.ndarray]:
    blended = base_scores.copy()
    selected_channel = np.zeros_like(base_scores)
    for aspect, settings in selection.items():
        columns = aspects == aspect
        variant = str(settings["variant"])
        weight = float(settings["weight"])
        channel = channel_variants[variant]
        blended[:, columns] = (
            (1.0 - weight) * blended[:, columns]
            + weight * channel[:, columns]
        )
        selected_channel[:, columns] = channel[:, columns]
    return blended, selected_channel


def tune_channel_by_aspect(
    base_scores: np.ndarray,
    channel_variants: dict[str, np.ndarray],
    aspects: np.ndarray,
    targets: np.ndarray,
    terms: list[str],
    ontology: GeneOntology,
    ia_weights: dict[str, float],
    truth_ia_totals: dict[str, np.ndarray],
    threshold_step: float,
    max_predictions: int,
    weight_step: float,
    max_channel_weight: float,
) -> dict[str, dict[str, float | str]]:
    if not 0.0 < weight_step <= 1.0:
        raise ValueError("ensemble_weight_step must be in (0, 1]")
    if not 0.0 <= max_channel_weight <= 1.0:
        raise ValueError("ensemble_max_channel_weight must be in [0, 1]")
    weights = np.arange(0.0, max_channel_weight + weight_step / 2.0, weight_step)
    best = {
        aspect: {"variant": next(iter(channel_variants)), "weight": 0.0, "validation_fmax": -np.inf}
        for aspect in ("P", "F", "C")
    }
    for variant, channel_scores in channel_variants.items():
        for weight in weights:
            candidate = blend_scores_by_aspect(
                base_scores,
                channel_scores,
                aspects,
                {aspect: float(weight) for aspect in ("P", "F", "C")},
            )
            summary, _, _ = evaluate_cafa_predictions(
                targets,
                candidate,
                terms,
                ontology,
                ia_weights,
                truth_ia_totals,
                threshold_step,
                max_predictions,
            )
            for aspect in ("P", "F", "C"):
                aspect_score = float(summary[f"cafa_fmax_{aspect}"])
                if aspect_score > float(best[aspect]["validation_fmax"]):
                    best[aspect] = {
                        "variant": variant,
                        "weight": float(weight),
                        "validation_fmax": aspect_score,
                    }
    return best
