"""Explain the trained multi-label network with SHAP.

SHAP values come from `shap.GradientExplainer` (expected gradients) on one GO term's logit at a
time, against a background of training proteins. Three levels are reported:

    groups      ESM-2 vs handcrafted vs kingdom share of |SHAP| for every GO term
    features    the 56 handcrafted features, for chosen deep-dive terms (beeswarms)
    dimensions  each ESM-2 dimension: which matter, for which terms, and what they correlate with

Group permutation importance (per-term AUC drop when a group is shuffled) complements SHAP in the
same units as the ablation. Neither shap nor sklearn permutes feature groups and scores
ontology-propagated predictions, so that part is a short custom loop.

    cafa6-explain --model-dirs data/final/winner_esm2_prot_tax/seed{0,1,2}/model --out results/explainability
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr
from torch import nn

from modeling.data import build_feature_table
from modeling.go_evaluation import NAMESPACE_TO_ASPECT, GeneOntology, limit_top_predictions, roc_auc_summary
from modeling.predict_multilabel_nn import TrainedModel, load_trained_model
from modeling.term_groups import classify_terms, load_term_names

LOGGER = logging.getLogger("cafa6.modeling.explain")
GROUPS = ("esm2", "handcrafted", "kingdom")
DEEP_DIVE_TERMS = {
    "GO:0009408": "response",
    "GO:0034976": "response",
    "GO:0006979": "response",
    "GO:0003700": "DNA-binding",
    "GO:0003677": "DNA-binding",
    "GO:0006915": "disease-linked",
    "GO:0006954": "disease-linked",
    "GO:0007268": "disease-linked",
    "GO:0006281": "disease-linked",
    "GO:0005886": "sanity check",
    "GO:0005634": "sanity check",
    "GO:0005576": "sanity check",
    "GO:0004672": "sanity check",
}


def feature_groups(feature_names: list[str]) -> np.ndarray:
    """Group label per input column. The network's `manual_taxonomy` branch mixes handcrafted and
    kingdom columns; attribution separates them by column name."""
    names = np.asarray(feature_names, dtype=str)
    return np.where(np.char.startswith(names, "esm2_"), "esm2", np.where(np.char.startswith(names, "kingdom_"), "kingdom", "handcrafted"))


class TermLogit(nn.Module):
    """One GO term's logit, as a single-output model for shap."""

    def __init__(self, model: nn.Module, term_index: int) -> None:
        super().__init__()
        self.model = model
        self.term_index = term_index

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)[:, self.term_index : self.term_index + 1]


@dataclass
class ExplainData:
    """Everything needed to explain one trained seed, on its own held-out test split."""

    trained: TrainedModel
    groups: np.ndarray
    test_ids: np.ndarray
    x_test: np.ndarray  # standardised, as the network sees it
    raw_test: pd.DataFrame  # original units, for colouring plots
    targets: np.ndarray  # test proteins x terms, ancestor-closed labels
    background: torch.Tensor
    terms: pd.DataFrame  # term, aspect, name, train_positive, test_positive
    config: dict


def standardise(trained: TrainedModel, frame: pd.DataFrame) -> np.ndarray:
    """Training-time preprocessing: median-impute non-finite values, then z-score."""
    values = frame.to_numpy(dtype=np.float32).copy()
    rows, columns = np.nonzero(~np.isfinite(values))
    values[rows, columns] = trained.medians[columns]
    return ((values - trained.means) / trained.scales).astype(np.float32)


def load_explain_data(model_dir: Path, background_size: int = 256, seed: int = 0) -> ExplainData:
    config = json.loads((model_dir / "run_config.json").read_text())
    trained = load_trained_model(model_dir, "cpu")
    table = build_feature_table(config["embeddings_dir"], config["protein_features_path"], config["taxonomy_features_path"])
    if list(table.columns) != trained.feature_names:
        raise ValueError(f"Feature layout of {model_dir} does not match the rebuilt feature table")
    with np.load(model_dir / "test_predictions.npz") as predictions:
        test_ids = predictions["protein_ids"].astype(str)
        if [str(term) for term in predictions["terms"]] != trained.terms:
            raise ValueError("Saved test labels and checkpoint terms are in a different order")
        targets = predictions["targets"]
    train_ids = pd.read_csv(model_dir / "splits" / "train_protein_ids.tsv", sep="\t")["protein_id"].to_numpy(str)
    rng = np.random.default_rng(seed)
    background = torch.from_numpy(standardise(trained, table.loc[rng.choice(train_ids, background_size, replace=False)]))
    terms = pd.read_csv(model_dir / "terms.tsv", sep="\t")[["term", "aspect", "train_positive", "test_positive"]]
    terms["name"] = terms["term"].map(load_term_names(config["go_obo_path"]))
    return ExplainData(
        trained=trained,
        groups=feature_groups(trained.feature_names),
        test_ids=test_ids,
        x_test=standardise(trained, table.loc[test_ids]),
        raw_test=table.loc[test_ids],
        targets=targets,
        background=background,
        terms=terms,
        config=config,
    )


def term_shap(data: ExplainData, term_index: int, rows: np.ndarray, nsamples: int, seed: int = 0) -> np.ndarray:
    """SHAP values (proteins x features) for one term's logit on the given test rows."""
    explainer = shap.GradientExplainer(TermLogit(data.trained.model, term_index).eval(), data.background)
    values = explainer.shap_values(torch.from_numpy(data.x_test[rows]), nsamples=nsamples, rseed=seed)
    return np.asarray(values, dtype=np.float32).reshape(len(rows), -1)


def global_attribution(data: ExplainData, max_positives: int = 100, nsamples: int = 50, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Per term: mean group share of |SHAP| over (up to `max_positives`) test positives, and the
    term x feature matrix of mean |SHAP|."""
    rng = np.random.default_rng(seed)
    importance = np.full((len(data.trained.terms), data.x_test.shape[1]), np.nan, dtype=np.float32)
    rows = []
    for term_index, term in enumerate(data.trained.terms):
        positives = np.flatnonzero(data.targets[:, term_index])
        if not len(positives):
            continue
        if len(positives) > max_positives:
            positives = np.sort(rng.choice(positives, max_positives, replace=False))
        magnitude = np.abs(term_shap(data, term_index, positives, nsamples, seed))
        importance[term_index] = magnitude.mean(axis=0)
        totals = magnitude.sum(axis=1, keepdims=True)
        shares = {group: float((magnitude[:, data.groups == group].sum(axis=1, keepdims=True) / totals).mean()) for group in GROUPS}
        rows.append({"term": term, "n_explained": len(positives), **shares})
        if term_index % 200 == 0:
            LOGGER.info("explained %d/%d terms", term_index + 1, len(data.trained.terms))
    return pd.DataFrame(rows), importance


def _propagated_auc(data: ExplainData, ontology: GeneOntology, x: np.ndarray, aspects: np.ndarray, max_predictions: int) -> np.ndarray:
    with torch.inference_mode():
        scores = torch.sigmoid(data.trained.model(torch.from_numpy(x))).numpy()
    propagated = ontology.propagate_scores(limit_top_predictions(scores, max_predictions), data.trained.terms, copy=False)
    return roc_auc_summary(data.targets, propagated, aspects)[1]


def permutation_importance(data: ExplainData, repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """Per-term drop in held-out ROC-AUC when one feature group is shuffled across proteins."""
    ontology = GeneOntology.from_obo(data.config["go_obo_path"])
    aspects = np.asarray([NAMESPACE_TO_ASPECT[ontology.namespaces[term]] for term in data.trained.terms])
    max_predictions = data.config["max_predictions_per_protein"]
    baseline = _propagated_auc(data, ontology, data.x_test, aspects, max_predictions)
    rng = np.random.default_rng(seed)
    result = pd.DataFrame({"term": data.trained.terms, "baseline_auc": baseline})
    for group in GROUPS:
        columns = np.flatnonzero(data.groups == group)
        drops = []
        for _ in range(repeats):
            shuffled = data.x_test.copy()
            shuffled[:, columns] = shuffled[rng.permutation(len(shuffled))][:, columns]
            drops.append(baseline - _propagated_auc(data, ontology, shuffled, aspects, max_predictions))
        result[f"{group}_auc_drop"] = np.mean(drops, axis=0)
    return result


def deep_dive(data: ExplainData, term: str, n_per_class: int = 500, nsamples: int = 200, seed: int = 0) -> shap.Explanation:
    """SHAP for every input feature (ESM-2 dimensions, handcrafted, kingdom) on a term's test
    positives plus as many negatives, so the beeswarm shows what separates them. Feature values are
    kept in original units."""
    term_index = data.trained.terms.index(term)
    rng = np.random.default_rng(seed)
    labels = data.targets[:, term_index]
    rows = []
    for value in (1, 0):
        candidates = np.flatnonzero(labels == value)
        rows.append(rng.choice(candidates, min(n_per_class, len(candidates)), replace=False))
    rows = np.sort(np.concatenate(rows))
    return shap.Explanation(
        values=term_shap(data, term_index, rows, nsamples, seed),
        data=data.raw_test.iloc[rows].to_numpy(dtype=float),
        feature_names=list(data.trained.feature_names),
    )


def dimension_feature_correlation(model_dir: Path, dimensions: list[str], max_proteins: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Spearman correlation between chosen ESM-2 dimensions and every handcrafted feature, over
    training proteins (a seeded subsample keeps it fast)."""
    config = json.loads((model_dir / "run_config.json").read_text())
    table = build_feature_table(config["embeddings_dir"], config["protein_features_path"], None)
    table = table.sample(n=min(max_proteins, len(table)), random_state=seed)
    handcrafted = [column for column in table.columns if feature_groups([column])[0] == "handcrafted"]
    rho = spearmanr(table[dimensions].to_numpy(), table[handcrafted].to_numpy(), nan_policy="omit").statistic
    block = rho[: len(dimensions), len(dimensions) :]
    return pd.DataFrame(block, index=dimensions, columns=handcrafted)


def term_metadata(data: ExplainData) -> pd.DataFrame:
    """Aspect, name, response-term flag and frequency band for every model term."""
    ontology = GeneOntology.from_obo(data.config["go_obo_path"])
    names = load_term_names(data.config["go_obo_path"])
    meta = data.terms.copy()
    response = classify_terms(meta, ontology, names)["by_name"]
    meta["response_term"] = meta["term"].map(response).fillna(False).astype(bool)
    meta["frequency_band"] = pd.qcut(meta["train_positive"].rank(method="first"), 3, labels=["specific (rare)", "intermediate", "general (common)"])
    meta["deep_dive"] = meta["term"].map(DEEP_DIVE_TERMS)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dirs", type=Path, nargs="+", required=True, help="One trained model per seed")
    parser.add_argument("--out", type=Path, default=Path("results/explainability"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/explainability"), help="Per-seed SHAP results, reused on rerun")
    parser.add_argument("--max-positives", type=int, default=100, help="Test positives explained per term in the global pass")
    parser.add_argument("--nsamples", type=int, default=50, help="Expected-gradient samples per protein in the global pass")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from modeling.explain_report import write_explainability_report

    write_explainability_report(args.model_dirs, args.out, args.cache_dir, args.max_positives, args.nsamples)


if __name__ == "__main__":
    main()
