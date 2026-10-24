"""Train one XGBoost pipeline per top-N GO term and record CV/hold-out results.

Each GO term gets its own binary classifier over the same merged feature table
(ESM-2 embeddings + biophysical/topology protein features) and the same
train/test protein split, so scores are comparable across terms. The grid
search itself runs on a positive-biased subsample (cheap); the winning
hyperparameters are then refit on the full training set for the reported test
score. Models are not persisted; only grid-search results, best params,
feature importances, and CV/hold-out ROC-AUC are written to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from modeling.data import (
    build_feature_table,
    build_label_matrix,
    load_embedding_table,
    load_term_counts,
    make_train_test_split,
    select_top_terms,
)
from modeling.pipeline import FEATURE_SELECTION_METHODS, build_pipeline, default_param_grid

LOGGER = logging.getLogger("cafa6.modeling")


def _term_dirname(term: str) -> str:
    return term.replace(":", "_")


def _subsample_for_grid_search(y_train: np.ndarray, sample_size: int, min_positive: int, random_state: int) -> np.ndarray:
    """Row indices for the (cheap) grid-search phase: positive-biased so rare terms still get >= min_positive."""
    if len(y_train) <= sample_size:
        return np.arange(len(y_train))
    rng = np.random.RandomState(random_state)
    positive_idx = np.flatnonzero(y_train == 1)
    negative_idx = np.flatnonzero(y_train == 0)
    natural_positive = round(sample_size * len(positive_idx) / len(y_train))
    target_positive = min(len(positive_idx), max(min_positive, natural_positive))
    chosen_positive = positive_idx if target_positive >= len(positive_idx) else rng.choice(positive_idx, size=target_positive, replace=False)
    target_negative = min(len(negative_idx), sample_size - len(chosen_positive))
    chosen_negative = rng.choice(negative_idx, size=target_negative, replace=False)
    indices = np.concatenate([chosen_positive, chosen_negative])
    rng.shuffle(indices)
    return indices


def _feature_importances(pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Map classifier importances back to original feature names, honoring any selector mask."""
    selector = pipeline.named_steps["selector"]
    if selector == "passthrough":
        selected_names = feature_names
    else:
        mask = selector.get_support()
        selected_names = [name for name, keep in zip(feature_names, mask) if keep]
    importances = pipeline.named_steps["classifier"].feature_importances_
    return (
        pd.DataFrame({"feature": selected_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _aggregate_feature_importances(terms_dir: Path, n_models_total: int) -> pd.DataFrame:
    """Mean/max importance per feature across all per-term models that selected it."""
    per_term_tables = [
        pd.read_csv(path, sep="\t") for path in sorted(terms_dir.glob("*/feature_importances.tsv"))
    ]
    if not per_term_tables:
        return pd.DataFrame(columns=["feature", "mean_importance", "max_importance", "n_models_selected", "n_models_total", "selection_rate"])
    combined = pd.concat(per_term_tables, ignore_index=True)
    aggregated = combined.groupby("feature")["importance"].agg(mean_importance="mean", max_importance="max", n_models_selected="count")
    aggregated["n_models_total"] = n_models_total
    aggregated["selection_rate"] = aggregated["n_models_selected"] / n_models_total
    return aggregated.sort_values("mean_importance", ascending=False).reset_index()


def _fit_one_term(
    term: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    feature_selection: str,
    max_features: int,
    selection_threshold: float,
    n_estimators: int,
    cv: int,
    scoring: str,
    grid_search_sample_size: int,
    grid_search_min_positive: int,
    random_state: int,
    terms_dir: Path,
) -> dict:
    """Fit one term and write its artifacts to disk immediately (not batched until the whole run finishes)."""
    term_dir = terms_dir / _term_dirname(term)
    term_dir.mkdir(parents=True, exist_ok=True)
    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    n_pos_test = int(y_test.sum())

    summary: dict = {
        "term": term,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_pos_train": n_pos_train,
        "n_pos_test": n_pos_test,
        "prevalence_train": n_pos_train / len(y_train),
        "prevalence_test": n_pos_test / len(y_test),
        "n_features_total": x_train.shape[1],
    }

    min_class_count = min(n_pos_train, n_neg_train)
    effective_cv = min(cv, min_class_count)
    if effective_cv < 2 or n_pos_test == 0:
        summary.update(
            {
                "status": "skipped_insufficient_positives",
                "cv_folds_used": effective_cv,
                "cv_best_score": np.nan,
                "cv_scoring": scoring,
                "best_params": json.dumps({}),
                "n_features_selected": np.nan,
                "grid_search_sample_size": np.nan,
                "grid_search_n_pos": np.nan,
                "test_roc_auc": np.nan,
            }
        )
        (term_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    grid_idx = _subsample_for_grid_search(y_train, grid_search_sample_size, grid_search_min_positive, random_state)
    x_grid, y_grid = x_train[grid_idx], y_train[grid_idx]
    grid_min_class_count = min(int(y_grid.sum()), len(y_grid) - int(y_grid.sum()))
    grid_cv = min(effective_cv, grid_min_class_count)

    base_pipeline = build_pipeline(
        feature_selection=feature_selection,
        max_features=max_features,
        random_state=random_state,
        n_estimators=n_estimators,
        selection_threshold=selection_threshold,
    )
    param_grid = default_param_grid()

    splitter = StratifiedKFold(n_splits=grid_cv, shuffle=True, random_state=random_state)
    # n_jobs=1: parallelism happens across GO terms (see run()), not within a single term's grid search,
    # otherwise the last few combos of every term's grid leave cores idle before the next term can start.
    grid = GridSearchCV(
        base_pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=splitter,
        n_jobs=1,
        refit=False,
    )
    grid.fit(x_grid, y_grid)

    # Grid search picks hyperparameters on the cheap subsample; the reported model is refit on full training data.
    final_pipeline = clone(base_pipeline).set_params(**grid.best_params_)
    final_pipeline.fit(x_train, y_train)
    test_probabilities = final_pipeline.predict_proba(x_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, test_probabilities)
    print(f"{term}\ttest_roc_auc={test_roc_auc:.4f}\tcv_best={grid.best_score_:.4f}\tn_pos_train={n_pos_train}", flush=True)

    n_features_selected = x_train.shape[1]
    selector = final_pipeline.named_steps["selector"]
    if selector != "passthrough":
        n_features_selected = int(selector.get_support().sum())

    summary.update(
        {
            "status": "ok",
            "cv_folds_used": grid_cv,
            "cv_best_score": grid.best_score_,
            "cv_scoring": scoring,
            "best_params": json.dumps(grid.best_params_),
            "n_features_selected": n_features_selected,
            "grid_search_sample_size": len(grid_idx),
            "grid_search_n_pos": int(y_grid.sum()),
            "test_roc_auc": test_roc_auc,
        }
    )

    cv_results_frame = pd.DataFrame(grid.cv_results_)
    cv_fold_scores = [
        float(cv_results_frame.loc[grid.best_index_, f"split{fold}_test_score"]) for fold in range(grid_cv)
    ]

    cv_results_frame.to_csv(term_dir / "cv_results.tsv", sep="\t", index=False)
    _feature_importances(final_pipeline, feature_names).to_csv(term_dir / "feature_importances.tsv", sep="\t", index=False)
    (term_dir / "best_params.json").write_text(json.dumps(grid.best_params_, indent=2) + "\n")
    (term_dir / "scores.json").write_text(
        json.dumps(
            {
                "cv_scoring": scoring,
                "cv_fold_scores": cv_fold_scores,
                "cv_mean_score": float(grid.best_score_),
                "test_roc_auc": float(test_roc_auc),
            },
            indent=2,
        )
        + "\n"
    )
    (term_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run(
    embeddings_dir: Path,
    protein_features_path: Path,
    train_terms_path: Path,
    output_dir: Path,
    top_n: int,
    test_size: float,
    random_state: int,
    feature_selection: str,
    max_features: int,
    selection_threshold: float,
    n_estimators: int,
    cv: int,
    scoring: str,
    grid_search_sample_size: int,
    grid_search_min_positive: int,
    n_jobs: int,
    limit_terms: int | None,
    taxonomy_features_path: Path | None,
    promoter_nt_dir: Path | None,
    promoter_agront_dir: Path | None,
    promoter_feature_cache: Path,
    promoter_knn_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    terms_dir = output_dir / "terms"
    terms_dir.mkdir(exist_ok=True)

    promoter_embedding_dirs = None
    if promoter_nt_dir is not None or promoter_agront_dir is not None:
        promoter_embedding_dirs = {}
        if promoter_nt_dir is not None:
            promoter_embedding_dirs["promoter_nt"] = promoter_nt_dir
        if promoter_agront_dir is not None:
            promoter_embedding_dirs["promoter_agront"] = promoter_agront_dir

    # Split before building the feature table (not after, as before) so promoter imputation can
    # restrict its KNN/mean donors to train-set proteins only -- no test-set information leaks in.
    LOGGER.info("computing train/test split")
    all_protein_ids = load_embedding_table(embeddings_dir)["protein_id"]
    train_ids, test_ids = make_train_test_split(pd.Index(all_protein_ids), test_size=test_size, random_state=random_state)
    LOGGER.info("split: %d train / %d test proteins (shared across all GO terms)", len(train_ids), len(test_ids))

    LOGGER.info("loading feature table")
    features = build_feature_table(
        embeddings_dir,
        protein_features_path,
        taxonomy_features_path,
        promoter_embedding_dirs=promoter_embedding_dirs,
        promoter_feature_cache_path=promoter_feature_cache if promoter_embedding_dirs else None,
        promoter_knn_k=promoter_knn_k,
        promoter_train_ids=train_ids.tolist() if promoter_embedding_dirs else None,
    )

    LOGGER.info("ranking GO terms by prevalence")
    term_counts = load_term_counts(train_terms_path)
    terms = select_top_terms(term_counts, top_n=top_n)
    if limit_terms is not None:
        terms = terms[:limit_terms]

    labels = build_label_matrix(train_terms_path, features.index, terms)

    # Resume support: a term is already done if a prior run wrote its summary.json (the last
    # artifact _fit_one_term writes). Skip refitting it and reuse the saved summary instead.
    already_done_summaries = []
    terms_to_fit = []
    for term in terms:
        summary_path = terms_dir / _term_dirname(term) / "summary.json"
        if summary_path.exists():
            already_done_summaries.append(json.loads(summary_path.read_text()))
        else:
            terms_to_fit.append(term)
    if already_done_summaries:
        LOGGER.info(
            "resume: %d/%d terms already have saved results, fitting the remaining %d",
            len(already_done_summaries), len(terms), len(terms_to_fit),
        )

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    pd.Series(train_ids, name="protein_id").to_csv(splits_dir / "train_protein_ids.tsv", sep="\t", index=False)
    pd.Series(test_ids, name="protein_id").to_csv(splits_dir / "test_protein_ids.tsv", sep="\t", index=False)

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "embeddings_dir": str(embeddings_dir),
                "protein_features_path": str(protein_features_path),
                "taxonomy_features_path": str(taxonomy_features_path) if taxonomy_features_path else None,
                "train_terms_path": str(train_terms_path),
                "top_n": top_n,
                "test_size": test_size,
                "random_state": random_state,
                "feature_selection": feature_selection,
                "max_features": max_features,
                "selection_threshold": selection_threshold,
                "n_estimators": n_estimators,
                "cv": cv,
                "scoring": scoring,
                "grid_search_sample_size": grid_search_sample_size,
                "grid_search_min_positive": grid_search_min_positive,
                "n_terms_run": len(terms),
            },
            indent=2,
        )
        + "\n"
    )

    # Converted to plain ndarrays (rather than passed as DataFrames) so joblib can memory-map the
    # shared feature table once instead of re-pickling it for every dispatched GO-term task.
    feature_names = list(features.columns)
    x_train_array = features.loc[train_ids].to_numpy()
    x_test_array = features.loc[test_ids].to_numpy()

    LOGGER.info("fitting %d terms across up to %d parallel workers (one GridSearchCV per term)", len(terms_to_fit), n_jobs)
    # return_as="generator_unordered": summaries stream back in completion order (not submission order)
    # as each term finishes, so summary.tsv reflects live progress on multi-hour runs.
    results = Parallel(n_jobs=n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(_fit_one_term)(
            term=term,
            x_train=x_train_array,
            x_test=x_test_array,
            y_train=labels.loc[train_ids, term].to_numpy(),
            y_test=labels.loc[test_ids, term].to_numpy(),
            feature_names=feature_names,
            feature_selection=feature_selection,
            max_features=max_features,
            selection_threshold=selection_threshold,
            n_estimators=n_estimators,
            cv=cv,
            scoring=scoring,
            grid_search_sample_size=grid_search_sample_size,
            grid_search_min_positive=grid_search_min_positive,
            random_state=random_state,
            terms_dir=terms_dir,
        )
        for term in terms_to_fit
    )

    summaries = list(already_done_summaries)
    for summary in results:
        summaries.append(summary)
        if summary["status"] != "ok":
            LOGGER.warning("skipped %s: insufficient positives for %d-fold CV or empty test set", summary["term"], cv)
        pd.DataFrame(summaries).sort_values("cv_best_score", ascending=False).to_csv(output_dir / "summary.tsv", sep="\t", index=False)

    LOGGER.info("wrote summary for %d terms to %s", len(summaries), output_dir / "summary.tsv")

    importance_summary = _aggregate_feature_importances(terms_dir, n_models_total=len(terms))
    importance_summary.to_csv(output_dir / "feature_importance_summary.tsv", sep="\t", index=False)
    LOGGER.info("wrote mean/max feature importances across %d models to %s", len(terms), output_dir / "feature_importance_summary.tsv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/features/embeddings/train_esm2_35M"))
    parser.add_argument("--protein-features", type=Path, default=Path("data/features/sequence/train_features.tsv"))
    parser.add_argument(
        "--taxonomy-features",
        type=Path,
        default=Path("data/features/taxonomy/train_taxonomy_features.tsv"),
        help="One-hot taxonomy feature table; pass a nonexistent path or empty string to skip",
    )
    parser.add_argument(
        "--promoter-nt-dir",
        type=Path,
        default=Path("data/features/embeddings/train_promoter_nt100m"),
        help="NT-v2 promoter embedding dir (all species); pass a nonexistent path to skip promoter features entirely",
    )
    parser.add_argument(
        "--promoter-agront-dir",
        type=Path,
        default=Path("data/features/embeddings/train_promoter_agront1b"),
        help="AgroNT promoter embedding dir (plant species only, zero-filled elsewhere); pass a nonexistent path to skip",
    )
    parser.add_argument(
        "--promoter-feature-cache",
        type=Path,
        default=Path("data/features/promoters/imputed_promoter_features.csv"),
        help="Cached (CSV) complete promoter feature table -- delete to force a rebuild after re-extracting/re-embedding",
    )
    parser.add_argument(
        "--promoter-knn-k",
        type=int,
        default=5,
        help="Neighbors used to impute promoter features for proteins whose species has no promoter data",
    )
    parser.add_argument("--train-terms", type=Path, default=Path("data/Train/train_terms.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/go_term_xgboost"))
    parser.add_argument("--top-n", type=int, default=100, help="Number of most prevalent GO terms to model")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--feature-selection", choices=FEATURE_SELECTION_METHODS, default="lightgbm")
    parser.add_argument("--max-features", type=int, default=250)
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=-np.inf,
        help="Minimum feature importance to keep a feature, on top of the --max-features cap (default: no threshold)",
    )
    parser.add_argument("--n-estimators", type=int, default=500, help="Fixed XGBoost tree count (not searched by GridSearchCV)")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--scoring", default="roc_auc", help="GridSearchCV scoring; roc_auc is comparable across term prevalence")
    parser.add_argument(
        "--grid-search-sample-size",
        type=int,
        default=10_000,
        help="Row subsample used for the grid search only; final model is refit on full training data",
    )
    parser.add_argument(
        "--grid-search-min-positive",
        type=int,
        default=500,
        help="Minimum positives to bias into the grid-search subsample (or all, if a term has fewer)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="GO terms processed concurrently (each term's GridSearchCV runs single-threaded); -1 uses all cores",
    )
    parser.add_argument("--limit-terms", type=int, default=None, help="Debug: only run the first N of the top-n terms")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    arguments = vars(parser.parse_args())
    log_level = arguments.pop("log_level")
    logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    taxonomy_features_path = arguments["taxonomy_features"]
    if taxonomy_features_path is not None and not taxonomy_features_path.exists():
        LOGGER.warning("taxonomy features not found at %s; proceeding without them", taxonomy_features_path)
        taxonomy_features_path = None
    promoter_nt_dir = arguments["promoter_nt_dir"]
    if promoter_nt_dir is not None and not (promoter_nt_dir / "metadata.json").exists():
        LOGGER.warning("promoter NT-v2 embeddings not found at %s; proceeding without them", promoter_nt_dir)
        promoter_nt_dir = None
    promoter_agront_dir = arguments["promoter_agront_dir"]
    if promoter_agront_dir is not None and not (promoter_agront_dir / "metadata.json").exists():
        LOGGER.warning("promoter AgroNT embeddings not found at %s; proceeding without them", promoter_agront_dir)
        promoter_agront_dir = None
    run(
        embeddings_dir=arguments["embeddings_dir"],
        protein_features_path=arguments["protein_features"],
        train_terms_path=arguments["train_terms"],
        output_dir=arguments["output_dir"],
        top_n=arguments["top_n"],
        test_size=arguments["test_size"],
        random_state=arguments["random_state"],
        feature_selection=arguments["feature_selection"],
        max_features=arguments["max_features"],
        selection_threshold=arguments["selection_threshold"],
        n_estimators=arguments["n_estimators"],
        cv=arguments["cv"],
        scoring=arguments["scoring"],
        grid_search_sample_size=arguments["grid_search_sample_size"],
        grid_search_min_positive=arguments["grid_search_min_positive"],
        n_jobs=arguments["n_jobs"],
        limit_terms=arguments["limit_terms"],
        taxonomy_features_path=taxonomy_features_path,
        promoter_nt_dir=promoter_nt_dir,
        promoter_agront_dir=promoter_agront_dir,
        promoter_feature_cache=arguments["promoter_feature_cache"],
        promoter_knn_k=arguments["promoter_knn_k"],
    )


if __name__ == "__main__":
    main()
