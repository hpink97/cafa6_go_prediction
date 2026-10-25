"""Pipeline factory: impute -> scale -> LightGBM feature selection -> XGBoost."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURE_SELECTION_METHODS = ("none", "lightgbm", "logreg")


def _build_selector(method: str, max_features: int, random_state: int, selection_threshold: float = -np.inf):
    if method == "none":
        return "passthrough"
    if method == "lightgbm":
        # LightGBM is the fast default selector estimator (histogram splits, no tree_method tuning needed).
        estimator = LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            n_jobs=1,
            random_state=random_state,
            verbosity=-1,
        )
    elif method == "logreg":
        estimator = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown feature-selection method {method!r}; choose from {FEATURE_SELECTION_METHODS}")
    # threshold=-inf keeps exactly `max_features` (ranked by importance/|coef|) regardless of how
    # low that importance is; pass a real threshold to also drop features that don't clear it, even
    # if fewer than max_features remain.
    return SelectFromModel(estimator=estimator, max_features=max_features, threshold=selection_threshold)


def build_pipeline(
    feature_selection: str = "lightgbm",
    max_features: int = 250,
    random_state: int = 42,
    n_estimators: int = 500,
    selection_threshold: float = -np.inf,
) -> Pipeline:
    """Build the shared preprocessing + classifier pipeline for one GO-term model.

    No `memory=` caching: sklearn's Pipeline cache keys on (X, y) together, and since y differs
    per GO term, nothing is reused across terms — it only wrote large one-off arrays to disk.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", _build_selector(feature_selection, max_features, random_state, selection_threshold)),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=n_estimators,
                    tree_method="hist",
                    n_jobs=1,
                    random_state=random_state,
                    eval_metric="logloss",
                ),
            ),
        ],
    )


def default_param_grid() -> dict[str, list]:
    """Fixed 6-combo grid: n_estimators and the selector are both fixed, not searched."""
    return {
        "classifier__max_depth": [5, 7, 9],
        "classifier__learning_rate": [0.05, 0.1],
    }

