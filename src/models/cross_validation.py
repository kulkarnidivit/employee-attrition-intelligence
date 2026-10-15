"""Cross-validation plan: 5-fold stratified, repeated 3 times, fixed seed.

Primary metric is PR-AUC (average precision); the others are secondary.
"""

import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.config import CV_N_REPEATS, CV_N_SPLITS, RANDOM_STATE

PRIMARY_METRIC = "pr_auc"

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
}


def make_cv() -> RepeatedStratifiedKFold:
    """The project's cross-validation splitter."""
    return RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=RANDOM_STATE
    )


def run_cross_validation(estimator, X: pd.DataFrame, y: pd.Series, cv=None, scoring=None) -> pd.DataFrame:
    """Cross-validate and return one row per fold with one column per metric."""
    cv = cv if cv is not None else make_cv()
    scoring = scoring if scoring is not None else SCORING
    results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, error_score="raise")
    return pd.DataFrame({name: results[f"test_{name}"] for name in scoring})


def summarize(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across the folds."""
    return pd.DataFrame({"mean": fold_scores.mean(), "std": fold_scores.std()})
