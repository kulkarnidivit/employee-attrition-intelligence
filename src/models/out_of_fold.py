"""Honest (out-of-fold) predicted probabilities for every training row."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import CV_N_REPEATS, CV_N_SPLITS, RANDOM_STATE


def get_oof_probabilities(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = CV_N_REPEATS,
    n_splits: int = CV_N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Predicted probability of leaving for every row, from models that never saw that row.

    Runs stratified K-fold n_repeats times with different shuffles and averages the
    out-of-fold probabilities, which makes them steadier than a single run.
    """
    total = np.zeros(len(y))
    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        total += proba
    return total / n_repeats
