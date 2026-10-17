"""Evaluation helpers: pooled scores, threshold table, calibration table."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def pooled_scores(y_true, proba) -> dict:
    """PR-AUC and ROC-AUC over all predictions at once."""
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def threshold_table(y_true, proba, thresholds) -> pd.DataFrame:
    """For each cutoff: how many employees get flagged and how well that works.

    caught = leavers flagged, false_alarms = stayers flagged, missed = leavers not flagged.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rows = []
    for threshold in thresholds:
        flagged = proba >= threshold
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = int((~flagged & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": threshold,
                "flagged": int(flagged.sum()),
                "caught": tp,
                "false_alarms": fp,
                "missed": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y_true, proba, n_bins: int = 5) -> pd.DataFrame:
    """Compare predicted probability with the real attrition rate, in equal-sized groups."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    return (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), actual_rate=("y", "mean"))
        .reset_index(drop=True)
    )
