"""Correlation helpers for EDA (rank-based, so robust to skew and ordinal scales)."""

import pandas as pd

from src.config import TARGET_COL


def top_correlated_pairs(df: pd.DataFrame, columns: list[str], threshold: float = 0.6) -> pd.DataFrame:
    """List feature pairs whose absolute Spearman correlation is >= threshold.

    Columns: feature_a, feature_b, spearman. Sorted by absolute correlation, strongest first.
    """
    corr = df[columns].corr(method="spearman")
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            rho = corr.loc[a, b]
            if abs(rho) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "spearman": rho})
    result = pd.DataFrame(rows, columns=["feature_a", "feature_b", "spearman"])
    return result.sort_values("spearman", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def target_correlations(
    df: pd.DataFrame,
    columns: list[str],
    target_col: str = TARGET_COL,
    positive_label: str = "Yes",
) -> pd.Series:
    """Spearman correlation of each column with the binary target (leaver = 1).

    Sorted by absolute value, strongest first. Measures steady up/down association only.
    """
    is_leaver = (df[target_col] == positive_label).astype(int)
    corr = df[columns].corrwith(is_leaver, method="spearman")
    return corr.sort_values(key=lambda s: s.abs(), ascending=False)
