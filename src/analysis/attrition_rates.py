"""Attrition rate tables with uncertainty, used by EDA and later by the dashboard."""

import numpy as np
import pandas as pd

from src.config import TARGET_COL


def wilson_interval(successes, n, z: float = 1.96):
    """95% Wilson score interval for a proportion (robust for small groups)."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def attrition_rate_table(
    df: pd.DataFrame,
    column: str,
    target_col: str = TARGET_COL,
    positive_label: str = "Yes",
    n_bins: int = 5,
    max_discrete: int = 10,
) -> pd.DataFrame:
    """Return the attrition rate (with 95% interval) for each group of `column`.

    Numeric columns with more than `max_discrete` distinct values are split into
    quantile bins; all other columns use their own values as groups.

    Columns: group, n, leavers, attrition_rate, ci_low, ci_high.
    """
    series = df[column]
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > max_discrete:
        groups = pd.qcut(series, q=n_bins, duplicates="drop", precision=0)
    else:
        groups = series

    is_leaver = df[target_col] == positive_label
    table = is_leaver.groupby(groups, observed=True).agg(n="size", leavers="sum").reset_index()
    table = table.rename(columns={table.columns[0]: "group"})
    table["group"] = table["group"].astype(str)
    table["attrition_rate"] = table["leavers"] / table["n"]
    table["ci_low"], table["ci_high"] = wilson_interval(table["leavers"], table["n"])
    return table
