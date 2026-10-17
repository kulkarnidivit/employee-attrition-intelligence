"""Error analysis helpers: who gets flagged, who gets missed, and does it differ by group."""

import numpy as np
import pandas as pd


def flag_top_share(proba, share: float) -> np.ndarray:
    """Flag the top `share` of employees by score (for example 0.16 = the top 16%)."""
    proba = np.asarray(proba)
    cutoff = np.quantile(proba, 1 - share)
    return proba >= cutoff


def group_report(df: pd.DataFrame, group_col: str, y_col: str = "attrition", flagged_col: str = "flagged") -> pd.DataFrame:
    """Per group: size, real attrition rate, share flagged, recall and precision."""
    rows = []
    for group, part in df.groupby(group_col, observed=True):
        leavers = int((part[y_col] == 1).sum())
        flagged = int(part[flagged_col].sum())
        caught = int(((part[y_col] == 1) & part[flagged_col]).sum())
        rows.append(
            {
                group_col: group,
                "n": len(part),
                "leaver_rate": leavers / len(part),
                "flagged_rate": flagged / len(part),
                "recall": caught / leavers if leavers else np.nan,
                "precision": caught / flagged if flagged else np.nan,
            }
        )
    return pd.DataFrame(rows)
