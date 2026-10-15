"""Load features (X) and target (y) for model training.

Only the training file is exposed on purpose. The locked test set gets its own
loader in the final evaluation step.
"""

from pathlib import Path

import pandas as pd

from src.config import TARGET_COL, TRAIN_DATA_PATH


def split_xy(df: pd.DataFrame, target_col: str = TARGET_COL) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned DataFrame into features X and a 0/1 target y (1 = employee left)."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    if not df[target_col].isin(["Yes", "No"]).all():
        raise ValueError(f"'{target_col}' must contain only 'Yes' or 'No'")
    y = (df[target_col] == "Yes").astype(int)
    X = df.drop(columns=[target_col])
    return X, y


def load_xy(path: Path | str) -> tuple[pd.DataFrame, pd.Series]:
    """Read a CSV and return (X, y)."""
    return split_xy(pd.read_csv(path))


def load_train_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Load the training set as (X, y)."""
    if not Path(TRAIN_DATA_PATH).exists():
        raise FileNotFoundError(f"{TRAIN_DATA_PATH} not found. Run: python -m src.data.split_data")
    return load_xy(TRAIN_DATA_PATH)
