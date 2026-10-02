"""Clean the raw HR dataset and save the result to data/processed/.

Cleaning is deliberately conservative: it normalizes text, validates, and drops
columns known to carry no information. It does not delete rows or outliers.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_PATH, TARGET_COL
from src.data import schema
from src.data.load_data import load_raw_data
from src.data.validate_data import DataValidationError, raise_if_invalid, validate_raw_data

logger = logging.getLogger(__name__)


def strip_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading/trailing whitespace from all text columns (returns a copy)."""
    df = df.copy()
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].str.strip()
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the columns listed in schema.CONSTANT_COLUMNS, after checking they really are constant."""
    to_drop = [c for c in schema.CONSTANT_COLUMNS if c in df.columns]
    for col in to_drop:
        n_distinct = df[col].nunique(dropna=False)
        if n_distinct != 1:
            raise ValueError(f"'{col}' is listed as constant but has {n_distinct} distinct values")
    logger.info("Dropped constant columns: %s", to_drop)
    return df.drop(columns=to_drop)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataset. Raises DataValidationError on invalid data."""
    df = strip_text_columns(df)
    raise_if_invalid(validate_raw_data(df))
    df = drop_constant_columns(df)
    return df.reset_index(drop=True)


def save_clean_data(df: pd.DataFrame, path: Path | str = PROCESSED_DATA_PATH) -> Path:
    """Write the cleaned dataset to CSV, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data to %s", path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    raw = load_raw_data()
    try:
        clean = clean_data(raw)
    except DataValidationError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    save_clean_data(clean)

    print(f"\nShape before: {raw.shape}")
    print(f"Shape after:  {clean.shape}")
    print(f"Target before: {raw[TARGET_COL].value_counts().to_dict()}")
    print(f"Target after:  {clean[TARGET_COL].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
