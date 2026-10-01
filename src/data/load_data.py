"""Load the raw HR attrition dataset."""

import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COL

logger = logging.getLogger(__name__)


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and run minimal sanity checks.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the target column is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Download the IBM HR Analytics dataset from Kaggle and save it as data/raw/hr_attrition.csv."
        )

    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns from %s", df.shape[0], df.shape[1], path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}")

    return df
