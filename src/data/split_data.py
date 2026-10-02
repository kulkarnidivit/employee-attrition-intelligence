"""Create the stratified train/test split.

The test set is a locked hold-out: it is used only for final evaluation and must
not influence EDA, feature engineering or model-selection decisions.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
)

logger = logging.getLogger(__name__)


def split_train_test(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test), stratified by the target column.

    Raises:
        ValueError: if the target column is missing.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
        shuffle=True,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_path: Path | str = TRAIN_DATA_PATH,
    test_path: Path | str = TEST_DATA_PATH,
) -> None:
    """Write both splits to CSV, creating folders if needed."""
    for df, path in ((train, Path(train_path)), (test, Path(test_path))):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if not PROCESSED_DATA_PATH.exists():
        logger.error(
            "Cleaned data not found at %s. Run: python -m src.data.clean_data", PROCESSED_DATA_PATH
        )
        sys.exit(1)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    train, test = split_train_test(df)
    save_splits(train, test)

    # Integrity check only: sizes and class balance, nothing about feature values.
    print()
    for name, part in (("train", train), ("test", test)):
        print(f"{name}: shape={part.shape}, {TARGET_COL}={part[TARGET_COL].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
