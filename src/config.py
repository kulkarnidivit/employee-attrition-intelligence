"""Central configuration: paths and constants. No other module hardcodes these."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Can be overridden with an environment variable (useful for Docker later).
RAW_DATA_PATH = Path(
    os.getenv("ATTRITION_RAW_DATA_PATH", PROJECT_ROOT / "data" / "raw" / "hr_attrition.csv")
)

TARGET_COL = "Attrition"

PROCESSED_DATA_PATH = Path(
    os.getenv(
        "ATTRITION_PROCESSED_DATA_PATH",
        PROJECT_ROOT / "data" / "processed" / "hr_attrition_clean.csv",
    )
)

# Reproducible train/test split
RANDOM_STATE = 42
TEST_SIZE = 0.2

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv"
