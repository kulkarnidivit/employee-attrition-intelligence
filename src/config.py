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
