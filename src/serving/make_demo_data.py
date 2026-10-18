"""Build data/demo/employees.csv: cleaned synthetic data, sensitive columns removed, training rows flagged."""

import pandas as pd

from src.config import PROCESSED_DATA_PATH, TRAIN_DATA_PATH
from src.data.schema import SENSITIVE_COLUMNS
from src.serving.demo_data import DEMO_DATA_PATH


def main() -> None:
    clean = pd.read_csv(PROCESSED_DATA_PATH)
    train_ids = set(pd.read_csv(TRAIN_DATA_PATH, usecols=["EmployeeNumber"])["EmployeeNumber"])
    clean["in_training"] = clean["EmployeeNumber"].isin(train_ids)
    clean = clean.drop(columns=SENSITIVE_COLUMNS)
    DEMO_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(DEMO_DATA_PATH, index=False)
    print(f"Wrote {len(clean)} rows to {DEMO_DATA_PATH}")


if __name__ == "__main__":
    main()
