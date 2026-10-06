"""Validate engineered features on the TRAINING set only and summarize their attrition patterns."""

import logging
import sys

import pandas as pd

from src.analysis.attrition_rates import attrition_rate_table
from src.analysis.correlation import target_correlations
from src.config import TARGET_COL, TRAIN_DATA_PATH
from src.features.engineering import ENGINEERED_FEATURES, AttritionFeatureEngineer, find_feature_problems


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    train = pd.read_csv(TRAIN_DATA_PATH)
    features = AttritionFeatureEngineer().fit_transform(train.drop(columns=[TARGET_COL]))[ENGINEERED_FEATURES]

    print("\n=== FEATURE PROBLEMS ===")
    problems = find_feature_problems(features)
    print("none" if not problems else "\n".join(problems))

    print("\n=== SUMMARY (min / mean / max) ===")
    print(features.describe().T[["min", "mean", "max"]].round(3).to_string())

    df = features.copy()
    df[TARGET_COL] = train[TARGET_COL].to_numpy()

    print("\n=== ATTRITION RATE BY FEATURE (training set) ===")
    for col in ENGINEERED_FEATURES:
        table = attrition_rate_table(df, col)
        parts = [f"{r.group}: {r.attrition_rate:.2f} (n={r.n})" for r in table.itertuples()]
        print(f"{col}\n    " + " | ".join(parts))

    print("\n=== SPEARMAN CORRELATION WITH ATTRITION ===")
    print(target_correlations(df, ENGINEERED_FEATURES).round(3).to_string())

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
