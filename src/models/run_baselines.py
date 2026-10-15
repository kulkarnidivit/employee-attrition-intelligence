"""Run baseline models through the project's cross-validation and save the results.

Phase 6 will add Logistic Regression, Random Forest and XGBoost to get_baseline_models().
"""

import logging

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.config import REPORTS_DIR
from src.features.preprocessing import build_pipeline
from src.models.cross_validation import PRIMARY_METRIC, SCORING, run_cross_validation, summarize
from src.models.data import load_train_xy

logger = logging.getLogger(__name__)


def get_baseline_models() -> dict:
    """The models to compare, each one a full pipeline."""
    return {
        "dummy_nobody_leaves": build_pipeline(DummyClassifier(strategy="prior"), scale_numeric=True),
        "dummy_everybody_leaves": build_pipeline(
            DummyClassifier(strategy="constant", constant=1), scale_numeric=True
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    X, y = load_train_xy()

    summary_rows, fold_tables = [], []
    for name, model in get_baseline_models().items():
        logger.info("Cross-validating %s ...", name)
        scores = run_cross_validation(model, X, y)
        for metric, row in summarize(scores).iterrows():
            summary_rows.append({"model": name, "metric": metric, "mean": row["mean"], "std": row["std"]})
        fold_tables.append(scores.assign(model=name))

    summary = pd.DataFrame(summary_rows)
    order = list(SCORING)
    means = summary.pivot(index="model", columns="metric", values="mean")[order]
    stds = summary.pivot(index="model", columns="metric", values="std")[order]

    print("\n=== MEAN OVER ALL FOLDS ===")
    print(means.round(3).to_string())
    print("\n=== STANDARD DEVIATION OVER ALL FOLDS ===")
    print(stds.round(3).to_string())
    print(f"\nPrimary metric: {PRIMARY_METRIC}. Share of leavers in the training set: {y.mean():.3f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "baseline_cv_summary.csv", index=False)
    pd.concat(fold_tables).to_csv(REPORTS_DIR / "baseline_cv_folds.csv", index=False)
    logger.info("Saved results to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
