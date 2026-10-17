"""Run the baseline models through the project's cross-validation and save the results.

All baselines use the same pipeline (engineered features, sensitive columns off) and class
weights for the 16% imbalance. Tuning comes later (Phase 8).
"""

import logging

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.config import REPORTS_DIR
from src.features.preprocessing import build_pipeline
from src.models.cross_validation import PRIMARY_METRIC, SCORING, run_cross_validation, summarize
from src.models.data import load_train_xy
from src.models.estimators import make_logistic_regression, make_random_forest, make_xgboost

logger = logging.getLogger(__name__)


def get_baseline_models(scale_pos_weight: float = 5.0) -> dict:
    """The models to compare, each one a full pipeline.

    scale_pos_weight is how much more a leaver counts for XGBoost (about stayers / leavers).
    """
    return {
        "dummy_nobody_leaves": build_pipeline(DummyClassifier(strategy="prior"), scale_numeric=True),
        "dummy_everybody_leaves": build_pipeline(
            DummyClassifier(strategy="constant", constant=1), scale_numeric=True
        ),
        "logistic_regression": build_pipeline(make_logistic_regression("balanced"), scale_numeric=True),
        "random_forest": build_pipeline(make_random_forest("balanced")),
        "xgboost": build_pipeline(make_xgboost(scale_pos_weight)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    X, y = load_train_xy()
    leaver_share = y.mean()
    models = get_baseline_models(scale_pos_weight=(1 - leaver_share) / leaver_share)

    summary_rows, fold_tables = [], []
    for name, model in models.items():
        logger.info("Cross-validating %s ...", name)
        scores = run_cross_validation(model, X, y)
        for metric, row in summarize(scores).iterrows():
            summary_rows.append({"model": name, "metric": metric, "mean": row["mean"], "std": row["std"]})
        fold_tables.append(scores.assign(model=name))

    summary = pd.DataFrame(summary_rows)
    order = list(SCORING)
    means = summary.pivot(index="model", columns="metric", values="mean")[order]
    means = means.sort_values(PRIMARY_METRIC, ascending=False)
    stds = summary.pivot(index="model", columns="metric", values="std")[order].loc[means.index]

    print("\n=== MEAN OVER ALL FOLDS (best PR-AUC first) ===")
    print(means.round(3).to_string())
    print("\n=== STANDARD DEVIATION OVER ALL FOLDS ===")
    print(stds.round(3).to_string())

    print(f"\nPR-AUC compared with the floor (share of leavers = {leaver_share:.3f}):")
    for model, value in means[PRIMARY_METRIC].items():
        print(f"  {model:24s} {value:.3f}  ({value / leaver_share:.2f}x the floor)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "baseline_cv_summary.csv", index=False)
    pd.concat(fold_tables).to_csv(REPORTS_DIR / "baseline_cv_folds.csv", index=False)
    logger.info("Saved results to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
