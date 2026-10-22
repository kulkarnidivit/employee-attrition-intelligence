"""Build models/model_card.json: the results the web app shows on its Model performance page.

Uses the saved final-model results, out-of-fold predictions on the TRAINING set, and one scoring pass of
the already-final model over the test set for the calibration chart. Nothing here refits or changes the model.
"""

import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    METADATA_PATH, MODEL_CARD_PATH, MODEL_PATH, REPORTS_DIR, TEST_DATA_PATH, TEST_RESULTS_PATH,
)
from src.data.schema import SENSITIVE_COLUMNS
from src.features.preprocessing import build_pipeline
from src.models.data import load_train_xy, load_xy
from src.models.evaluation import calibration_table
from src.models.out_of_fold import get_oof_probabilities
from src.models.train_final import load_best_C, make_lr
from src.serving.service import risk_level

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
}

NOTES = [
    "The test set was split off before any exploration and used once, after the model and cutoff were chosen.",
    "The cutoff was chosen on out-of-fold training predictions only, never on the test set.",
    "Probabilities are calibrated: class weights alone made the raw scores over-confident.",
    "Age, gender and marital status are not model inputs.",
    "The test set has few leavers, so scores carry noticeable uncertainty.",
]


def main() -> None:
    metadata = json.loads(METADATA_PATH.read_text())
    test_results = json.loads(TEST_RESULTS_PATH.read_text())

    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_xy(TEST_DATA_PATH)
    probability = model.predict_proba(X_test)[:, 1]
    calibration = calibration_table(y_test, probability, n_bins=4)
    test_calibration = [
        {"predicted": float(r.mean_predicted), "actual": float(r.actual_rate), "n": int(r.n)}
        for r in calibration.itertuples()
    ]

    X, y = load_train_xy()
    final = CalibratedClassifierCV(
        build_pipeline(make_lr(load_best_C()), scale_numeric=True), method="sigmoid", cv=5
    )
    oof = get_oof_probabilities(final, X, y)
    levels = pd.Series([risk_level(p, metadata["risk_bands"]) for p in oof])
    y_values = y.to_numpy()
    risk_bands = []
    for level in ["Low", "Medium", "High"]:
        mask = (levels == level).to_numpy()
        risk_bands.append(
            {
                "risk_level": level,
                "employees": int(mask.sum()),
                "actual_rate": float(y_values[mask].mean()) if mask.any() else 0.0,
            }
        )

    summary = pd.read_csv(REPORTS_DIR / "baseline_cv_summary.csv")
    pr = summary[summary["metric"] == "pr_auc"].set_index("model")
    roc = summary[summary["metric"] == "roc_auc"].set_index("model")
    cv_comparison = [
        {
            "model": label,
            "pr_auc": float(pr.loc[key, "mean"]),
            "pr_auc_std": float(pr.loc[key, "std"]),
            "roc_auc": float(roc.loc[key, "mean"]),
        }
        for key, label in MODEL_NAMES.items()
    ]

    card = {
        "model": {
            "name": metadata["model"],
            "C": metadata["C"],
            "cutoff": metadata["threshold"],
            "risk_bands": metadata["risk_bands"],
            "trained_rows": metadata["trained_rows"],
        },
        "test": test_results,
        "test_calibration": test_calibration,
        "cv_comparison": cv_comparison,
        "cv_floor_pr_auc": float(pr.loc["dummy_nobody_leaves", "mean"]),
        "risk_bands_oof": risk_bands,
        "excluded_columns": SENSITIVE_COLUMNS,
        "notes": NOTES,
    }
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
    print(f"Wrote {MODEL_CARD_PATH}")


if __name__ == "__main__":
    main()
