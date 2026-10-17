"""Train the final model on the whole training set, save it, and evaluate on the locked test set ONCE.

Final model: Logistic Regression (balanced class weights) + sigmoid probability calibration.
The cutoff is chosen on out-of-fold TRAINING predictions only.
"""

import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
)

from src.config import (
    BEST_PARAMS_PATH, EXPLAINER_PATH, METADATA_PATH, MODEL_PATH, MODELS_DIR,
    RANDOM_STATE, RISK_BANDS, TEST_DATA_PATH, TEST_RESULTS_PATH,
)
from src.features.preprocessing import build_pipeline
from src.models.data import load_train_xy, load_xy
from src.models.evaluation import calibration_table
from src.models.out_of_fold import get_oof_probabilities
from src.serving.service import risk_level

logger = logging.getLogger(__name__)


def make_lr(C: float) -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", C=C, max_iter=2000, random_state=RANDOM_STATE)


def load_best_C() -> float:
    if BEST_PARAMS_PATH.exists():
        return float(json.loads(BEST_PARAMS_PATH.read_text())["logistic_regression"]["C"])
    return 1.0


def pick_threshold(y, proba) -> float:
    """The cutoff with the best F1 on the given (out-of-fold, training) predictions."""
    grid = np.round(np.arange(0.05, 0.80, 0.01), 2)
    scores = [f1_score(y, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if TEST_RESULTS_PATH.exists() and "--force" not in sys.argv:
        raise SystemExit(
            f"{TEST_RESULTS_PATH} already exists. The test set is meant to be used ONCE. "
            "Re-run with --force only if you accept that."
        )

    X, y = load_train_xy()
    C = load_best_C()
    logger.info("Final model: Logistic Regression, balanced weights, C=%s, sigmoid calibration", C)

    def new_model():
        return CalibratedClassifierCV(build_pipeline(make_lr(C), scale_numeric=True), method="sigmoid", cv=5)

    oof = get_oof_probabilities(new_model(), X, y)
    threshold = pick_threshold(y, oof)
    flagged_oof = oof >= threshold
    print(f"\nChosen cutoff (best F1 on out-of-fold training predictions): {threshold:.2f}")
    print(
        f"Training out-of-fold: PR-AUC {average_precision_score(y, oof):.3f}, "
        f"ROC-AUC {roc_auc_score(y, oof):.3f}, Brier {brier_score_loss(y, oof):.3f}, "
        f"precision {precision_score(y, flagged_oof, zero_division=0):.3f}, "
        f"recall {recall_score(y, flagged_oof):.3f}, F1 {f1_score(y, flagged_oof):.3f}"
    )
    bands = pd.Series([risk_level(p) for p in oof])
    band_table = pd.DataFrame({"n": bands.value_counts(), "actual_rate": y.groupby(bands.values).mean()})
    print("\nRisk bands on out-of-fold training predictions (bands in src/config.py):")
    print(band_table.reindex(["Low", "Medium", "High"]).round(3).to_string())

    final_model = new_model().fit(X, y)
    explainer_pipe = build_pipeline(make_lr(C), scale_numeric=True).fit(X, y)
    background_mean = explainer_pipe[:-1].transform(X).mean()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump({"pipeline": explainer_pipe, "background_mean": background_mean}, EXPLAINER_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "model": "logistic_regression + sigmoid calibration",
                "C": C,
                "threshold": threshold,
                "risk_bands": RISK_BANDS,
                "trained_rows": int(len(y)),
                "sklearn_version": sklearn.__version__,
            },
            indent=2,
        )
    )

    # ---- the one-time test evaluation ----
    X_test, y_test = load_xy(TEST_DATA_PATH)
    proba = final_model.predict_proba(X_test)[:, 1]
    flagged = proba >= threshold
    results = {
        "n_test": int(len(y_test)),
        "leavers": int(y_test.sum()),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "threshold": threshold,
        "flagged": int(flagged.sum()),
        "caught": int((flagged & (y_test == 1)).sum()),
        "precision": float(precision_score(y_test, flagged, zero_division=0)),
        "recall": float(recall_score(y_test, flagged)),
        "f1": float(f1_score(y_test, flagged)),
        "floor_pr_auc": float(y_test.mean()),
    }
    TEST_RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== TEST SET RESULTS (used once) ===")
    for key, value in results.items():
        print(f"  {key:12s} {value:.3f}" if isinstance(value, float) else f"  {key:12s} {value}")
    table = calibration_table(y_test, proba, n_bins=4)
    print("\nTest calibration (predicted -> actual):")
    print("  " + " | ".join(f"{r.mean_predicted:.2f}->{r.actual_rate:.2f}" for r in table.itertuples()))
    print(f"\nSaved model files to {MODELS_DIR}")


if __name__ == "__main__":
    main()
