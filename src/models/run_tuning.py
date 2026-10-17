"""Tune Logistic Regression (C) and XGBoost with the project's cross-validation (training set only)."""

import json
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

from src.config import BEST_PARAMS_PATH, RANDOM_STATE, REPORTS_DIR
from src.features.preprocessing import build_pipeline
from src.models.cross_validation import make_cv
from src.models.data import load_train_xy


def _plain(params: dict) -> dict:
    return {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    X, y = load_train_xy()
    share = y.mean()
    cv = make_cv()

    lr_pipe = build_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        scale_numeric=True,
    )
    lr_search = GridSearchCV(
        lr_pipe, {"model__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
        scoring="average_precision", cv=cv,
    ).fit(X, y)

    print("\n=== LOGISTIC REGRESSION: PR-AUC by C ===")
    table = pd.DataFrame(lr_search.cv_results_)[["param_model__C", "mean_test_score", "std_test_score"]]
    print(table.round(3).to_string(index=False))
    print(f"Best C: {lr_search.best_params_['model__C']}  (PR-AUC {lr_search.best_score_:.3f})")

    xgb_pipe = build_pipeline(
        XGBClassifier(
            eval_metric="logloss", scale_pos_weight=(1 - share) / share, n_jobs=-1, random_state=RANDOM_STATE
        )
    )
    xgb_search = RandomizedSearchCV(
        xgb_pipe,
        {
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__n_estimators": [100, 200, 300, 500],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_lambda": [1, 5, 10],
        },
        n_iter=20, scoring="average_precision", cv=cv, random_state=RANDOM_STATE,
    ).fit(X, y)

    print("\n=== XGBOOST (random search, 20 tries) ===")
    print(f"Best PR-AUC {xgb_search.best_score_:.3f} with {_plain(xgb_search.best_params_)}")
    print("\nNote: the best score of many tries is slightly optimistic. Untuned baselines were "
          "Logistic Regression 0.626 and XGBoost 0.573.")

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(
        json.dumps(
            {
                "logistic_regression": {**_plain(lr_search.best_params_), "pr_auc_cv": float(lr_search.best_score_)},
                "xgboost": {**_plain(xgb_search.best_params_), "pr_auc_cv": float(xgb_search.best_score_)},
            },
            indent=2,
        )
    )
    logging.info("Saved %s", BEST_PARAMS_PATH)


if __name__ == "__main__":
    main()
