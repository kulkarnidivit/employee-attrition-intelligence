"""Ablation on the training set (cross-validation only): do engineered features and sensitive columns help?"""

import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, REPORTS_DIR
from src.features.preprocessing import build_pipeline
from src.models.cross_validation import run_cross_validation
from src.models.data import load_train_xy


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    X, y = load_train_xy()
    share = y.mean()
    makers = {
        "logistic_regression": lambda: LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(1 - share) / share, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }

    rows = []
    for model_name, make in makers.items():
        for feature_set in ("raw", "engineered"):
            for sensitive in (False, True):
                logging.info("%s | features=%s | sensitive columns=%s", model_name, feature_set, sensitive)
                pipe = build_pipeline(
                    make(), feature_set, sensitive, scale_numeric=(model_name == "logistic_regression")
                )
                scores = run_cross_validation(pipe, X, y)
                rows.append(
                    {
                        "model": model_name,
                        "features": feature_set,
                        "sensitive_cols": sensitive,
                        "pr_auc": scores["pr_auc"].mean(),
                        "pr_auc_std": scores["pr_auc"].std(),
                        "roc_auc": scores["roc_auc"].mean(),
                        "f1": scores["f1"].mean(),
                    }
                )

    result = pd.DataFrame(rows)
    print("\n=== ABLATION (training CV, 15 folds) ===")
    print(result.round(3).to_string(index=False))

    print()
    for model_name in makers:
        part = result[result["model"] == model_name].set_index(["features", "sensitive_cols"])["pr_auc"]
        gain = part[("engineered", False)] - part[("raw", False)]
        sens = part[("engineered", True)] - part[("engineered", False)]
        print(f"{model_name}: engineered vs raw {gain:+.3f} PR-AUC | adding sensitive columns {sens:+.3f} PR-AUC")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "ablation.csv", index=False)


if __name__ == "__main__":
    main()
