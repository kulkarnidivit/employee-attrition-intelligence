"""Model factories shared by the baseline, calibration, ablation and tuning runs."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


def make_logistic_regression(class_weight="balanced", C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(class_weight=class_weight, C=C, max_iter=2000, random_state=RANDOM_STATE)


def make_random_forest(class_weight="balanced") -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_xgboost(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
