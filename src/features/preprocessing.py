"""Preprocessing pipeline: feature engineering + encoding + optional scaling.

Everything that learns from data (engineered medians, scaler means, one-hot categories)
lives inside one scikit-learn Pipeline, so it is re-fit on the training rows of each
cross-validation fold and never sees validation or test rows.
"""

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    ORDINAL_COLUMNS,
    SENSITIVE_COLUMNS,
)
from src.features.engineering import ENGINEERED_FEATURES, AttritionFeatureEngineer

FEATURE_SETS = ("raw", "engineered")


def get_model_columns(
    feature_set: str = "engineered", include_sensitive: bool = False
) -> dict[str, list[str]]:
    """Return the columns the model will use, grouped as 'categorical' and 'numeric'.

    EmployeeNumber (ID) and the target are never included. Sensitive columns are left
    out unless include_sensitive is True.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got '{feature_set}'")

    categorical = list(CATEGORICAL_COLUMNS)
    numeric = list(NUMERIC_COLUMNS) + list(ORDINAL_COLUMNS)  # rating scales stay numeric
    if feature_set == "engineered":
        numeric += ENGINEERED_FEATURES
    if not include_sensitive:
        categorical = [c for c in categorical if c not in SENSITIVE_COLUMNS]
        numeric = [c for c in numeric if c not in SENSITIVE_COLUMNS]
    return {"categorical": categorical, "numeric": numeric}


def build_preprocessor(
    feature_set: str = "engineered",
    include_sensitive: bool = False,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """One-hot encode categories; optionally scale numbers (needed for Logistic Regression)."""
    cols = get_model_columns(feature_set, include_sensitive)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cols["categorical"],
            ),
            ("numeric", StandardScaler() if scale_numeric else "passthrough", cols["numeric"]),
        ],
        remainder="drop",  # anything not listed (ID, sensitive columns) is dropped
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")  # readable column names (needed for SHAP later)
    return preprocessor


def build_pipeline(
    estimator: BaseEstimator,
    feature_set: str = "engineered",
    include_sensitive: bool = False,
    scale_numeric: bool = False,
) -> Pipeline:
    """Full pipeline: [feature engineering] -> preprocessing -> model.

    feature_set='raw' skips feature engineering and uses the original columns only.
    """
    steps = []
    if feature_set == "engineered":
        steps.append(("features", AttritionFeatureEngineer()))
    steps.append(("preprocess", build_preprocessor(feature_set, include_sensitive, scale_numeric)))
    steps.append(("model", estimator))
    return Pipeline(steps)
