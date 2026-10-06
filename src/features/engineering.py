"""Feature engineering as a scikit-learn transformer.

Stateless features use only the employee's own row. The one stateful feature,
pay_vs_level_median, learns the median income per job level in fit(), so it can be
fit on training data only (and re-fit inside each cross-validation fold).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.config import (
    BUSINESS_TRAVEL_LEVELS,
    FIRST_YEAR_MAX_YEARS,
    JUNIOR_JOB_LEVEL,
    MANAGER_CHANGE_MAX_YEARS,
)

REQUIRED_COLUMNS = [
    "YearsAtCompany", "TotalWorkingYears", "YearsInCurrentRole", "YearsWithCurrManager",
    "YearsSinceLastPromotion", "NumCompaniesWorked", "OverTime", "BusinessTravel",
    "JobLevel", "MonthlyIncome",
]

RATIO_FEATURES = [
    "tenure_ratio", "role_stability", "manager_stability",
    "promotion_delay_ratio", "job_hopping", "pay_vs_level_median",
]
FLAG_FEATURES = ["is_first_year", "recent_manager_change", "junior_overtime"]
ENGINEERED_FEATURES = RATIO_FEATURES[:5] + ["work_stress_score"] + FLAG_FEATURES + ["pay_vs_level_median"]


def _check_required_columns(X: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _overtime_flag(series: pd.Series) -> pd.Series:
    valid = series.isin(["Yes", "No"])
    if not valid.all():
        bad = sorted(series[~valid].astype(str).unique())
        raise ValueError(f"Unexpected OverTime values: {bad}")
    return (series == "Yes").astype(int)


def _travel_level(series: pd.Series) -> pd.Series:
    level = series.map(BUSINESS_TRAVEL_LEVELS)
    if level.isna().any():
        bad = sorted(series[level.isna()].astype(str).unique())
        raise ValueError(f"Unexpected BusinessTravel values: {bad}")
    return level.astype(int)


class AttritionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered features to the cleaned HR data (the target must not be included)."""

    def __init__(
        self,
        first_year_max: int = FIRST_YEAR_MAX_YEARS,
        manager_change_max: int = MANAGER_CHANGE_MAX_YEARS,
        junior_level: int = JUNIOR_JOB_LEVEL,
    ):
        self.first_year_max = first_year_max
        self.manager_change_max = manager_change_max
        self.junior_level = junior_level

    def fit(self, X: pd.DataFrame, y=None):
        """Learn median income per job level (use training data only)."""
        _check_required_columns(X)
        self.level_median_income_ = X.groupby("JobLevel")["MonthlyIncome"].median()
        self.global_median_income_ = float(X["MonthlyIncome"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of X with the engineered columns added."""
        check_is_fitted(self, ["level_median_income_", "global_median_income_"])
        _check_required_columns(X)

        overtime = _overtime_flag(X["OverTime"])
        travel_level = _travel_level(X["BusinessTravel"])
        years = X["YearsAtCompany"]

        out = X.copy()
        out["tenure_ratio"] = years / (X["TotalWorkingYears"] + 1)
        out["role_stability"] = X["YearsInCurrentRole"] / (years + 1)
        out["manager_stability"] = X["YearsWithCurrManager"] / (years + 1)
        out["promotion_delay_ratio"] = X["YearsSinceLastPromotion"] / (years + 1)
        out["job_hopping"] = X["NumCompaniesWorked"] / (X["TotalWorkingYears"] + 1)
        out["work_stress_score"] = overtime + travel_level
        out["is_first_year"] = (years <= self.first_year_max).astype(int)
        out["recent_manager_change"] = (
            (years > self.first_year_max) & (X["YearsWithCurrManager"] <= self.manager_change_max)
        ).astype(int)
        out["junior_overtime"] = ((X["JobLevel"] == self.junior_level) & (overtime == 1)).astype(int)

        level_median = X["JobLevel"].map(self.level_median_income_).fillna(self.global_median_income_)
        out["pay_vs_level_median"] = X["MonthlyIncome"] / level_median
        return out


def find_feature_problems(features: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems found in the engineered columns."""
    problems = []
    for col in ENGINEERED_FEATURES:
        if col not in features.columns:
            problems.append(f"{col}: column is missing")
            continue
        values = features[col]
        if values.isna().any():
            problems.append(f"{col}: contains NaN")
        if np.isinf(values.to_numpy(dtype=float)).any():
            problems.append(f"{col}: contains infinite values")
        if col in RATIO_FEATURES and (values < 0).any():
            problems.append(f"{col}: contains negative values")
        if col in FLAG_FEATURES and not values.isin([0, 1]).all():
            problems.append(f"{col}: should contain only 0 or 1")
        if col == "work_stress_score" and not values.between(0, 3).all():
            problems.append(f"{col}: should be between 0 and 3")
    return problems
