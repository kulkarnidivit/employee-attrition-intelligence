import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.data.schema import SENSITIVE_COLUMNS
from src.features.engineering import ENGINEERED_FEATURES
from src.features.preprocessing import build_pipeline, get_model_columns
from src.models.data import split_xy
from tests.factories import make_dataset


def test_default_columns_exclude_id_sensitive_and_target():
    cols = get_model_columns()
    used = cols["categorical"] + cols["numeric"]
    assert "EmployeeNumber" not in used
    assert "Attrition" not in used
    assert not set(SENSITIVE_COLUMNS) & set(used)
    assert set(ENGINEERED_FEATURES) <= set(used)
    assert len(used) == len(set(used))


def test_include_sensitive_switch_adds_them_back():
    used = sum(get_model_columns(include_sensitive=True).values(), [])
    assert set(SENSITIVE_COLUMNS) <= set(used)


def test_invalid_feature_set_raises():
    with pytest.raises(ValueError):
        get_model_columns(feature_set="everything")


def test_transformed_output_is_a_clean_numeric_dataframe():
    X, y = split_xy(make_dataset())
    out = build_pipeline(DummyClassifier())[:-1].fit_transform(X, y)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(X)
    assert not out.isna().any().any()
    assert out.dtypes.map(pd.api.types.is_numeric_dtype).all()
    assert "OverTime_Yes" in out.columns
    for banned in ["EmployeeNumber", "Age", "Gender_Male", "MaritalStatus_Single"]:
        assert banned not in out.columns


def test_unseen_category_is_ignored_not_crash():
    X, y = split_xy(make_dataset())
    prep = build_pipeline(DummyClassifier())[:-1].fit(X, y)
    unseen = X.iloc[[0]].copy()
    unseen["JobRole"] = "Manager"  # not in the training data
    out = prep.transform(unseen)
    role_cols = [c for c in out.columns if c.startswith("JobRole_")]
    assert role_cols
    assert (out[role_cols].iloc[0] == 0).all()


def test_scaling_switch():
    X, y = split_xy(make_dataset())
    scaled = build_pipeline(DummyClassifier(), scale_numeric=True)[:-1].fit_transform(X, y)
    unscaled = build_pipeline(DummyClassifier(), scale_numeric=False)[:-1].fit_transform(X, y)
    assert scaled["MonthlyIncome"].mean() == pytest.approx(0, abs=1e-9)
    assert scaled["MonthlyIncome"].std(ddof=0) == pytest.approx(1)
    assert unscaled["MonthlyIncome"].tolist() == X["MonthlyIncome"].tolist()


def test_feature_set_switch_controls_engineered_columns():
    X, y = split_xy(make_dataset())
    raw = build_pipeline(DummyClassifier(), feature_set="raw")[:-1].fit_transform(X, y)
    eng = build_pipeline(DummyClassifier(), feature_set="engineered")[:-1].fit_transform(X, y)
    assert not set(ENGINEERED_FEATURES) & set(raw.columns)
    assert set(ENGINEERED_FEATURES) <= set(eng.columns)


def test_scaler_and_medians_are_fit_on_training_rows_only():
    X, y = split_xy(make_dataset())
    X_train, y_train = X.iloc[:80], y.iloc[:80]
    pipe = build_pipeline(DummyClassifier(), "engineered", scale_numeric=True).fit(X_train, y_train)

    scaler = pipe.named_steps["preprocess"].named_transformers_["numeric"]
    i = list(scaler.feature_names_in_).index("MonthlyIncome")
    assert scaler.mean_[i] == pytest.approx(X_train["MonthlyIncome"].mean())
    assert scaler.mean_[i] != pytest.approx(X["MonthlyIncome"].mean())  # not the full-data mean

    engineer = pipe.named_steps["features"]
    assert engineer.global_median_income_ == pytest.approx(X_train["MonthlyIncome"].median())


def test_pipeline_does_not_modify_its_input():
    X, y = split_xy(make_dataset())
    before = X.copy()
    build_pipeline(DummyClassifier(), scale_numeric=True).fit(X, y)
    pd.testing.assert_frame_equal(X, before)
