import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.features.engineering import (
    ENGINEERED_FEATURES,
    AttritionFeatureEngineer,
    find_feature_problems,
)
from tests.helpers import make_row


def row(**overrides) -> dict:
    r = make_row(1)
    r.update(overrides)
    return r


def test_adds_engineered_features_without_changing_inputs():
    df = pd.DataFrame([row(EmployeeNumber=1), row(EmployeeNumber=2, JobLevel=3, MonthlyIncome=9000)])
    before = df.copy()
    out = AttritionFeatureEngineer().fit_transform(df)
    assert set(ENGINEERED_FEATURES) <= set(out.columns)
    pd.testing.assert_frame_equal(out[df.columns], before)  # original columns untouched
    pd.testing.assert_frame_equal(df, before)  # input not mutated


def test_ratio_features_match_hand_calculation():
    # make_row: YearsAtCompany=5, TotalWorkingYears=10, InCurrentRole=3, WithCurrManager=2,
    # SinceLastPromotion=1, NumCompaniesWorked=2, OverTime=No, Travel_Rarely, MonthlyIncome=5000
    out = AttritionFeatureEngineer().fit_transform(pd.DataFrame([row()]))
    r = out.iloc[0]
    assert r["tenure_ratio"] == pytest.approx(5 / 11)
    assert r["role_stability"] == pytest.approx(3 / 6)
    assert r["manager_stability"] == pytest.approx(2 / 6)
    assert r["promotion_delay_ratio"] == pytest.approx(1 / 6)
    assert r["job_hopping"] == pytest.approx(2 / 11)
    assert r["work_stress_score"] == 1
    assert r["pay_vs_level_median"] == pytest.approx(1.0)


def test_flag_features():
    df = pd.DataFrame(
        [
            row(YearsAtCompany=1, TotalWorkingYears=3, YearsInCurrentRole=1, YearsWithCurrManager=1,
                YearsSinceLastPromotion=0),  # first year
            row(YearsAtCompany=6, TotalWorkingYears=8, YearsInCurrentRole=4, YearsWithCurrManager=1),  # manager change
            row(JobLevel=1, OverTime="Yes"),  # junior with overtime
        ]
    )
    out = AttritionFeatureEngineer().fit_transform(df)
    assert out["is_first_year"].tolist() == [1, 0, 0]
    assert out["recent_manager_change"].tolist() == [0, 1, 0]
    assert out["junior_overtime"].tolist() == [0, 0, 1]


def test_work_stress_score_range():
    df = pd.DataFrame(
        [
            row(OverTime="No", BusinessTravel="Non-Travel"),
            row(OverTime="Yes", BusinessTravel="Travel_Frequently"),
        ]
    )
    out = AttritionFeatureEngineer().fit_transform(df)
    assert out["work_stress_score"].tolist() == [0, 3]


def test_pay_vs_level_median_uses_training_medians_only():
    train = pd.DataFrame(
        [row(JobLevel=2, MonthlyIncome=4000), row(JobLevel=2, MonthlyIncome=6000),
         row(JobLevel=3, MonthlyIncome=8000), row(JobLevel=3, MonthlyIncome=12000)]
    )
    fe = AttritionFeatureEngineer().fit(train)
    test = pd.DataFrame([row(JobLevel=2, MonthlyIncome=7500), row(JobLevel=5, MonthlyIncome=14000)])
    out = fe.transform(test)
    # level 2 median (train) = 5000; unseen level 5 falls back to the global train median = 7000
    assert out["pay_vs_level_median"].tolist() == pytest.approx([1.5, 2.0])
    assert fe.level_median_income_.to_dict() == {2: 5000.0, 3: 10000.0}  # unchanged by transform


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        AttritionFeatureEngineer().transform(pd.DataFrame([row()]))


def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        AttritionFeatureEngineer().fit(pd.DataFrame([row()]).drop(columns=["OverTime"]))


def test_unknown_business_travel_raises():
    fe = AttritionFeatureEngineer().fit(pd.DataFrame([row()]))
    with pytest.raises(ValueError):
        fe.transform(pd.DataFrame([row(BusinessTravel="Sometimes")]))


def test_transformer_can_be_cloned():
    assert clone(AttritionFeatureEngineer(first_year_max=2)).first_year_max == 2


def test_find_feature_problems_clean_data():
    out = AttritionFeatureEngineer().fit_transform(pd.DataFrame([row(), row(EmployeeNumber=2)]))
    assert find_feature_problems(out) == []


def test_find_feature_problems_detects_issues():
    out = AttritionFeatureEngineer().fit_transform(pd.DataFrame([row(), row(EmployeeNumber=2)]))
    out.loc[0, "tenure_ratio"] = float("nan")
    out.loc[1, "is_first_year"] = 2
    problems = find_feature_problems(out)
    assert any("tenure_ratio" in p for p in problems)
    assert any("is_first_year" in p for p in problems)
