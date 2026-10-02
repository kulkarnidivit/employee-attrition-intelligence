import pandas as pd
import pytest

from src.data.validate_data import DataValidationError, raise_if_invalid, validate_raw_data
from tests.helpers import make_row


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame([make_row(1), make_row(2), make_row(3)])


def has_error(report, check: str) -> bool:
    return any(issue.check == check for issue in report.errors)


def test_valid_data_passes(valid_df):
    report = validate_raw_data(valid_df)
    assert report.is_valid
    raise_if_invalid(report)  # must not raise


def test_missing_column_is_error(valid_df):
    report = validate_raw_data(valid_df.drop(columns=["Age"]))
    assert has_error(report, "columns")


def test_invalid_target_value_is_error(valid_df):
    valid_df.loc[0, "Attrition"] = "Maybe"
    assert has_error(validate_raw_data(valid_df), "categories")


def test_out_of_range_age_is_error(valid_df):
    valid_df.loc[0, "Age"] = 5
    assert has_error(validate_raw_data(valid_df), "ranges")


def test_year_order_violation_is_error(valid_df):
    valid_df.loc[0, "YearsInCurrentRole"] = 9  # > YearsAtCompany (5)
    assert has_error(validate_raw_data(valid_df), "order_rules")


def test_duplicate_id_is_error(valid_df):
    valid_df.loc[1, "EmployeeNumber"] = 1
    assert has_error(validate_raw_data(valid_df), "id_unique")


def test_raise_if_invalid_raises(valid_df):
    valid_df.loc[0, "Attrition"] = "Maybe"
    with pytest.raises(DataValidationError):
        raise_if_invalid(validate_raw_data(valid_df))
