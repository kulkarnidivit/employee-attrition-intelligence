import pandas as pd
import pytest

from src.data.clean_data import clean_data, drop_constant_columns
from src.data.validate_data import DataValidationError
from tests.helpers import make_row


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame([make_row(1), make_row(2), make_row(3)])


def test_constant_columns_dropped_and_rest_kept(raw_df):
    cleaned = clean_data(raw_df)
    assert not {"EmployeeCount", "Over18", "StandardHours"} & set(cleaned.columns)
    assert cleaned.shape == (3, raw_df.shape[1] - 3)
    assert {"EmployeeNumber", "Attrition"} <= set(cleaned.columns)


def test_text_whitespace_is_stripped(raw_df):
    raw_df.loc[0, "Department"] = "  Sales "
    assert clean_data(raw_df).loc[0, "Department"] == "Sales"


def test_invalid_data_raises(raw_df):
    raw_df.loc[0, "Attrition"] = "Maybe"
    with pytest.raises(DataValidationError):
        clean_data(raw_df)


def test_input_is_not_mutated(raw_df):
    before = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_non_constant_column_listed_as_constant_raises(raw_df):
    raw_df.loc[0, "EmployeeCount"] = 2
    with pytest.raises(ValueError):
        drop_constant_columns(raw_df)
