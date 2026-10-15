import pandas as pd
import pytest

from src.models.data import load_xy, split_xy


def test_split_xy_encodes_target_and_removes_it_from_x():
    df = pd.DataFrame({"a": [1, 2, 3], "Attrition": ["Yes", "No", "No"]})
    X, y = split_xy(df)
    assert y.tolist() == [1, 0, 0]
    assert list(X.columns) == ["a"]


def test_split_xy_rejects_unexpected_target_values():
    df = pd.DataFrame({"a": [1, 2], "Attrition": ["Yes", "Maybe"]})
    with pytest.raises(ValueError):
        split_xy(df)


def test_load_xy_reads_a_csv(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "Attrition": ["No", "Yes"]}).to_csv(path, index=False)
    X, y = load_xy(path)
    assert X.shape == (2, 1)
    assert y.tolist() == [0, 1]
