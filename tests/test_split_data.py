import pandas as pd
import pytest

from src.data.schema import ID_COL
from src.data.split_data import save_splits, split_train_test

TARGET = "Attrition"


def make_df(n: int = 500, positive_rate: float = 0.16) -> pd.DataFrame:
    n_yes = int(n * positive_rate)
    return pd.DataFrame(
        {
            ID_COL: range(1, n + 1),
            TARGET: ["Yes"] * n_yes + ["No"] * (n - n_yes),
            "Age": [20 + i % 40 for i in range(n)],
        }
    )


def test_train_and_test_do_not_overlap():
    df = make_df()
    train, test = split_train_test(df, test_size=0.2, random_state=42)
    train_ids, test_ids = set(train[ID_COL]), set(test[ID_COL])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(df[ID_COL])  # every row ends up in exactly one split


def test_row_counts_are_correct():
    df = make_df(n=500)
    train, test = split_train_test(df, test_size=0.2, random_state=42)
    assert len(test) == 100
    assert len(train) == 400
    assert len(train) + len(test) == len(df)


def test_target_distribution_is_preserved():
    df = make_df(n=500, positive_rate=0.16)
    train, test = split_train_test(df, test_size=0.2, random_state=42)
    overall = (df[TARGET] == "Yes").mean()
    assert (train[TARGET] == "Yes").mean() == pytest.approx(overall, abs=0.01)
    assert (test[TARGET] == "Yes").mean() == pytest.approx(overall, abs=0.01)


def test_same_seed_gives_identical_split_different_seed_differs():
    df = make_df()
    train_a, test_a = split_train_test(df, random_state=42)
    train_b, test_b = split_train_test(df, random_state=42)
    _, test_c = split_train_test(df, random_state=7)
    pd.testing.assert_frame_equal(test_a, test_b)
    pd.testing.assert_frame_equal(train_a, train_b)
    assert set(test_a[ID_COL]) != set(test_c[ID_COL])


def test_missing_target_column_raises():
    with pytest.raises(ValueError):
        split_train_test(make_df().drop(columns=[TARGET]))


def test_save_splits_roundtrip(tmp_path):
    train, test = split_train_test(make_df())
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    save_splits(train, test, train_path, test_path)
    assert len(pd.read_csv(train_path)) == len(train)
    assert set(pd.read_csv(test_path)[ID_COL]) == set(test[ID_COL])
