import pandas as pd
import pytest

from src.analysis.attrition_rates import attrition_rate_table


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dept": ["A"] * 100 + ["B"] * 50,
            "Attrition": ["Yes"] * 20 + ["No"] * 80 + ["Yes"] * 5 + ["No"] * 45,
        }
    )


def test_counts_and_rates_per_group():
    table = attrition_rate_table(make_df(), "Dept").set_index("group")
    assert table.loc["A", "n"] == 100
    assert table.loc["A", "leavers"] == 20
    assert table.loc["A", "attrition_rate"] == pytest.approx(0.20)
    assert table.loc["B", "n"] == 50
    assert table.loc["B", "leavers"] == 5
    assert table.loc["B", "attrition_rate"] == pytest.approx(0.10)


def test_interval_brackets_rate_and_stays_in_bounds():
    table = attrition_rate_table(make_df(), "Dept")
    assert (table["ci_low"] <= table["attrition_rate"]).all()
    assert (table["attrition_rate"] <= table["ci_high"]).all()
    assert (table["ci_low"] >= 0).all()
    assert (table["ci_high"] <= 1).all()


def test_numeric_column_with_many_values_is_binned():
    df = pd.DataFrame(
        {"Score": list(range(100)), "Attrition": ["Yes", "No"] * 50}
    )
    table = attrition_rate_table(df, "Score", n_bins=5)
    assert len(table) == 5
    assert table["n"].sum() == len(df)


def test_discrete_numeric_column_uses_its_values_as_groups():
    df = pd.DataFrame({"Level": [1, 2, 3] * 10, "Attrition": ["Yes", "No", "No"] * 10})
    table = attrition_rate_table(df, "Level")
    assert table["group"].tolist() == ["1", "2", "3"]


def test_group_with_no_leavers_has_zero_lower_bound():
    df = pd.DataFrame({"G": ["C"] * 30, "Attrition": ["No"] * 30})
    table = attrition_rate_table(df, "G")
    assert table.loc[0, "attrition_rate"] == 0
    assert table.loc[0, "ci_low"] == pytest.approx(0, abs=1e-12)
    assert table.loc[0, "ci_high"] > 0
