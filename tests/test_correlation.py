import pandas as pd
import pytest

from src.analysis.correlation import target_correlations, top_correlated_pairs


def make_df() -> pd.DataFrame:
    x = list(range(100))
    return pd.DataFrame(
        {
            "a": x,
            "b": [2 * v for v in x],  # perfectly rank-correlated with a
            "noise": [0, 1] * 50,  # alternating, essentially uncorrelated with a
            "Attrition": ["No"] * 80 + ["Yes"] * 20,
        }
    )


def test_top_pairs_finds_only_the_strong_pair():
    pairs = top_correlated_pairs(make_df(), ["a", "b", "noise"], threshold=0.6)
    assert len(pairs) == 1
    assert {pairs.loc[0, "feature_a"], pairs.loc[0, "feature_b"]} == {"a", "b"}
    assert pairs.loc[0, "spearman"] == pytest.approx(1.0)


def test_top_pairs_returns_empty_table_when_nothing_passes():
    pairs = top_correlated_pairs(make_df(), ["a", "noise"], threshold=0.6)
    assert pairs.empty
    assert list(pairs.columns) == ["feature_a", "feature_b", "spearman"]


def test_target_correlations_ranks_the_related_feature_first():
    corr = target_correlations(make_df(), ["a", "noise"])
    assert corr.index[0] == "a"
    assert corr["a"] > 0.5
    assert abs(corr["noise"]) < 0.1
