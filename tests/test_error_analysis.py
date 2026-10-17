import numpy as np
import pandas as pd
import pytest

from src.models.error_analysis import flag_top_share, group_report


def test_flag_top_share_flags_the_right_number():
    proba = np.arange(100) / 100
    flagged = flag_top_share(proba, 0.2)
    assert flagged.sum() == 20
    assert flagged[-20:].all()  # the highest scores are the ones flagged


def test_group_report_counts_and_rates():
    df = pd.DataFrame(
        {
            "g": ["A", "A", "A", "A", "B", "B"],
            "attrition": [1, 1, 0, 0, 0, 0],
            "flagged": [True, False, True, False, False, False],
        }
    )
    report = group_report(df, "g").set_index("g")
    assert report.loc["A", "n"] == 4
    assert report.loc["A", "leaver_rate"] == pytest.approx(0.5)
    assert report.loc["A", "flagged_rate"] == pytest.approx(0.5)
    assert report.loc["A", "recall"] == pytest.approx(0.5)
    assert report.loc["A", "precision"] == pytest.approx(0.5)
    assert np.isnan(report.loc["B", "recall"])  # no leavers in group B
    assert np.isnan(report.loc["B", "precision"])  # nobody flagged in group B
