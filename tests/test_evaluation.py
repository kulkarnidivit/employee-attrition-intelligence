import numpy as np
import pytest
from sklearn.dummy import DummyClassifier  # noqa: F401  (kept for parity with other tests)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.features.preprocessing import build_pipeline
from src.models.data import split_xy
from src.models.evaluation import calibration_table, pooled_scores, threshold_table
from src.models.out_of_fold import get_oof_probabilities
from tests.factories import make_dataset


def test_threshold_table_counts_are_correct():
    y = [1, 1, 0, 0]
    proba = [0.9, 0.4, 0.6, 0.1]
    table = threshold_table(y, proba, [0.0, 0.5, 1.0]).set_index("threshold")

    half = table.loc[0.5]  # flags employees 0 and 2
    assert (half["flagged"], half["caught"], half["false_alarms"], half["missed"]) == (2, 1, 1, 1)
    assert half["precision"] == pytest.approx(0.5)
    assert half["recall"] == pytest.approx(0.5)
    assert half["f1"] == pytest.approx(0.5)

    everyone = table.loc[0.0]
    assert everyone["flagged"] == 4
    assert everyone["recall"] == pytest.approx(1.0)
    assert everyone["precision"] == pytest.approx(0.5)

    nobody = table.loc[1.0]
    assert nobody["flagged"] == 0
    assert nobody["precision"] == 0
    assert nobody["recall"] == 0


def test_threshold_table_flags_fewer_people_as_cutoff_rises():
    rng = np.random.default_rng(0)
    proba = rng.random(200)
    y = (rng.random(200) < 0.3).astype(int)
    table = threshold_table(y, proba, [0.2, 0.4, 0.6, 0.8])
    assert table["flagged"].is_monotonic_decreasing


def test_pooled_scores_for_perfect_separation():
    scores = pooled_scores([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert scores["pr_auc"] == pytest.approx(1.0)
    assert scores["roc_auc"] == pytest.approx(1.0)


def test_calibration_table_groups_cover_all_rows():
    proba = np.linspace(0.01, 0.99, 100)
    y = [0, 1] * 50
    table = calibration_table(y, proba, n_bins=5)
    assert len(table) == 5
    assert table["n"].sum() == 100
    assert table["mean_predicted"].is_monotonic_increasing


def test_oof_probabilities_have_right_shape_range_and_are_reproducible():
    X, y = split_xy(make_dataset())
    pipe = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0))
    first = get_oof_probabilities(pipe, X, y, n_repeats=1)
    second = get_oof_probabilities(pipe, X, y, n_repeats=1)
    assert first.shape == (len(y),)
    assert ((first >= 0) & (first <= 1)).all()
    np.testing.assert_allclose(first, second)


def test_oof_predictions_are_honest_unlike_in_sample_predictions():
    # random labels: nothing real to learn, so honest scores must be near chance
    X, _ = split_xy(make_dataset(200))
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.3).astype(int)

    pipe = build_pipeline(RandomForestClassifier(n_estimators=100, random_state=0))
    in_sample_auc = roc_auc_score(y, pipe.fit(X, y).predict_proba(X)[:, 1])
    oof_auc = roc_auc_score(y, get_oof_probabilities(pipe, X, y, n_repeats=1))
    assert in_sample_auc > 0.9  # memorized
    assert oof_auc < 0.7  # honest: close to chance
