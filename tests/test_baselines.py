import pytest
from sklearn.model_selection import RepeatedStratifiedKFold

from src.models.cross_validation import run_cross_validation
from src.models.data import split_xy
from src.models.run_baselines import get_baseline_models
from tests.factories import make_dataset


def test_dummy_baselines_have_the_expected_floor_scores():
    X, y = split_xy(make_dataset())  # 25% leavers
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=0)
    models = get_baseline_models()

    nobody = run_cross_validation(models["dummy_nobody_leaves"], X, y, cv=cv).mean()
    assert nobody["recall"] == 0
    assert nobody["f1"] == 0
    assert nobody["pr_auc"] == pytest.approx(0.25, abs=0.02)

    everybody = run_cross_validation(models["dummy_everybody_leaves"], X, y, cv=cv).mean()
    assert everybody["recall"] == pytest.approx(1.0)
    assert everybody["precision"] == pytest.approx(0.25)
    assert everybody["f1"] == pytest.approx(0.4)  # 2 * 0.25 * 1 / (0.25 + 1)
