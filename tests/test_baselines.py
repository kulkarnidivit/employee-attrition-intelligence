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


def test_registry_contains_dummies_and_three_real_models():
    assert set(get_baseline_models()) == {
        "dummy_nobody_leaves",
        "dummy_everybody_leaves",
        "logistic_regression",
        "random_forest",
        "xgboost",
    }


def test_real_baselines_use_class_weights():
    models = get_baseline_models(scale_pos_weight=5.2)
    assert models["logistic_regression"].named_steps["model"].class_weight == "balanced"
    assert models["random_forest"].named_steps["model"].class_weight == "balanced"
    assert models["xgboost"].named_steps["model"].scale_pos_weight == 5.2


def test_real_baselines_fit_and_give_probabilities():
    X, y = split_xy(make_dataset())
    for name in ("logistic_regression", "random_forest", "xgboost"):
        pipe = get_baseline_models()[name]
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert ((proba >= 0) & (proba <= 1)).all()
