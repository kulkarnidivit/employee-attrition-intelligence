import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from src.features.preprocessing import build_pipeline
from src.models.cross_validation import make_cv, run_cross_validation, summarize
from src.models.data import split_xy
from tests.factories import make_dataset


def test_make_cv_is_five_folds_repeated_three_times():
    assert make_cv().get_n_splits() == 15


def test_pipeline_runs_inside_cross_validation_with_dummy_model():
    X, y = split_xy(make_dataset())
    pipe = build_pipeline(DummyClassifier(), "engineered", scale_numeric=True)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=0)
    scores = run_cross_validation(pipe, X, y, cv=cv)
    assert scores.shape == (3, 5)
    assert not scores.isna().any().any()
    # a model that knows nothing scores PR-AUC = share of leavers (25%) and ROC-AUC = 0.5
    assert scores["pr_auc"].tolist() == pytest.approx([0.25] * 3, abs=0.02)
    assert scores["roc_auc"].tolist() == pytest.approx([0.5] * 3)


def test_summarize_gives_mean_and_std():
    summary = summarize(pd.DataFrame({"pr_auc": [0.2, 0.4]}))
    assert summary.loc["pr_auc", "mean"] == pytest.approx(0.3)
    assert summary.loc["pr_auc", "std"] == pytest.approx(0.1414, abs=1e-3)
