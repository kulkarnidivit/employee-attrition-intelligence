import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from src.features.preprocessing import build_pipeline
from src.models.data import split_xy
from src.serving.service import AttritionService, linear_contributions, risk_level
from tests.factories import make_dataset

BANDS = {"medium": 0.15, "high": 0.35}


def make_lr():
    return LogisticRegression(class_weight="balanced", max_iter=2000)


def build_service():
    X, y = split_xy(make_dataset())
    model = CalibratedClassifierCV(build_pipeline(make_lr(), scale_numeric=True), method="sigmoid", cv=3).fit(X, y)
    pipe = build_pipeline(make_lr(), scale_numeric=True).fit(X, y)
    mean = pipe[:-1].transform(X).mean()
    return AttritionService(model, pipe, mean, {"threshold": 0.3, "risk_bands": BANDS}), X, pipe, mean


def test_risk_level_boundaries():
    assert risk_level(0.10, BANDS) == "Low"
    assert risk_level(0.15, BANDS) == "Medium"
    assert risk_level(0.34, BANDS) == "Medium"
    assert risk_level(0.35, BANDS) == "High"


def test_contributions_add_up_to_the_models_log_odds():
    _, X, pipe, mean = build_service()
    contributions = linear_contributions(pipe, mean, X)
    lr = pipe.named_steps["model"]
    total = contributions.sum(axis=1) + lr.intercept_[0] + float((mean.to_numpy() * lr.coef_[0]).sum())
    np.testing.assert_allclose(total.to_numpy(), pipe.decision_function(X))


def test_service_scores_and_explains():
    service, X, _, _ = build_service()
    scores = service.score(X.head(5))
    assert list(scores.columns) == ["probability", "risk_level", "flagged"]
    assert scores["probability"].between(0, 1).all()
    assert set(scores["risk_level"]) <= {"Low", "Medium", "High"}
    drivers = service.drivers(X.head(2), k=3)
    assert len(drivers) == 2 and all(len(d) == 3 for d in drivers)
    assert set(drivers[0][0]) == {"feature", "contribution", "direction"}


def test_simulation_reports_before_and_after():
    service, X, _, _ = build_service()
    result = service.simulate(X.iloc[0].to_dict(), {"OverTime": "Yes"})
    assert set(result) == {"before", "after", "change_in_probability", "note"}
    assert "not" in result["note"].lower()
