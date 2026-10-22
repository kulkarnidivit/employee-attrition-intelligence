import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import EmployeeInput
from src.config import MODEL_PATH
from src.serving.demo_data import DEMO_DATA_PATH
from src.serving.service import review_areas
from tests.helpers import make_row


@pytest.fixture(scope="module")
def client():
    if not MODEL_PATH.exists() or not DEMO_DATA_PATH.exists():
        pytest.skip("model or demo data missing; run train_final and src.serving.make_demo_data")
    with TestClient(app) as test_client:
        yield test_client


def payload(**overrides):
    row = {name: make_row(1)[name] for name in EmployeeInput.model_fields}
    row.update(overrides)
    return row


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_probability_and_risk_level(client):
    body = client.post("/predict", json=payload()).json()
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert isinstance(body["flagged"], bool)


def test_invalid_inputs_are_rejected(client):
    assert client.post("/predict", json=payload(OverTime="Sometimes")).status_code == 422
    assert client.post("/predict", json=payload(JobSatisfaction=9)).status_code == 422
    assert client.post("/predict", json=payload(YearsInCurrentRole=20)).status_code == 422
    missing = payload()
    missing.pop("MonthlyIncome")
    assert client.post("/predict", json=missing).status_code == 422


def test_explain_gives_drivers_and_a_careful_note(client):
    body = client.post("/explain", json=payload(OverTime="Yes", JobLevel=1)).json()
    assert len(body["drivers"]) == 5
    assert "not causes" in body["note"]


def test_simulate_compares_before_and_after(client):
    body = client.post("/simulate", json={"employee": payload(OverTime="Yes"), "changes": {"OverTime": "No"}}).json()
    assert set(body) == {"before", "after", "change_in_probability", "note"}
    assert body["after"]["probability"] <= body["before"]["probability"]


def test_simulate_rejects_unknown_or_invalid_changes(client):
    assert client.post("/simulate", json={"employee": payload(), "changes": {"Foo": 1}}).status_code == 422
    assert client.post("/simulate", json={"employee": payload(), "changes": {"JobSatisfaction": 9}}).status_code == 422


def test_review_areas_only_use_factors_that_raise_the_score():
    drivers = [
        {"feature": "OverTime_Yes", "contribution": 0.5, "direction": "raises"},
        {"feature": "JobSatisfaction", "contribution": -0.2, "direction": "lowers"},
        {"feature": "unknown_feature", "contribution": 0.1, "direction": "raises"},
    ]
    assert review_areas(drivers) == ["Workload and scheduling review"]


def test_summary_has_three_risk_levels_and_they_separate(client):
    body = client.get("/summary").json()
    levels = body["by_risk_level"]
    assert [x["risk_level"] for x in levels] == ["Low", "Medium", "High"]
    assert body["employees"] > 0
    assert levels[2]["actual_attrition_rate"] > levels[0]["actual_attrition_rate"]


def test_global_drivers(client):
    body = client.get("/drivers/global").json()
    assert len(body) == 15
    assert set(body[0]) == {"feature", "importance"}


def test_employee_list_filters_and_paginates(client):
    body = client.get("/employees", params={"risk_level": "High", "limit": 5}).json()
    assert body["total"] > 0 and len(body["items"]) <= 5
    assert all(item["risk_level"] == "High" for item in body["items"])
    probabilities = [item["probability"] for item in body["items"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert client.get("/employees", params={"risk_level": "Extreme"}).status_code == 422


def test_employee_detail_and_not_found(client):
    first = client.get("/employees", params={"limit": 1}).json()["items"][0]["EmployeeNumber"]
    detail = client.get(f"/employees/{first}").json()
    assert set(detail) == {"employee_number", "in_training", "profile", "score", "drivers", "review_areas", "record"}
    assert set(detail["record"]) == set(EmployeeInput.model_fields)
    repeat = client.post("/predict", json=detail["record"]).json()
    assert repeat["probability"] == pytest.approx(detail["score"]["probability"])
    assert client.get("/employees/999999").status_code == 404


def test_api_key_is_required_when_configured(client, monkeypatch):
    monkeypatch.setattr("api.main.API_KEY", "secret")
    assert client.get("/summary").status_code == 401
    assert client.get("/summary", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_cors_allows_the_frontend_origin(client):
    response = client.options(
        "/predict",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_model_info_returns_the_model_card(client):
    from src.config import MODEL_CARD_PATH

    if not MODEL_CARD_PATH.exists():
        pytest.skip("model card not built; run: python -m src.serving.make_model_card")
    body = client.get("/model/info").json()
    assert {"model", "test", "test_calibration", "cv_comparison", "risk_bands_oof"} <= set(body)


def test_model_info_returns_the_model_card(client):
    from src.config import MODEL_CARD_PATH

    if not MODEL_CARD_PATH.exists():
        pytest.skip("model card not built; run: python -m src.serving.make_model_card")
    body = client.get("/model/info").json()
    assert {"model", "test", "test_calibration", "cv_comparison", "risk_bands_oof"} <= set(body)
