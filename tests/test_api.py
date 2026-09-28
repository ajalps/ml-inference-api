"""Unit tests for the inference API: response codes + schema validation."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 35,
    "income": 55000,
    "loan_amount": 15000,
    "credit_score": 680,
    "debt_to_income": 0.25,
    "employment_type": "Salaried",
    "state": "ST03",
}


def test_health_returns_200_and_model_loaded():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_payload_returns_200_with_expected_schema():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200

    body = resp.json()
    assert set(body.keys()) == {"prediction", "probability_default", "risk_label"}
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_default"] <= 1.0
    assert body["risk_label"] in ("Low", "Medium", "High")


def test_predict_allows_missing_optional_fields():
    # Only the two required fields (loan_amount, state) are provided; the
    # pipeline's imputers should handle the rest, same as during training.
    payload = {"loan_amount": 10000, "state": "ST01"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert "probability_default" in resp.json()


def test_predict_missing_required_field_returns_422():
    payload = {"loan_amount": 10000}  # missing required 'state'
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_wrong_type_returns_422():
    payload = {**VALID_PAYLOAD, "loan_amount": "not-a-number"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_out_of_range_credit_score_returns_422():
    payload = {**VALID_PAYLOAD, "credit_score": 999}  # max allowed is 850
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_employment_type_returns_422():
    payload = {**VALID_PAYLOAD, "employment_type": "Freelancer"}  # not in the allowed set
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_unseen_state_still_returns_200():
    # The pipeline's target encoder handles unseen categories gracefully
    # (falls back to the overall target mean), so this should not error.
    payload = {**VALID_PAYLOAD, "state": "STXX"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
