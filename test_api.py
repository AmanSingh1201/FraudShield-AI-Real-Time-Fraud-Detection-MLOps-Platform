import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.fraudshield.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_payload():
    df = pd.read_csv("data/processed/test.csv")
    row = df[df.Class == 0].iloc[0].drop("Class").to_dict()
    row["transaction_id"] = "test_tx_1"
    return row


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    assert "model_version" in r.json()


def test_predict_valid_request(client, sample_payload):
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    for key in ("transaction_id", "fraud_probability", "risk_level", "decision",
                "model_version", "threshold_version", "latency_ms"):
        assert key in body


def test_predict_missing_field_returns_422(client, sample_payload):
    bad = dict(sample_payload)
    del bad["V10"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_malformed_type_returns_422(client, sample_payload):
    bad = dict(sample_payload)
    bad["Amount"] = "not_a_number"
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_negative_amount_rejected(client, sample_payload):
    bad = dict(sample_payload)
    bad["Amount"] = -5.0
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_batch(client, sample_payload):
    r = client.post("/predict/batch", json={"transactions": [sample_payload, sample_payload]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "n_requests" in r.json()


def test_no_stack_trace_leaked_on_error(client, sample_payload):
    bad = dict(sample_payload)
    del bad["Time"]
    r = client.post("/predict", json=bad)
    assert "Traceback" not in r.text
