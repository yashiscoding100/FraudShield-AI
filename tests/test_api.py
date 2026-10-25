"""
Unit and integration tests for FastAPI endpoints.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

from api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert data["feature_pipeline"] is True
    assert data["explainer_ready"] is True


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "pr_auc" in data
    assert "roc_auc" in data
    assert data["pr_auc"] > 0.75


def test_invalid_prediction_payload(client):
    # Missing required 'Amount' field or negative amount
    response = client.post("/predict", json={"Time": 100.0, "Amount": -10.0})
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation failure)


def test_predict_real_transaction(client):
    # Test legitimate transaction
    payload = {
        "transaction_id": "TXN-UNITTEST-001",
        "Time": 45000.0,
        "Amount": 45.50,
        "V14": 0.1,
        "V12": -0.2,
        "V10": 0.05,
        "V4": -0.1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["transaction_id"] == "TXN-UNITTEST-001"
    assert "risk_score" in res
    assert 0.0 <= res["risk_score"] <= 100.0
    assert res["risk_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert len(res["top_risk_drivers"]) > 0
    assert "investigator_dossier" in res
    assert "executive_summary" in res["investigator_dossier"]
