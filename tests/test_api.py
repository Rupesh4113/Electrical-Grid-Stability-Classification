"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint responds with service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Electrical Grid Stability Classification API"
    assert data["status"] == "operational"


def test_health_endpoint():
    """Verify health endpoint responds with valid schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_endpoint_validation_error():
    """Verify invalid payloads trigger Pydantic validation errors (HTTP 422)."""
    # Missing parameters
    response = client.post("/predict", json={"tau1": 1.0})
    assert response.status_code == 422

    # Non-numeric value
    bad_payload = {
        "tau1": "text", "tau2": 1.0, "tau3": 1.0, "tau4": 1.0,
        "p1": 3.0, "p2": -1.0, "p3": -1.0, "p4": -1.0,
        "g1": 0.5, "g2": 0.5, "g3": 0.5, "g4": 0.5
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_endpoint_out_of_bounds_error():
    """Verify out-of-range values trigger Pydantic validation error."""
    out_of_range_payload = {
        "tau1": 999.0,  # Exceeds max bound of 20.0
        "tau2": 1.0, "tau3": 1.0, "tau4": 1.0,
        "p1": 3.0, "p2": -1.0, "p3": -1.0, "p4": -1.0,
        "g1": 0.5, "g2": 0.5, "g3": 0.5, "g4": 0.5
    }
    response = client.post("/predict", json=out_of_range_payload)
    assert response.status_code == 422
