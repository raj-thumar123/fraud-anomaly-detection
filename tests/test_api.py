# tests/test_api.py

import os
import sys
import numpy as np
import pandas as pd
import joblib
from fastapi.testclient import TestClient

# ----------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# ----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ----------------------------------------------------
# IMPORT FASTAPI APP (CORRECT LOCATION)
# ----------------------------------------------------
from api.main import app

client = TestClient(app)

# ----------------------------------------------------
# LOAD SCALER & FEATURE NAMES
# ----------------------------------------------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

scaler = joblib.load(SCALER_PATH)
feature_names = list(scaler.feature_names_in_)

# ----------------------------------------------------
# HELPER: BUILD VALID API PAYLOAD (NO WARNINGS)
# ----------------------------------------------------
def build_valid_payload():
    """
    Build a valid API payload using a DataFrame to preserve feature names.
    This avoids sklearn feature-name warnings.
    """
    # zero vector with feature names
    df_scaled = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # inverse transform to raw feature space
    df_raw = pd.DataFrame(
        scaler.inverse_transform(df_scaled),
        columns=feature_names
    )

    return {
        "features": {
            col: float(df_raw.iloc[0][col])
            for col in feature_names
        }
    }

# ----------------------------------------------------
# TEST 1: VALID REQUEST
# ----------------------------------------------------
def test_valid_transaction():
    payload = build_valid_payload()
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()
    assert "anomaly_score" in body
    assert "fraud_probability" in body
    assert "is_fraud" in body
    assert "risk_level" in body
    assert "reasons" in body

# ----------------------------------------------------
# TEST 2: MISSING REQUIRED FEATURE
# ----------------------------------------------------
def test_missing_feature():
    payload = build_valid_payload()
    payload["features"].pop(feature_names[0])

    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]

# ----------------------------------------------------
# TEST 3: INVALID FEATURE TYPE
# ----------------------------------------------------
def test_invalid_feature_type():
    payload = build_valid_payload()
    payload["features"][feature_names[0]] = "invalid_value"

    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]
