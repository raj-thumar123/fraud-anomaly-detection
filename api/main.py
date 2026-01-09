from fastapi import FastAPI, HTTPException
import numpy as np
import logging
from datetime import datetime

from api.schemas import TransactionRequest, PredictionResponse
from api.model_loader import load_models
from api.explain import generate_reasons

# ----------------------------------------------------
# LOGGING SETUP (Logging = Monitoring for assignment)
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("fraud_api")

# ----------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------
app = FastAPI(
    title="Fraud Anomaly Detection API",
    version="1.0.0"
)

# ----------------------------------------------------
# LOAD MODEL ARTIFACTS (ONCE)
# ----------------------------------------------------
models = load_models()
logger.info("Model and scaler loaded successfully")


# ----------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok"}


# ----------------------------------------------------
# PREDICTION ENDPOINT
# ----------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):

    try:
        input_features = request.features
        expected_features = models.feature_names

        # ---------- REQUEST VALIDATION ----------
        if set(input_features.keys()) != set(expected_features):
            missing = list(set(expected_features) - set(input_features.keys()))
            extra = list(set(input_features.keys()) - set(expected_features))

            logger.warning(
                {
                    "event": "validation_error",
                    "missing_features": missing,
                    "extra_features": extra
                }
            )

            raise HTTPException(
                status_code=400,
                detail={
                    "missing_features": missing,
                    "extra_features": extra
                }
            )

        # ---------- ORDER FEATURES ----------
        x = np.array(
            [input_features[f] for f in expected_features]
        ).reshape(1, -1)

        # ---------- SCALE ----------
        x_scaled = models.scaler.transform(x)

        # ---------- RECONSTRUCT ----------
        x_hat = models.autoencoder.predict(x_scaled)

        # ---------- ANOMALY SCORE ----------
        feature_errors = np.square(x_scaled - x_hat).flatten()
        anomaly_score = float(np.mean(feature_errors))

        # ---------- FRAUD DECISION ----------
        is_fraud = anomaly_score > models.threshold

        # ---------- SEVERITY (SCORE ONLY) ----------
        if anomaly_score < models.threshold * 0.8:
            risk_level = "LOW"
        elif anomaly_score < models.threshold:
            risk_level = "MEDIUM"
        elif anomaly_score < models.threshold * 3:
            risk_level = "HIGH"
        else:
            risk_level = "EXTREME"

        # ---------- SOFT FRAUD PROBABILITY ----------
        normalized_score = (anomaly_score - models.threshold) / models.threshold
        fraud_probability = float(1 / (1 + np.exp(-normalized_score)))
        fraud_probability = max(0.0, min(fraud_probability, 1.0))

        # ---------- INTERPRETATION (DIRECTION ONLY) ----------
        reasons = generate_reasons(
            feature_errors=feature_errors,
            feature_names=models.feature_names,
            risk_level=risk_level
        )

        # ---------- LOG PREDICTION (MONITORING) ----------
        logger.info(
            {
                "event": "prediction",
                "timestamp": datetime.utcnow().isoformat(),
                "anomaly_score": anomaly_score,
                "fraud_probability": fraud_probability,
                "is_fraud": is_fraud,
                "risk_level": risk_level,
                "num_reasons": len(reasons)
            }
        )

        return PredictionResponse(
            anomaly_score=anomaly_score,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            risk_level=risk_level,
            reasons=reasons
        )

    # ---------- EXPECTED ERRORS ----------
    except HTTPException:
        raise

    # ---------- UNEXPECTED ERRORS ----------
    except Exception as e:
        logger.error(
            {
                "event": "internal_error",
                "error": str(e),
                "endpoint": "/predict"
            }
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
