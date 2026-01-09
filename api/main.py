from fastapi import FastAPI
from api.schemas import TransactionInput, FraudPredictionResponse

app = FastAPI(
    title="Fraud Anomaly Detection API",
    description="Autoencoder-based fraud detection with explainability",
    version="1.0"
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Fraud Anomaly Detection API is running"
    }


@app.post("/predict", response_model=FraudPredictionResponse)
def predict_fraud(transaction: TransactionInput):
    """
    Dummy prediction endpoint.
    Model inference will be added in the next step.
    """

    return FraudPredictionResponse(
        fraud_probability=0.0,
        is_fraud=False,
        top_contributing_features=[]
    )
