from pydantic import BaseModel
from typing import List


class TransactionInput(BaseModel):
    features: List[float]


class FraudPredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    top_contributing_features: List[str]
