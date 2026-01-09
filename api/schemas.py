from pydantic import BaseModel, Field
from typing import Dict, List


class TransactionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Transaction features as a name:value mapping"
    )


class PredictionResponse(BaseModel):
    anomaly_score: float = Field(
        ..., description="Reconstruction error from autoencoder"
    )
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized fraud probability"
    )
    is_fraud: bool = Field(
        ..., description="Fraud decision based on threshold"
    )
    risk_level: str = Field(
        ..., description="LOW | MEDIUM | HIGH | EXTREME"
    )
    reasons: List[str] = Field(
        ..., description="Human-readable reasons for the decision"
    )
