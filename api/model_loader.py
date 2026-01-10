import os
import joblib
import numpy as np
from dataclasses import dataclass

from src.config import (
    FEATURES,
    MODEL_PATH,
    SCALER_PATH,
    ANOMALY_THRESHOLD
)

# ----------------------------------------------------
# BASE DIRECTORY
# ----------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ----------------------------------------------------
# MODEL REGISTRY
# ----------------------------------------------------
@dataclass
class ModelRegistry:
    scaler: object
    autoencoder: object
    feature_names: list
    threshold: float


# ----------------------------------------------------
# LOAD MODELS (ONCE AT STARTUP)
# ----------------------------------------------------
def load_models() -> ModelRegistry:
    """
    Loads all model artifacts and configuration values.
    Acts as a single source of truth for the API.
    """

    # Load scaler
    scaler = joblib.load(
        os.path.join(BASE_DIR, SCALER_PATH)
    )

    # Load autoencoder
    autoencoder = joblib.load(
        os.path.join(BASE_DIR, MODEL_PATH)
    )

    # Register everything
    registry = ModelRegistry(
        scaler=scaler,
        autoencoder=autoencoder,
        feature_names=FEATURES,
        threshold=ANOMALY_THRESHOLD
    )

    return registry
