import joblib
import numpy as np
import os
from typing import List

# Absolute paths work both locally and on Hugging Face
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


class ModelRegistry:
    scaler = None
    autoencoder = None
    threshold = None
    feature_names: List[str] = []


def load_models():
    """
    Load all ML artifacts once at startup.
    This function MUST be called only once.
    """

    if ModelRegistry.scaler is None:
        ModelRegistry.scaler = joblib.load(
            os.path.join(MODEL_DIR, "scaler.pkl")
        )

    if ModelRegistry.autoencoder is None:
        ModelRegistry.autoencoder = joblib.load(
            os.path.join(MODEL_DIR, "autoencoder.pkl")
        )

    # Source of truth for feature order
    ModelRegistry.feature_names = list(
        ModelRegistry.scaler.feature_names_in_
    )

    # ---- FRAUD THRESHOLD ----
    # Derived offline from validation set (99.3 percentile)
    # DO NOT recompute at runtime
    ModelRegistry.threshold = 1.25  # <-- stable, conservative value

    return ModelRegistry
