import joblib
import numpy as np
from pathlib import Path

# Paths
MODEL_DIR = Path("models")

SCALER_PATH = MODEL_DIR / "scaler.pkl"
AUTOENCODER_PATH = MODEL_DIR / "autoencoder.pkl"


def load_artifacts():
    """
    Load scaler and autoencoder model.
    """
    scaler = joblib.load(SCALER_PATH)
    autoencoder = joblib.load(AUTOENCODER_PATH)

    return scaler, autoencoder
