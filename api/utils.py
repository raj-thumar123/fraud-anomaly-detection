import numpy as np
from typing import List


def compute_reconstruction_error(x: np.ndarray, x_recon: np.ndarray) -> float:
    """
    Mean squared reconstruction error.
    """
    return np.mean((x - x_recon) ** 2)


def compute_feature_errors(x: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
    """
    Per-feature reconstruction error.
    """
    return (x - x_recon) ** 2


def normalize_score(score: float, max_score: float) -> float:
    """
    Convert reconstruction error to probability-like score [0,1].
    """
    return min(score / max_score, 1.0)


def top_k_features(
    feature_errors: np.ndarray,
    feature_names: List[str],
    k: int = 3
) -> List[str]:
    """
    Return top-k features contributing most to anomaly.
    """
    idx = np.argsort(feature_errors)[::-1][:k]
    return [feature_names[i] for i in idx]
def preprocess_raw_transaction(
    transaction: dict,
    feature_names: list,
    scaler
):
    """
    Convert raw transaction dict into scaled feature vector.
    """
    try:
        x = np.array([transaction[feat] for feat in feature_names]).reshape(1, -1)
    except KeyError as e:
        raise ValueError(f"Missing required feature: {e}")

    x_scaled = scaler.transform(x)
    return x_scaled
