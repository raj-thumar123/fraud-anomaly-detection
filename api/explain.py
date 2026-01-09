from typing import List, Tuple
import numpy as np


# Map features to human-readable reason buckets
FEATURE_REASON_MAP = {
    # TIME
    "hour": ("TIME", "Unusual transaction timing"),
    "day_of_week": ("TIME", "Unusual transaction timing"),
    "month": ("TIME", "Seasonal transaction anomaly"),

    # LOCATION
    "home_lat": ("LOCATION", "Transaction location deviates from customer's home"),
    "home_lon": ("LOCATION", "Transaction location deviates from customer's home"),
    "merchant_lat": ("LOCATION", "Merchant location anomaly"),
    "merchant_lon": ("LOCATION", "Merchant location anomaly"),
    "distance_from_home": ("LOCATION", "Transaction occurred far from home location"),

    # AMOUNT
    "Amount": ("AMOUNT", "Unusual transaction amount"),
}



def get_risk_level(score: float, threshold: float) -> str:
    """
    Convert anomaly score into a risk category.
    """
    if score < threshold * 0.8:
        return "LOW"
    elif score < threshold:
        return "MEDIUM"
    elif score < threshold * 3:
        return "HIGH"
    else:
        return "EXTREME"


def generate_reasons(
    feature_errors,
    feature_names,
    risk_level,
    top_k=10
):
    """
    Reasons explain WHICH directions contributed to the anomaly,
    not how dangerous it is.

    LOW     -> max 1 reason
    MEDIUM  -> max 2 reasons
    HIGH    -> all meaningful reasons
    EXTREME -> all meaningful reasons
    """

    top_indices = np.argsort(feature_errors)[::-1][:top_k]

    reasons = []
    used_domains = set()

    for idx in top_indices:
        fname = feature_names[idx]

        if fname in FEATURE_REASON_MAP:
            domain, reason = FEATURE_REASON_MAP[fname]
        else:
            domain, reason = "BEHAVIOR", "Unusual transaction behavior pattern"

        # One reason per domain
        if domain not in used_domains:
            reasons.append(reason)
            used_domains.add(domain)

    # ---- APPLY LIMITS BASED ON SEVERITY ----
    if risk_level == "LOW":
        return reasons[:1]

    if risk_level == "MEDIUM":
        return reasons[:2]

    # HIGH / EXTREME → return ALL meaningful reasons
    return reasons


