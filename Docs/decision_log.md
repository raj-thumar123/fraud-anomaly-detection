# Decision Log

This document records the key technical decisions made during the project and the reasoning behind them. The goal is to document why certain approaches were selected or rejected during development.

---

## Decision 1: How to frame the fraud detection problem

- **Tried:** Treating fraud detection strictly as a binary prediction problem using the fraud label.
- **Issue:** Fraud transactions make up only ~0.17% of the data, causing models to bias heavily toward the majority (normal) class and making evaluation unstable.
- **Decision:** Frame the problem as anomaly detection, where models learn normal transaction behavior and flag deviations as potential fraud.

---

## Decision 2: Isolation Forest as the first model

- **Tried:** Isolation Forest as a baseline anomaly detection method.
- **Observed:**  
  - Good ability to isolate rare points  
  - High recall on fraudulent transactions  
  - However, many normal transactions were also flagged
- **Issue:** High number of false positives would cause customer friction and increase manual review cost.
- **Decision:** Keep Isolation Forest only as a comparison model, not for deployment.

---

## Decision 3: One-Class SVM for tighter boundaries

- **Tried:** One-Class SVM to learn a tighter boundary around normal transactions.
- **Observed:**  
  - Very high recall  
  - Predictions were unstable across different feature scales  
- **Issue:** Low precision and poor scalability make it risky for high-volume transaction systems.
- **Decision:** Use One-Class SVM for benchmarking only.

---

## Decision 4: Autoencoder for non-linear behavior modeling

- **Tried:** Autoencoder trained on non-fraudulent transactions.
- **Observed:**  
  - Stable anomaly scores  
  - Better balance between precision and recall  
  - Fraud transactions showed consistently higher reconstruction error
- **Decision:** Select autoencoder as the final model for deployment.

---

## Decision 5: Choosing an anomaly threshold

- **Tried:** Multiple anomaly score thresholds during validation.
- **Observed:**  
  - Lower thresholds increased recall but caused excessive false positives  
  - Higher thresholds reduced false alerts but missed some fraud
- **Decision:** Select a conservative threshold at the 99.3 percentile of reconstruction error to balance detection and customer impact.
