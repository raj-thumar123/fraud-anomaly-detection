# Fraud Anomaly Detection System

This project implements a **production-ready fraud detection system** using **unsupervised anomaly detection techniques**.  
The system learns normal transaction behavior and flags deviations as potential fraud, making it suitable for **highly imbalanced real-world financial datasets**.

The project includes feature enrichment, model comparison, FastAPI deployment, unit testing, configuration management, and detailed documentation.

---

## 📌 Dataset Information

### Dataset Used
**Credit Card Fraud Detection Dataset (Kaggle)**  
🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- Transactions made by European cardholders  
- Extremely imbalanced dataset (~0.17% fraud)  
- Features are PCA-transformed (`V1`–`V28`)  
- Fraud labels are used **only for evaluation**, not training  

---

## 📥 Dataset Setup (IMPORTANT)

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Extract the file `creditcard.csv`

3. Create a folder named `data/` in the project root

4. Place the dataset here:

```
data/creditcard.csv
```

> ⚠️ The notebooks assume this exact file location.

---

## ⚙️ Setup Instructions

### 1️⃣ Create and activate virtual environment

**Windows**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Feature Engineering & Enrichment

- Original PCA features (`V1`–`V28`) are retained
- Additional **synthetic but realistic features** are added:
  - Temporal: `hour`, `day_of_week`, `month`
  - Spatial: `home_lat`, `home_lon`, `merchant_lat`, `merchant_lon`
  - Behavioral: `distance_from_home`

This makes the dataset **non-standard** and closer to real production data.

---

## 🤖 Models Implemented

The following anomaly detection models were evaluated:

- Isolation Forest  
- One-Class SVM  
- Autoencoder (**final selected model**)  

The autoencoder was selected due to:
- Better precision–recall balance  
- Lower false positives  
- Stable anomaly scores  
- Suitability for real-time inference  

---

## 🚀 Running the API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

---

## 🔍 API Usage

### Health Check
```http
GET /
```

Response:
```json
{
  "status": "ok"
}
```

---

### Fraud Prediction
```http
POST /predict
```

#### Example Request
```json
{
  "features": {
    "Time": 42851,
    "Amount": 28.0,
    "V1": 1.34,
    "...": "...",
    "home_lat": 52.20,
    "home_lon": 11.28,
    "merchant_lat": 56.80,
    "merchant_lon": -6.60,
    "distance_from_home": 1258.35,
    "hour": 11,
    "day_of_week": 6,
    "month": 9
  }
}
```

> All features defined in `src/config.py` must be present.

---

#### Example Response
```json
{
  "anomaly_score": 0.87,
  "fraud_probability": 0.71,
  "is_fraud": true,
  "risk_level": "HIGH",
  "reasons": [
    "Unusual transaction distance",
    "Abnormal temporal behavior"
  ]
}
```

---

## 🧪 Running Unit Tests

Unit tests validate API behavior, input validation, and error handling.

```bash
pytest
```

Expected output:
```
3 passed
```

---

## 🧠 Configuration Management

All configurable values are centralized in:

```
src/config.py
```

This includes:
- Feature schema
- Model file paths
- Anomaly detection threshold

System behavior can be modified **without changing API logic**.

---

## 📊 Documentation & Process

- **Decision Log:** documents design choices and rejected approaches  
- **Jupyter Notebooks:** show EDA, feature engineering, and model iterations  
- **Technical Report:** explains methodology, results, edge cases, and deployment considerations  

---

## ⚠️ Constraints & Design Decisions

- Target inference latency: < 100ms  
- False positives are treated as more costly than false negatives  
- Models trained only on normal transactions  
- Designed for real-time deployment  

---

## 👤 Author

Raj Thumar  
M.Tech – Data Science  
IIT Guwahati
