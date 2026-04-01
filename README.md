# 💊 Pharma Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **End-to-end machine learning pipeline** for pharmaceutical drug sales forecasting using time-series analysis, feature engineering, and ensemble models (XGBoost + Random Forest) — with a production-ready FastAPI inference endpoint.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline](#-pipeline)
- [Quick Start](#-quick-start)
- [API Usage](#-api-usage)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🔍 Overview

This project builds a complete forecasting pipeline for 8 pharmaceutical drug categories using real-world transactional sales data. The pipeline covers exploratory data analysis, advanced feature engineering (lag features, rolling statistics, cyclical encodings), ensemble model training with hyperparameter tuning, and deployment via a REST API.

**Key capabilities:**
- Multi-granularity data (hourly → daily → weekly → monthly)
- Time-aware train/test split to prevent future leakage
- XGBoost and Random Forest with RandomizedSearchCV tuning
- FastAPI `/predict_sales` endpoint for real-time inference
- Feature importance analysis and model persistence

---

## 📊 Dataset

Sales data across **8 ATC-coded drug categories** spanning multiple years:

| Code | Class | Example Drug | Granularity |
|------|-------|-------------|-------------|
| M01AB | Anti-inflammatory | Diclofenac | Hourly / Daily / Weekly / Monthly |
| M01AE | Anti-inflammatory | Ibuprofen | Hourly / Daily / Weekly / Monthly |
| N02BA | Analgesics | Aspirin | Hourly / Daily / Weekly / Monthly |
| N02BE | Analgesics | Paracetamol | Hourly / Daily / Weekly / Monthly |
| N05B | Anxiolytics | Diazepam | Hourly / Daily / Weekly / Monthly |
| N05C | Hypnotics / Sedatives | Zolpidem | Hourly / Daily / Weekly / Monthly |
| R03 | Anti-asthma | Salbutamol | Hourly / Daily / Weekly / Monthly |
| R06 | Antihistamines | Loratadine | Hourly / Daily / Weekly / Monthly |

**Data files:**

| File | Description |
|------|-------------|
| `saleshourly.csv` | Hourly sales records |
| `salesdaily.csv` | Daily sales with weekday metadata |
| `salesweekly.csv` | Weekly aggregated sales |
| `salesmonthly.csv` | Monthly aggregated sales |

---

## 📁 Project Structure

```
pharma-demand-forecasting/
│
├── 📓 eda.ipynb                    # Exploratory Data Analysis notebook
├── 📓 feature_engineering.ipynb    # Feature pipeline notebook
├── 📓 model_training.ipynb         # Model training & evaluation notebook
│
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore
├── 📄 README.md
│
├── 📊 saleshourly.csv              # Raw data — hourly granularity
├── 📊 salesdaily.csv               # Raw data — daily with weekday info
├── 📊 salesweekly.csv              # Raw data — weekly aggregated
└── 📊 salesmonthly.csv             # Raw data — monthly aggregated
```

---

## 🔧 Pipeline

```
Raw CSV Data
    │
    ▼
┌─────────────────────────────┐
│  1. EDA (eda.ipynb)         │  → Distribution plots, trends,
│     - Shape & types         │    correlation matrix, seasonality
│     - Missing value audit   │    analysis, YoY comparison
│     - Pattern analysis      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Feature Engineering             │  → Enriched feature matrix
│     (feature_engineering.ipynb)     │    ready for ML
│     - Imputation                    │
│     - Calendar & cyclical features  │
│     - Lag features (1,2,3,6,12 mo.) │
│     - Rolling mean / std            │
│     - Cross-drug aggregates         │
│     - StandardScaler (train only)   │
│     - Time-aware train/test split   │
└──────────────┬──────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  3. Model Training                     │  → Trained models + API
│     (model_training.ipynb)             │
│     - XGBoost Regressor               │
│     - Random Forest Regressor          │
│     - RandomizedSearchCV tuning        │
│     - Metrics: RMSE, MAE, R²          │
│     - Feature importance plots         │
│     - joblib model persistence         │
│     - FastAPI /predict_sales endpoint  │
└────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/kahledawad2019-netizen/pharma-demand-forecasting.git
cd pharma-demand-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks in order

Open and run the notebooks sequentially in Jupyter:

```
1. eda.ipynb                  → Explore the data
2. feature_engineering.ipynb  → Build feature matrix
3. model_training.ipynb       → Train models & launch API
```

```bash
jupyter notebook
```

---

## 🌐 API Usage

After running `model_training.ipynb`, the FastAPI server starts automatically. You can query it via:

```bash
curl -X POST http://localhost:8000/predict_sales \
  -H "Content-Type: application/json" \
  -d '{
    "drug_name": "N02BE",
    "category": "Analgesics",
    "month": 10,
    "previous_sales": 1100.0
  }'
```

**Response:**

```json
{
  "predicted_sales": 1234.56,
  "model": "XGBRegressor",
  "drug_name": "N02BE",
  "month": 10,
  "confidence_note": "This prediction uses approximated lag features..."
}
```

Interactive API docs are available at: `http://localhost:8000/docs`

---

## 📈 Results

The pipeline evaluates models using three metrics across all 8 drug categories:

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **MAE** | Mean Absolute Error — average prediction deviation |
| **R²** | Coefficient of determination — explained variance |

Models trained: **XGBoost Regressor** and **Random Forest Regressor**, with the best-performing model selected automatically and persisted via `joblib`.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Visualization | matplotlib, seaborn |
| API | FastAPI, uvicorn |
| Model Persistence | joblib |
| Notebooks | Jupyter |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ for pharmaceutical demand intelligence
</div>
