# Credit Risk Probability Model

Production-oriented credit risk scoring project for Bati Bank's buy-now-pay-later partnership with an eCommerce platform. The system transforms customer transaction behavior into a proxy credit-risk label, trains a supervised model, tracks experiments with MLflow, serves predictions through FastAPI, and provides an interactive Streamlit dashboard for decision support.

## Launch The Dashboard

**Run this command to open the Streamlit app:**

**`.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py`**

After startup, open `http://localhost:8501`.

## Project Overview

This repository covers the full workflow required in the task:

- business framing for proxy-based credit scoring
- exploratory analysis and customer-behavior insight generation
- reproducible feature engineering in Python scripts
- RFM-based proxy target creation with KMeans clustering
- model training, evaluation, and MLflow registration
- FastAPI deployment for inference
- Streamlit dashboard for interactive scoring
- unit tests, linting, Docker support, and CI

## Credit Scoring Business Understanding

### Basel II and interpretability

Basel II emphasizes disciplined credit-risk measurement, documentation, and governance. In practice, this means the bank cannot rely on a model that is only accurate; it must also be explainable enough to justify approvals, rejections, and pricing decisions to internal stakeholders, auditors, and regulators. That is why this project keeps the data pipeline explicit, logs experiments, and documents the proxy-label design.

### Why a proxy target is necessary

The source dataset does not include a real loan-default outcome. Because supervised credit models require a target, this project creates a behavioral proxy using Recency, Frequency, and Monetary patterns. Customers in the least-engaged cluster are treated as higher-risk proxies. This makes modeling possible, but it also introduces business risk: a proxy is not the same thing as actual default, so false signals can lead to unfair denials, weak approvals, or mispriced loans.

### Simple vs. complex models in a regulated setting

Simple models such as Logistic Regression with Weight of Evidence are easier to explain, validate, and audit. More complex models such as Random Forest or Gradient Boosting can capture nonlinear relationships and may perform better, but they are harder to interpret and govern. In a regulated environment, the best choice is not always the most accurate model in isolation; it is the model that offers a credible balance between performance, transparency, stability, and operational trust.

## EDA Highlights

- Customer activity is highly skewed, with a small number of customers generating a large share of total transaction volume.
- Transaction values and counts contain strong outliers, so aggregated features and scaling matter for both clustering and classification.
- Temporal behavior is informative, which supports the inclusion of recency and transaction-timing features.
- Product categories and channels are diverse, making one-hot encoding a better fit than manual ordinal mappings for most categorical inputs.
- Fraud cases are rare, so `FraudResult` is useful as a signal but not as a target.

## Solution Architecture

### 1. Data processing

Raw transaction data is converted into customer-level features, including:

- total transaction amount
- average transaction amount
- transaction count
- transaction amount standard deviation
- maximum transaction amount
- total transaction value
- average and standard deviation of transaction hour
- average and standard deviation of transaction day
- average and standard deviation of transaction month
- average transaction year
- RFM metrics: `recency_days`, `frequency`, `monetary`

### 2. Proxy target engineering

The target column `is_high_risk` is created by:

1. computing RFM metrics for each customer
2. scaling the RFM features
3. clustering customers into 3 groups with KMeans
4. selecting the least-engaged cluster as the high-risk proxy segment

### 3. Modeling

The training script evaluates multiple models, tunes them with grid search, logs metrics to MLflow, and registers the best model under:

- registered model name: `credit_risk_model`
- alias: `champion`

### 4. Inference outputs

For a new customer, the system returns:

- risk probability
- binary risk label
- derived credit score
- recommended loan amount
- recommended loan duration

## Project Structure

```text
Credit-Risk-Probability-Model/
├── .github/workflows/ci.yml
├── DATA/
├── notebooks/
├── src/
│   ├── api/
│   ├── config.py
│   ├── data_processing.py
│   ├── model_loader.py
│   ├── predict.py
│   └── train.py
├── tests/
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Create and activate the environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Build the processed dataset

```powershell
.\.venv\Scripts\python.exe -m src.data_processing --input DATA\data.csv --output data\processed\customer_features.csv
```

### 3. Train and register the model

```powershell
.\.venv\Scripts\python.exe -m src.train --input DATA\data.csv --output models\best_model.joblib --processed-output data\processed\customer_features.csv
```

### 4. Run the FastAPI service

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`.

### 5. Run the Streamlit dashboard

**`.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py`**

Open `http://localhost:8501`.

## Streamlit Dashboard

The dashboard provides:

- single-customer scoring from an interactive sidebar
- credit score and loan recommendation display
- batch CSV upload for bulk scoring
- downloadable prediction output
- preview of the processed reference dataset

## API

### Endpoints

- `GET /` returns service metadata
- `GET /health` returns health status and the active model source
- `POST /predict` returns risk probability, label, credit score, and loan recommendation

### Example request payload

```json
{
  "CurrencyCode": "UGX",
  "CountryCode": 256,
  "ProviderId": "ProviderId_4",
  "ProductId": "ProductId_6",
  "ProductCategory": "financial_services",
  "ChannelId": "ChannelId_2",
  "PricingStrategy": 2,
  "FraudResult": 0,
  "TotalTransactionAmount": 12500.0,
  "AverageTransactionAmount": 625.0,
  "TransactionCount": 20,
  "StdDevTransactionAmount": 240.5,
  "MaxTransactionAmount": 1500.0,
  "TotalValue": 12500.0,
  "AverageTransactionHour": 14.0,
  "StdTransactionHour": 4.2,
  "AverageTransactionDay": 15.0,
  "StdTransactionDay": 6.1,
  "AverageTransactionMonth": 7.0,
  "StdTransactionMonth": 2.0,
  "AverageTransactionYear": 2018.0,
  "recency_days": 12.0,
  "frequency": 20,
  "monetary": 12500.0
}
```

## Testing And Quality Checks

Run the automated checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m flake8 src tests streamlit_app.py --max-line-length=120 --count --show-source --statistics
.\.venv\Scripts\python.exe -m black --check src tests streamlit_app.py --line-length=120
```

## Docker

```powershell
docker build -t credit-risk-api .
docker-compose up --build
```

## Key Outputs

- processed dataset: `data/processed/customer_features.csv`
- trained model: `models/best_model.joblib`
- MLflow tracking: `mlruns/`
- Streamlit app: `streamlit_app.py`

## Current Verified Status

The repository has been verified locally with:

- processed dataset generation working
- model training working
- MLflow model registration working
- FastAPI startup and prediction working
- Streamlit model loading working
- unit tests passing
- lint and formatting checks passing
