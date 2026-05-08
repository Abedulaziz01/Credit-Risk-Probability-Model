# Credit-Risk-Probability-Model

## Credit Scoring Business Understanding

### Basel II and Interpretability
The Basel II Accord requires banks to measure credit risk rigorously and document model decisions. A credit risk model in this context must therefore be interpretable, auditable, and transparent so the bank can explain how scores are generated and defend decisions to regulators.

### Why a proxy target is necessary
The dataset does not include a true default label, so we create an engagement-based proxy target using RFM and clustering. This is necessary to build a supervised model, but it also introduces business risk because the proxy may not perfectly reflect actual loan defaults and could misclassify customers based on behavior that is not credit risk.

### Simple vs. complex models in finance
Simple models like Logistic Regression with Weight of Evidence are easier to explain, validate, and audit. Complex models such as Gradient Boosting can improve predictive power, but they are harder to interpret and may be more difficult to justify in a regulated financial context. The key trade-off is between explainability and performance.

## EDA Highlights

- Customer activity is highly uneven: transaction counts and transaction values are strongly right-skewed, meaning a small number of customers drive a large share of platform volume.
- Most transactions are non-fraudulent, so `FraudResult` is a low-frequency but still useful categorical signal rather than a balanced label.
- Customer behavior changes meaningfully over time, which makes recency and temporal usage patterns important ingredients for the proxy target.
- Transaction categories and channels are diverse, so one-hot encoding is more appropriate than manual ordinal encoding for most categorical features.
- Outliers exist in transaction frequency and value, which makes robust aggregation and scaling important before clustering or fitting the classifier.

## Main project features

- Automated feature engineering pipeline for customer transaction data
- Proxy target construction using RFM and KMeans clustering
- Customer-level aggregate transaction features for risk modeling
- Model training with MLflow experiment tracking and hyperparameter search
- Credit score derivation and heuristic loan amount and duration recommendations
- FastAPI inference service with request validation
- Docker containerization and GitHub Actions CI for testing and linting

## Sample features

- TotalTransactionAmount
- AverageTransactionAmount
- TransactionCount
- StdDevTransactionAmount
- MaxTransactionAmount
- TotalValue
- AverageTransactionHour
- StdTransactionHour
- AverageTransactionDay
- StdTransactionDay
- AverageTransactionMonth
- StdTransactionMonth
- AverageTransactionYear
- recency_days
- frequency
- monetary
- CurrencyCode
- CountryCode
- ProviderId
- ProductId
- ProductCategory
- ChannelId
- PricingStrategy
- FraudResult
- is_high_risk

## How to run

1. Create and activate a virtual environment
```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Build the processed customer-level dataset
```bash
python -m src.data_processing --input DATA/data.csv --output data/processed/customer_features.csv
```

3. Train the model
```bash
python -m src.train --input DATA/data.csv --output models/best_model.joblib --processed-output data/processed/customer_features.csv
```

Training logs experiments to local MLflow tracking in `mlruns/`, saves the best model to `models/best_model.joblib`, and registers the winner in the local MLflow Model Registry as `credit_risk_model@champion`.

4. Run the API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API first tries to load `credit_risk_model@champion` from the local MLflow registry. If that alias is not available, it falls back to `models/best_model.joblib`.

5. Run the Streamlit dashboard
```bash
streamlit run streamlit_app.py
```

6. Open the apps
- API docs: http://localhost:8000/docs
- Streamlit dashboard: http://localhost:8501

7. Run tests
```bash
pytest -q
```

8. Run lint checks
```bash
flake8 src tests --max-line-length=120 --count --show-source --statistics
black --check src tests --line-length=120
```

9. Build and run with Docker
```bash
docker build -t credit-risk-api .
docker-compose up --build
```

## Example Prediction Payload

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
