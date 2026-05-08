# Credit-Risk-Probability-Model

## Credit Scoring Business Understanding

### Basel II and Interpretability
The Basel II Accord requires banks to measure credit risk rigorously and document model decisions. A credit risk model in this context must therefore be interpretable, auditable, and transparent so the bank can explain how scores are generated and defend decisions to regulators.

### Why a proxy target is necessary
The dataset does not include a true default label, so we create an engagement-based proxy target using RFM and clustering. This is necessary to build a supervised model, but it also introduces business risk because the proxy may not perfectly reflect actual loan defaults and could misclassify customers based on behavior that is not credit risk.

### Simple vs. complex models in finance
Simple models like Logistic Regression with Weight of Evidence are easier to explain, validate, and audit. Complex models such as Gradient Boosting can improve predictive power, but they are harder to interpret and may be more difficult to justify in a regulated financial context. The key trade-off is between explainability and performance.

## Main project features

- Automated feature engineering pipeline for customer transaction data
- Proxy target construction using RFM and KMeans clustering
- Customer-level aggregate transaction features for risk modeling
- Model training with MLflow experiment tracking and hyperparameter search
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
- AverageTransactionDay
- AverageTransactionMonth
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

2. Train the model
```bash
python src/train.py --input data/data.xlsx --output models/best_model.joblib
```

3. Run the API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

4. Open the dashboard
- http://localhost:8000/docs

5. Run tests
```bash
pytest -q
```

6. Build and run with Docker
```bash
docker build -t credit-risk-api .
docker-compose up --build
```
