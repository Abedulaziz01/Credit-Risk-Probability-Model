import argparse
from typing import Any, Dict

import joblib
import pandas as pd


def load_model(path: str):
    return joblib.load(path)


def risk_probability_to_credit_score(probability: float) -> int:
    bounded_probability = min(max(probability, 0.0), 1.0)
    return int(round(850 - (bounded_probability * 550)))


def recommend_loan_terms(data: Dict[str, Any], risk_probability: float) -> tuple[float, int]:
    transaction_count = max(float(data.get("TransactionCount", 0)), 1.0)
    total_value = max(float(data.get("TotalValue", 0.0)), 0.0)
    average_transaction_amount = max(float(data.get("AverageTransactionAmount", 0.0)), 0.0)
    recency_days = max(float(data.get("recency_days", 0.0)), 0.0)

    baseline_amount = max(average_transaction_amount * transaction_count * 0.15, total_value * 0.10, 25.0)
    recent_activity_multiplier = 1.0 if recency_days <= 30 else 0.8 if recency_days <= 90 else 0.6
    risk_multiplier = max(0.20, 1.0 - risk_probability)
    recommended_amount = round(baseline_amount * recent_activity_multiplier * risk_multiplier, 2)

    if risk_probability < 0.20:
        duration_days = 90
    elif risk_probability < 0.40:
        duration_days = 60
    elif risk_probability < 0.60:
        duration_days = 30
    else:
        duration_days = 14

    return recommended_amount, duration_days


def predict(model, data: Dict[str, Any]) -> Dict[str, Any]:
    X = pd.DataFrame([data])
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[0, 1]
    else:
        probability = float(model.predict(X))
    label = int(probability >= 0.5)
    credit_score = risk_probability_to_credit_score(probability)
    recommended_loan_amount, recommended_loan_duration_days = recommend_loan_terms(data, probability)
    return {
        "risk_probability": float(probability),
        "risk_label": label,
        "credit_score": credit_score,
        "recommended_loan_amount": recommended_loan_amount,
        "recommended_loan_duration_days": recommended_loan_duration_days,
    }


def main():
    parser = argparse.ArgumentParser(description="Run credit risk prediction")
    parser.add_argument("--model", type=str, default="models/best_model.joblib")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.csv")
    args = parser.parse_args()

    model = load_model(args.model)
    data = pd.read_csv(args.input)
    predictions = []
    for _, row in data.iterrows():
        predictions.append(predict(model, row.to_dict()))

    pd.DataFrame(predictions).to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
