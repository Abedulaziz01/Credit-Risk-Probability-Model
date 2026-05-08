import argparse
from typing import Any, Dict

import joblib
import pandas as pd


def load_model(path: str):
    return joblib.load(path)


def predict(model, data: Dict[str, Any]):
    X = pd.DataFrame([data])
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(X)[0, 1]
    else:
        probability = float(model.predict(X))
    label = int(probability >= 0.5)
    return probability, label


def main():
    parser = argparse.ArgumentParser(description='Run credit risk prediction')
    parser.add_argument('--model', type=str, default='models/best_model.joblib')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='predictions.csv')
    args = parser.parse_args()

    model = load_model(args.model)
    data = pd.read_csv(args.input)
    predictions = []
    for _, row in data.iterrows():
        probability, label = predict(model, row.to_dict())
        predictions.append({'risk_probability': probability, 'risk_label': label})

    pd.DataFrame(predictions).to_csv(args.output, index=False)
    print(f'Predictions saved to {args.output}')


if __name__ == '__main__':
    main()
