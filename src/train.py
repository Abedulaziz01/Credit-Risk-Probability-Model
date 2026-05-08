import argparse
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from src.data_processing import build_preprocessing_pipeline, prepare_model_dataframe


def evaluate(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
    }


def build_pipeline(model):
    preprocessor = build_preprocessing_pipeline()
    from sklearn.pipeline import Pipeline

    return Pipeline([('preprocessor', preprocessor), ('classifier', model)])


def train_models(data_path: str, output_model_path: str):
    df = pd.read_excel(data_path) if data_path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(data_path)
    df_model = prepare_model_dataframe(df)

    features = df_model.drop(columns=['CustomerId', 'is_high_risk'])
    target = df_model['is_high_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    experiments = {
        'logistic_regression': (
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            {'classifier__C': [0.01, 0.1, 1.0, 10.0]},
        ),
        'random_forest': (
            RandomForestClassifier(random_state=42),
            {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 10, 20],
                'classifier__min_samples_split': [2, 4],
            },
        ),
    }

    best_score = -np.inf
    best_model = None
    best_name = None
    best_metrics = {}

    mlflow.set_experiment('CreditRiskModel')

    for name, (estimator, params) in experiments.items():
        pipeline = build_pipeline(estimator)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring='roc_auc',
            cv=3,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        preds = search.predict(X_test)
        probs = search.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, preds, probs)

        with mlflow.start_run(run_name=name):
            mlflow.log_param('model', name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path='model')

        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model = search.best_estimator_
            best_name = name
            best_metrics = metrics

    os.makedirs(Path(output_model_path).parent, exist_ok=True)
    joblib.dump(best_model, output_model_path)

    return best_name, best_metrics


def main():
    parser = argparse.ArgumentParser(description='Train credit risk models')
    parser.add_argument(
        '--input',
        type=str,
        default='data/data.xlsx',
        help='Path to raw transaction dataset',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/best_model.joblib',
        help='Path to save the trained model',
    )
    args = parser.parse_args()

    model_name, metrics = train_models(args.input, args.output)
    print(f'Best model: {model_name}')
    print('Metrics:')
    for key, value in metrics.items():
        print(f'  {key}: {value:.4f}')


if __name__ == '__main__':
    main()
