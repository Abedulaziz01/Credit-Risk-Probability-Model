import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
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
from sklearn.pipeline import Pipeline

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DATA_PATH,
    DEFAULT_RAW_DATA_CANDIDATES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLRUNS_DIR,
    resolve_existing_path,
)
from src.data_processing import (
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    get_feature_columns,
    load_data,
    prepare_model_dataframe,
    save_data,
)


def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def build_pipeline(model):
    preprocessor = build_preprocessing_pipeline()
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def configure_mlflow() -> None:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    tracking_uri = MLRUNS_DIR.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def register_best_model(model_uri: str) -> str:
    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(MLFLOW_MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        pass

    registered_model = mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)
    client.set_registered_model_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS, registered_model.version)
    return registered_model.version


def train_models(data_path: str, output_model_path: str, processed_output_path: str):
    configure_mlflow()
    df = load_data(data_path)
    df_model = prepare_model_dataframe(df)
    save_data(df_model, processed_output_path)

    features = df_model[get_feature_columns()]
    target = df_model[TARGET_COLUMN]

    if target.nunique() < 2:
        raise ValueError("The proxy target contains only one class. Training requires both risk classes.")

    class_counts = target.value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError("The proxy target has too few minority-class samples to train reliably.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    cv_folds = max(2, min(5, min_class_count))

    experiments = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10, 20],
                "classifier__min_samples_split": [2, 4],
            },
        ),
    }

    best_score = -np.inf
    best_model = None
    best_name = None
    best_metrics = {}
    best_model_uri = None
    best_run_id = None

    for name, (estimator, params) in experiments.items():
        pipeline = build_pipeline(estimator)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring="roc_auc",
            cv=cv_folds,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        preds = search.predict(X_test)
        probs = search.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, preds, probs)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_param("feature_count", len(get_feature_columns()))
            mlflow.log_param("positive_rate", float(target.mean()))
            model_info = mlflow.sklearn.log_model(search.best_estimator_, name="model")
            run_id = mlflow.active_run().info.run_id
            model_uri = model_info.model_uri

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = search.best_estimator_
            best_name = name
            best_metrics = metrics
            best_model_uri = model_uri
            best_run_id = run_id

    os.makedirs(Path(output_model_path).parent, exist_ok=True)
    joblib.dump(best_model, output_model_path)

    registered_version = register_best_model(best_model_uri)

    return {
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "registered_model_name": MLFLOW_MODEL_NAME,
        "registered_model_alias": MLFLOW_MODEL_ALIAS,
        "registered_model_version": registered_version,
        "best_run_id": best_run_id,
        "processed_output_path": processed_output_path,
        "saved_model_path": output_model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train credit risk models")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the raw transaction dataset. If omitted, the script checks data/raw and DATA/.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--processed-output",
        type=str,
        default=str(DEFAULT_PROCESSED_DATA_PATH),
        help="Path to save the processed customer-level dataset used for training.",
    )
    args = parser.parse_args()

    input_path = args.input or str(resolve_existing_path(DEFAULT_RAW_DATA_CANDIDATES))
    result = train_models(input_path, args.output, args.processed_output)
    print(f"Best model: {result['best_model_name']}")
    print("Metrics:")
    for key, value in result["best_metrics"].items():
        print(f"  {key}: {value:.4f}")
    print(f"MLflow model: {result['registered_model_name']}@{result['registered_model_alias']}")
    print(f"MLflow version: {result['registered_model_version']}")
    print(f"Saved model: {result['saved_model_path']}")
    print(f"Processed dataset: {result['processed_output_path']}")


if __name__ == "__main__":
    main()
