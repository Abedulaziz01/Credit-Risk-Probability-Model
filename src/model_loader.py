import os
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.sklearn
from joblib import load as load_joblib

from src.config import MLFLOW_MODEL_ALIAS, MLFLOW_MODEL_NAME, MLRUNS_DIR


MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.resolve().as_uri())
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", MLFLOW_MODEL_NAME)
MLFLOW_REGISTERED_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", MLFLOW_MODEL_ALIAS)


def load_registered_or_local_model() -> Tuple[object, str]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

    try:
        model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{MLFLOW_REGISTERED_MODEL_ALIAS}"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        return loaded_model, model_uri
    except Exception:
        fallback_path = Path(MODEL_PATH)
        if fallback_path.exists():
            return load_joblib(fallback_path), str(fallback_path)
        raise FileNotFoundError(
            "Could not load a model from MLflow and no local fallback model was found at " f"{fallback_path.resolve()}"
        )
