from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_RAW_DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "data.csv",
    PROJECT_ROOT / "data" / "raw" / "data.xlsx",
    PROJECT_ROOT / "DATA" / "data.csv",
    PROJECT_ROOT / "DATA" / "data.xlsx",
]
DEFAULT_PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"

MLFLOW_EXPERIMENT_NAME = "CreditRiskModel"
MLFLOW_MODEL_NAME = "credit_risk_model"
MLFLOW_MODEL_ALIAS = "champion"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_list = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find an input dataset. Checked: {candidate_list}")
