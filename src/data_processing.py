import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_PROCESSED_DATA_PATH, DEFAULT_RAW_DATA_CANDIDATES, resolve_existing_path


NUMERIC_FEATURES = [
    "TotalTransactionAmount",
    "AverageTransactionAmount",
    "TransactionCount",
    "StdDevTransactionAmount",
    "MaxTransactionAmount",
    "TotalValue",
    "AverageTransactionHour",
    "StdTransactionHour",
    "AverageTransactionDay",
    "StdTransactionDay",
    "AverageTransactionMonth",
    "StdTransactionMonth",
    "AverageTransactionYear",
    "recency_days",
    "frequency",
    "monetary",
]

CATEGORICAL_FEATURES = [
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "PricingStrategy",
    "FraudResult",
]

TARGET_COLUMN = "is_high_risk"
ID_COLUMN = "CustomerId"


def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if path.lower().endswith((".xlsx", ".xls")):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce", utc=True)
    df["TransactionStartTime"] = df["TransactionStartTime"].dt.tz_localize(None)
    df["TransactionHour"] = df["TransactionStartTime"].dt.hour.fillna(-1).astype(int)
    df["TransactionDay"] = df["TransactionStartTime"].dt.day.fillna(0).astype(int)
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month.fillna(0).astype(int)
    df["TransactionYear"] = df["TransactionStartTime"].dt.year.fillna(0).astype(int)
    return df


def most_common(series: pd.Series):
    mode = series.mode()
    if not mode.empty:
        return mode.iloc[0]
    return series.iloc[0]


def compute_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(df)
    category_columns = CATEGORICAL_FEATURES

    aggregation = {
        "Amount": ["sum", "mean", "count", "std", "max"],
        "Value": ["sum"],
        "TransactionHour": ["mean", "std"],
        "TransactionDay": ["mean", "std"],
        "TransactionMonth": ["mean", "std"],
        "TransactionYear": ["mean"],
    }

    category_agg = {col: most_common for col in category_columns}

    numeric_summary = df.groupby(ID_COLUMN).agg(aggregation).reset_index()
    numeric_summary.columns = [
        ID_COLUMN,
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdDevTransactionAmount",
        "MaxTransactionAmount",
        "TotalValue",
        "AverageTransactionHour",
        "StdTransactionHour",
        "AverageTransactionDay",
        "StdTransactionDay",
        "AverageTransactionMonth",
        "StdTransactionMonth",
        "AverageTransactionYear",
    ]

    categorical_summary = df.groupby(ID_COLUMN).agg(category_agg).reset_index()

    summary = numeric_summary.merge(categorical_summary, on=ID_COLUMN, how="left")

    numeric_fill_values = {
        "StdDevTransactionAmount": 0.0,
        "StdTransactionHour": 0.0,
        "StdTransactionDay": 0.0,
        "StdTransactionMonth": 0.0,
    }
    for column, fill_value in numeric_fill_values.items():
        summary[column] = summary[column].fillna(fill_value)

    integer_like_columns = [
        "AverageTransactionHour",
        "AverageTransactionDay",
        "AverageTransactionMonth",
        "AverageTransactionYear",
    ]
    for column in integer_like_columns:
        summary[column] = summary[column].fillna(0).round().astype(int)

    return summary


def build_rfm(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    df = add_temporal_features(df)
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby(ID_COLUMN).agg(
        recency_days=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Value", "sum"),
    )
    rfm = rfm.reset_index()
    return rfm


def assign_high_risk_label(rfm: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rfm = rfm.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["recency_days", "frequency", "monetary"]])
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(scaled)

    cluster_stats = rfm.groupby("cluster").agg(
        recency_mean=("recency_days", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
    )

    normalized = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min()).replace(0, 1)
    cluster_stats["risk_score"] = (
        normalized["recency_mean"] + (1 - normalized["frequency_mean"]) + (1 - normalized["monetary_mean"])
    )

    high_risk_cluster = cluster_stats["risk_score"].idxmax()
    rfm[TARGET_COLUMN] = (rfm["cluster"] == high_risk_cluster).astype(int)
    return rfm[[ID_COLUMN, TARGET_COLUMN]]


def prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    customer_summary = compute_customer_summary(df)
    rfm = build_rfm(df)
    risk_target = assign_high_risk_label(rfm)
    merged = customer_summary.merge(rfm, on=ID_COLUMN, how="left")
    merged = merged.merge(risk_target, on=ID_COLUMN, how="left")
    merged[TARGET_COLUMN] = merged[TARGET_COLUMN].fillna(0).astype(int)
    return merged


def build_preprocessing_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformer = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline([("transformer", transformer)])


def get_feature_columns() -> list[str]:
    return [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def build_processed_dataset(input_path: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    resolved_input = str(resolve_existing_path(DEFAULT_RAW_DATA_CANDIDATES)) if input_path is None else input_path
    resolved_output = str(DEFAULT_PROCESSED_DATA_PATH) if output_path is None else output_path
    processed = prepare_model_dataframe(load_data(resolved_input))
    save_data(processed, resolved_output)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build customer-level features for credit risk modeling.")
    parser.add_argument("--input", type=str, default=None, help="Path to the raw transaction dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_PROCESSED_DATA_PATH),
        help="Path to save the processed customer-level dataset.",
    )
    args = parser.parse_args()

    processed = build_processed_dataset(args.input, args.output)
    print(f"Processed dataset saved to {args.output}")
    print(f"Rows: {len(processed)}, Columns: {len(processed.columns)}")


if __name__ == "__main__":
    main()
