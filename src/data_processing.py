import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(('.xlsx', '.xls')):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour.fillna(-1).astype(int)
    df['TransactionDay'] = df['TransactionStartTime'].dt.day.fillna(0).astype(int)
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month.fillna(0).astype(int)
    df['TransactionYear'] = df['TransactionStartTime'].dt.year.fillna(0).astype(int)
    return df


def most_common(series: pd.Series):
    mode = series.mode()
    if not mode.empty:
        return mode.iloc[0]
    return series.iloc[0]


def compute_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(df)
    category_columns = [
        'CurrencyCode',
        'CountryCode',
        'ProviderId',
        'ProductId',
        'ProductCategory',
        'ChannelId',
        'PricingStrategy',
        'FraudResult',
    ]

    aggregation = {
        'Amount': ['sum', 'mean', 'count', 'std', 'max'],
        'Value': ['sum'],
        'TransactionHour': ['mean', 'std'],
        'TransactionDay': ['mean', 'std'],
        'TransactionMonth': ['mean', 'std'],
        'TransactionYear': ['mean'],
    }

    category_agg = {col: most_common for col in category_columns}

    numeric_summary = (
        df.groupby('CustomerId')
        .agg(aggregation)
        .reset_index()
    )
    numeric_summary.columns = [
        'CustomerId',
        'TotalTransactionAmount',
        'AverageTransactionAmount',
        'TransactionCount',
        'StdDevTransactionAmount',
        'MaxTransactionAmount',
        'TotalValue',
        'AverageTransactionHour',
        'StdTransactionHour',
        'AverageTransactionDay',
        'StdTransactionDay',
        'AverageTransactionMonth',
        'StdTransactionMonth',
        'AverageTransactionYear',
    ]

    categorical_summary = (
        df.groupby('CustomerId')
        .agg(category_agg)
        .reset_index()
    )

    summary = numeric_summary.merge(categorical_summary, on='CustomerId', how='left')
    summary['StdDevTransactionAmount'] = summary['StdDevTransactionAmount'].fillna(0)
    summary['AverageTransactionHour'] = summary['AverageTransactionHour'].fillna(0).astype(int)
    summary['AverageTransactionDay'] = summary['AverageTransactionDay'].fillna(0).astype(int)
    summary['AverageTransactionMonth'] = summary['AverageTransactionMonth'].fillna(0).astype(int)
    summary['AverageTransactionYear'] = summary['AverageTransactionYear'].fillna(0).astype(int)
    return summary


def build_rfm(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    df = add_temporal_features(df)
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby('CustomerId')
        .agg(
            recency_days=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
            frequency=('TransactionId', 'count'),
            monetary=('Value', 'sum'),
        )
        .reset_index()
    )
    return rfm


def assign_high_risk_label(rfm: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[['recency_days', 'frequency', 'monetary']])
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(scaled)

    cluster_stats = rfm.groupby('cluster').agg(
        recency_mean=('recency_days', 'mean'),
        frequency_mean=('frequency', 'mean'),
        monetary_mean=('monetary', 'mean'),
    )

    cluster_stats['score'] = (
        cluster_stats['recency_mean'].rank(ascending=False)
        + cluster_stats['frequency_mean'].rank(ascending=True)
        + cluster_stats['monetary_mean'].rank(ascending=True)
    )

    high_risk_cluster = cluster_stats['score'].idxmax()
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    return rfm[['CustomerId', 'is_high_risk']]


def prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    customer_summary = compute_customer_summary(df)
    rfm = build_rfm(df)
    risk_target = assign_high_risk_label(rfm)
    merged = customer_summary.merge(risk_target, on='CustomerId', how='left')
    merged['is_high_risk'] = merged['is_high_risk'].fillna(0).astype(int)
    return merged


def build_preprocessing_pipeline() -> Pipeline:
    numeric_features = [
        'TotalTransactionAmount',
        'AverageTransactionAmount',
        'TransactionCount',
        'StdDevTransactionAmount',
        'MaxTransactionAmount',
        'TotalValue',
        'AverageTransactionHour',
        'AverageTransactionDay',
        'AverageTransactionMonth',
        'AverageTransactionYear',
        'recency_days',
        'frequency',
        'monetary',
    ]
    categorical_features = [
        'CurrencyCode',
        'CountryCode',
        'ProviderId',
        'ProductId',
        'ProductCategory',
        'ChannelId',
        'PricingStrategy',
        'FraudResult',
    ]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    transformer = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features),
    ])

    return Pipeline([('transformer', transformer)])
