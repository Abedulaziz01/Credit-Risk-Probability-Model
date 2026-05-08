from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DEFAULT_PROCESSED_DATA_PATH
from src.data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, get_feature_columns
from src.model_loader import load_registered_or_local_model
from src.predict import predict


st.set_page_config(page_title="Bati Bank Credit Risk Dashboard", page_icon=":bar_chart:", layout="wide")


def _load_reference_frame() -> pd.DataFrame:
    candidate_paths = [
        DEFAULT_PROCESSED_DATA_PATH,
        Path("data/processed/customer_features.csv"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.DataFrame(columns=["CustomerId", *get_feature_columns(), "is_high_risk"])


@st.cache_resource
def get_model():
    return load_registered_or_local_model()


@st.cache_data
def get_reference_frame() -> pd.DataFrame:
    return _load_reference_frame()


def _default_numeric_values(reference_df: pd.DataFrame) -> dict:
    defaults = {}
    for column in NUMERIC_FEATURES:
        if column in reference_df.columns and not reference_df[column].dropna().empty:
            defaults[column] = float(reference_df[column].median())
        else:
            defaults[column] = 0.0
    return defaults


def _default_categorical_values(reference_df: pd.DataFrame) -> dict:
    defaults = {}
    for column in CATEGORICAL_FEATURES:
        if column in reference_df.columns and not reference_df[column].dropna().empty:
            defaults[column] = reference_df[column].mode().iloc[0]
        else:
            defaults[column] = "unknown"
    return defaults


def render_sidebar(reference_df: pd.DataFrame) -> dict:
    st.sidebar.header("Customer Inputs")
    numeric_defaults = _default_numeric_values(reference_df)
    categorical_defaults = _default_categorical_values(reference_df)
    payload = {}

    for column in CATEGORICAL_FEATURES:
        raw_options = []
        if column in reference_df.columns:
            raw_options = reference_df[column].dropna().astype(str).unique().tolist()
        options = sorted(raw_options) if raw_options else [str(categorical_defaults[column])]
        selected = st.sidebar.selectbox(column, options=options, index=0)
        if column in {"CountryCode", "PricingStrategy", "FraudResult"}:
            payload[column] = int(float(selected))
        else:
            payload[column] = selected

    for column in NUMERIC_FEATURES:
        step = 1.0 if column not in {"TransactionCount", "frequency", "AverageTransactionYear"} else 1
        payload[column] = st.sidebar.number_input(column, value=float(numeric_defaults[column]), step=step)

    payload["TransactionCount"] = int(payload["TransactionCount"])
    payload["frequency"] = int(payload["frequency"])
    return payload


def render_prediction(prediction: dict) -> None:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Risk Probability", f"{prediction['risk_probability']:.2%}")
    metric_columns[1].metric("Risk Label", "High" if prediction["risk_label"] == 1 else "Low")
    metric_columns[2].metric("Credit Score", prediction["credit_score"])
    metric_columns[3].metric("Recommended Amount", f"{prediction['recommended_loan_amount']:.2f}")
    st.caption(f"Recommended loan duration: {prediction['recommended_loan_duration_days']} days")


def render_batch_prediction(model, reference_df: pd.DataFrame) -> None:
    st.subheader("Batch Prediction")
    st.write("Upload a CSV file with the same feature columns used by the model.")
    uploaded_file = st.file_uploader("Upload prediction CSV", type=["csv"])
    if uploaded_file is None:
        if not reference_df.empty:
            st.caption(
                "Tip: you can export rows from data/processed/customer_features.csv "
                "and remove CustomerId/is_high_risk."
            )
        return

    batch_df = pd.read_csv(uploaded_file)
    required_columns = get_feature_columns()
    missing_columns = [column for column in required_columns if column not in batch_df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return

    predictions = [predict(model, row) for row in batch_df[required_columns].to_dict(orient="records")]
    output_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    st.dataframe(output_df, use_container_width=True)
    st.download_button(
        "Download Predictions",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="credit_risk_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Bati Bank Credit Risk Dashboard")
    st.write("Interactive scoring dashboard for proxy-based customer credit risk assessment.")

    try:
        model, model_source = get_model()
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        st.stop()

    reference_df = get_reference_frame()

    top_left, top_right = st.columns([2, 1])
    top_left.info(f"Model source: {model_source}")
    top_right.metric("Reference Customers", len(reference_df))

    payload = render_sidebar(reference_df)

    st.subheader("Single Customer Prediction")
    left, right = st.columns([1, 1])

    with left:
        st.write("Current input payload")
        st.json(payload)

    with right:
        if st.button("Score Customer", type="primary", use_container_width=True):
            prediction = predict(model, payload)
            render_prediction(prediction)

    if not reference_df.empty:
        st.subheader("Reference Data Preview")
        preview_columns = ["CustomerId", *get_feature_columns(), "is_high_risk"]
        available_columns = [column for column in preview_columns if column in reference_df.columns]
        st.dataframe(reference_df[available_columns].head(10), use_container_width=True)

    render_batch_prediction(model, reference_df)


if __name__ == "__main__":
    main()
