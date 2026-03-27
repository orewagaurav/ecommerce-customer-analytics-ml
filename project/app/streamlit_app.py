"""Streamlit dashboard for the E-Commerce Customer Analytics Platform."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict import predict_customer


st.set_page_config(page_title="E-Commerce Customer Analytics", page_icon="📊", layout="wide")

DATA_PATH = PROJECT_ROOT / "data" / "processed_online_retail_II.csv"
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_customer_predictions() -> pd.DataFrame:
    pred_path = MODELS_DIR / "customer_predictions.csv"
    if pred_path.exists():
        return pd.read_csv(pred_path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame:
    segment_path = MODELS_DIR / "customer_segments.csv"
    if segment_path.exists():
        return pd.read_csv(segment_path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_training_report() -> Dict:
    report_path = MODELS_DIR / "training_report.json"
    if report_path.exists():
        return pd.read_json(report_path, typ="series").to_dict()
    return {}


def show_sidebar() -> str:
    st.sidebar.title("E-Commerce Analytics")
    pages = [
        "Overview Dashboard",
        "Customer Segmentation",
        "CLV Prediction",
        "Churn Prediction",
        "Recommendations",
    ]
    return st.sidebar.radio("Select Page", pages)


def show_not_trained_warning() -> None:
    st.warning(
        "Model artifacts are missing. Run: `python project/src/train_models.py --processed_csv project/data/processed_online_retail_II.csv --models_dir project/models`"
    )


def overview_page(df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    st.title("Overview Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"{df['TotalAmount'].sum():,.2f}")
    c2.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")
    c3.metric("Transactions", f"{len(df):,}")
    c4.metric("Avg Basket", f"{(df['TotalAmount'].sum()/df['Invoice'].nunique()):,.2f}")

    monthly = (
        df.assign(InvoiceMonth=df["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
        .groupby("InvoiceMonth", as_index=False)["TotalAmount"]
        .sum()
    )
    fig_rev = px.line(monthly, x="InvoiceMonth", y="TotalAmount", title="Monthly Revenue")
    st.plotly_chart(fig_rev, use_container_width=True)

    if not predictions.empty:
        churn_risk = (predictions["ChurnProbability"] > 0.7).mean() * 100
        st.info(f"Customers with high churn risk (>0.7): {churn_risk:.2f}%")


def segmentation_page(segments: pd.DataFrame) -> None:
    st.title("Customer Segmentation")

    if segments.empty:
        show_not_trained_warning()
        return

    counts = segments["ClusterLabel"].value_counts().reset_index()
    counts.columns = ["ClusterLabel", "Customers"]

    fig_pie = px.pie(counts, names="ClusterLabel", values="Customers", title="Cluster Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_scatter = px.scatter(
        segments,
        x="Recency",
        y="Monetary",
        color="ClusterLabel",
        hover_data=["CustomerID", "Frequency"],
        title="RFM Cluster View (Recency vs Monetary)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.dataframe(segments.head(30), use_container_width=True)


def clv_prediction_page(df: pd.DataFrame) -> None:
    st.title("CLV Prediction")

    if not (MODELS_DIR / "clv_model_artifacts.joblib").exists():
        show_not_trained_warning()
        return

    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=int(df["CustomerID"].min()),
        max_value=int(df["CustomerID"].max()),
        value=int(df["CustomerID"].iloc[0]),
        step=1,
    )

    if st.button("Predict CLV", key="predict_clv"):
        result = predict_customer(int(customer_id), DATA_PATH, MODELS_DIR)
        st.success(f"Predicted CLV (next window): {result['PredictedCLV']:.2f}")
        st.write("Cluster Label:", result["ClusterLabel"])

        importance_path = MODELS_DIR / "clv_feature_importance.csv"
        if importance_path.exists():
            imp = pd.read_csv(importance_path).head(10)
            fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top CLV Features")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.write("Why this CLV prediction:")
        for reason in result["Explanations"]["CLV"]:
            st.write(f"- {reason}")


def churn_prediction_page(df: pd.DataFrame) -> None:
    st.title("Churn Prediction")

    if not (MODELS_DIR / "churn_model_artifacts.joblib").exists():
        show_not_trained_warning()
        return

    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=int(df["CustomerID"].min()),
        max_value=int(df["CustomerID"].max()),
        value=int(df["CustomerID"].iloc[0]),
        step=1,
        key="churn_customer_id",
    )

    if st.button("Predict Churn", key="predict_churn"):
        result = predict_customer(int(customer_id), DATA_PATH, MODELS_DIR)
        st.warning(f"Churn Probability: {result['ChurnProbability']:.4f}")

        importance_path = MODELS_DIR / "churn_feature_importance.csv"
        if importance_path.exists():
            imp = pd.read_csv(importance_path).head(10)
            fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top Churn Features")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.write("Why this churn prediction:")
        for reason in result["Explanations"]["Churn"]:
            st.write(f"- {reason}")


def recommendations_page(df: pd.DataFrame) -> None:
    st.title("Recommendations")

    if not (MODELS_DIR / "churn_model_artifacts.joblib").exists():
        show_not_trained_warning()
        return

    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=int(df["CustomerID"].min()),
        max_value=int(df["CustomerID"].max()),
        value=int(df["CustomerID"].iloc[0]),
        step=1,
        key="recommend_customer_id",
    )

    if st.button("Generate Recommendation", key="recommend_action"):
        result = predict_customer(int(customer_id), DATA_PATH, MODELS_DIR)

        m1, m2, m3 = st.columns(3)
        m1.metric("Cluster Label", result["ClusterLabel"])
        m2.metric("Predicted CLV", f"{result['PredictedCLV']:.2f}")
        m3.metric("Churn Probability", f"{result['ChurnProbability']:.4f}")

        st.subheader("Recommended Actions")
        for action in result["RecommendationActions"]:
            st.write(f"- {action}")


def main() -> None:
    if not DATA_PATH.exists():
        st.error("Processed data not found. Run preprocessing first.")
        return

    page = show_sidebar()
    df = load_data()
    predictions = load_customer_predictions()
    segments = load_segments()

    if page == "Overview Dashboard":
        overview_page(df, predictions)
    elif page == "Customer Segmentation":
        segmentation_page(segments)
    elif page == "CLV Prediction":
        clv_prediction_page(df)
    elif page == "Churn Prediction":
        churn_prediction_page(df)
    elif page == "Recommendations":
        recommendations_page(df)


if __name__ == "__main__":
    main()
