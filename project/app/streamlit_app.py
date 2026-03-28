"""Streamlit dashboard for the E-Commerce Customer Analytics Platform."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Dict


def _ensure_runtime_dependencies() -> None:
    """Install missing runtime packages when Streamlit Cloud skips requirements resolution."""
    required_modules = {
        "joblib": "joblib",
        "plotly": "plotly",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "shap": "shap",
    }
    missing_packages = [
        package_name
        for module_name, package_name in required_modules.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing_packages:
        return

    cmd = [sys.executable, "-m", "pip", "install", *missing_packages]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to install runtime dependencies: "
            f"{' '.join(missing_packages)}\n{completed.stderr.strip()}"
        )


_ensure_runtime_dependencies()

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
        "Overview",
        "Segmentation",
        "CLV Prediction",
        "Churn Prediction",
        "Recommendations",
    ]
    return st.sidebar.radio("Select Page", pages)


def show_not_trained_warning() -> None:
    st.warning(
        "Model artifacts are missing. Run: `python project/src/train_models.py --processed_csv project/data/processed_online_retail_II.csv --models_dir project/models`"
    )


def _render_prediction_cards(result: Dict) -> None:
    """Render compact metric cards for key prediction outputs."""
    c1, c2, c3 = st.columns(3)
    c1.metric("CLV", f"{result['PredictedCLV']:.2f}")
    c2.metric("Churn Probability", f"{result['ChurnProbability']:.4f}")
    c3.metric("Customer Segment", result["ClusterLabel"])


def _render_shap_panel(result: Dict, prediction_key: str) -> None:
    """Show top SHAP features and human explanation for selected prediction type."""
    st.subheader("🔍 Why this prediction?")

    summary = result["Explanations"][prediction_key][0]
    st.info(summary)

    top_rows = result.get("ShapTopFeatures", {}).get(prediction_key, [])
    if top_rows:
        shap_df = pd.DataFrame(top_rows)
        chart = px.bar(
            shap_df,
            x="AbsContribution",
            y="Feature",
            color="Direction",
            orientation="h",
            title=f"Top {min(3, len(shap_df))} SHAP Contributors ({prediction_key})",
        )
        st.plotly_chart(chart, use_container_width=True)
    else:
        for line in result["Explanations"][prediction_key][1:4]:
            st.write(f"- {line}")


def _render_churn_gauge(churn_probability: float) -> None:
    """Render optional churn gauge visualization."""
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=churn_probability,
            number={"valueformat": ".2f"},
            title={"text": "Churn Risk"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#d62728"},
                "steps": [
                    {"range": [0.0, 0.4], "color": "#c7e9c0"},
                    {"range": [0.4, 0.7], "color": "#fdd49e"},
                    {"range": [0.7, 1.0], "color": "#fcae91"},
                ],
            },
        )
    )
    gauge.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(gauge, use_container_width=True)


def overview_page(df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    st.title("Overview")

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
    st.title("Segmentation")

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

    st.dataframe(segments.head(30), width="stretch")


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
        _render_prediction_cards(result)

        importance_path = MODELS_DIR / "clv_feature_importance.csv"
        if importance_path.exists():
            imp = pd.read_csv(importance_path).head(10)
            fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top CLV Features")
            st.plotly_chart(fig_imp, use_container_width=True)

        _render_shap_panel(result, prediction_key="CLV")


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
        _render_prediction_cards(result)

        c_left, c_right = st.columns([2, 1])
        with c_left:
            importance_path = MODELS_DIR / "churn_feature_importance.csv"
            if importance_path.exists():
                imp = pd.read_csv(importance_path).head(10)
                fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top Churn Features")
                st.plotly_chart(fig_imp, use_container_width=True)
        with c_right:
            _render_churn_gauge(result["ChurnProbability"])

        _render_shap_panel(result, prediction_key="Churn")


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

        _render_prediction_cards(result)

        decision = result.get("Decision", {})
        panel_left, panel_right = st.columns(2)
        with panel_left:
            st.subheader("Recommendation Panel")
            st.metric("Priority Level", decision.get("PriorityLevel", "Low"))
            st.write("Suggested Action:")
            st.success(decision.get("RecommendedAction", "Normal Engagement"))
        with panel_right:
            _render_churn_gauge(result["ChurnProbability"])

        st.subheader("Recommended Actions")
        for action in result["RecommendationActions"]:
            st.write(f"- {action}")

        _render_shap_panel(result, prediction_key="Churn")


def main() -> None:
    if not DATA_PATH.exists():
        st.error("Processed data not found. Run preprocessing first.")
        return

    page = show_sidebar()
    df = load_data()
    predictions = load_customer_predictions()
    segments = load_segments()

    if page == "Overview":
        overview_page(df, predictions)
    elif page == "Segmentation":
        segmentation_page(segments)
    elif page == "CLV Prediction":
        clv_prediction_page(df)
    elif page == "Churn Prediction":
        churn_prediction_page(df)
    elif page == "Recommendations":
        recommendations_page(df)


if __name__ == "__main__":
    main()
