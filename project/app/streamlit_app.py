"""Streamlit dashboard for the E-Commerce Customer Analytics Platform."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics_store import AnalyticsAggregates
from src.config import get_settings
from src.feature_store import FeatureStore
from app.api_client import AnalyticsApiClient, ApiError, CustomerNotFound
from app.pages_extra import (
    customer_360_page,
    executive_page,
    history_page,
    what_if_page,
)

SETTINGS = get_settings()

_HTTP_BACKEND = AnalyticsApiClient(
    base_url=SETTINGS.api_base_url,
    api_version=SETTINGS.api_version,
    timeout=SETTINGS.request_timeout_seconds,
)


@st.cache_resource(show_spinner=False)
def _local_backend():
    """In-process backend, used when no API is reachable."""
    from app.local_backend import LocalBackend

    return LocalBackend(SETTINGS)


@st.cache_resource(show_spinner="Connecting to scoring backend...")
def _resolve_backend():
    """Prefer the API; fall back to in-process scoring.

    Under docker compose or locally the FastAPI service answers and is used.
    On Streamlit Community Cloud there is no second process to host it, so the
    dashboard scores in-process instead. Both routes call the same
    PredictionService, so only the transport differs.
    """
    if _HTTP_BACKEND.is_available():
        return _HTTP_BACKEND, "api"
    return _local_backend(), "embedded"


API, BACKEND_MODE = _resolve_backend()


def predict_customer(customer_id: int, *_ignored) -> Dict:
    """Score a customer through the API and adapt to the dashboard's shape."""
    payload = API.predict(int(customer_id))
    return {
        "CustomerID": payload["customer_id"],
        "ClusterLabel": payload["cluster_label"],
        "PredictedCLV": payload["predicted_clv"],
        "ChurnProbability": payload["churn_probability"],
        "Decision": {
            "CustomerSegment": payload["decision"]["customer_segment"],
            "PriorityLevel": payload["decision"]["priority_level"],
            "RecommendedAction": payload["decision"]["recommended_action"],
        },
        "RecommendationActions": payload["recommendation_actions"],
        "Explanations": payload["explanations"],
        "ShapTopFeatures": {
            task: [
                {
                    "Feature": item["feature"],
                    "Contribution": item["contribution"],
                    "AbsContribution": item["abs_contribution"],
                    "Direction": item["direction"],
                }
                for item in rows
            ]
            for task, rows in payload["shap_top_features"].items()
        },
        "ModelVersion": payload["model_version"],
        "LatencyMs": payload["latency_ms"],
    }


st.set_page_config(page_title="E-Commerce Customer Analytics", page_icon="📊", layout="wide")

DATA_PATH = SETTINGS.data_path
MODELS_DIR = SETTINGS.model_path

AGGREGATES = AnalyticsAggregates(SETTINGS.analytics_store_path)
FEATURES = FeatureStore(SETTINGS.feature_store_path)


@st.cache_data(show_spinner=False)
def load_monthly_revenue() -> pd.DataFrame:
    """Precomputed monthly totals; the dashboard never reads raw transactions."""
    return AGGREGATES.monthly_revenue()


@st.cache_data(show_spinner=False)
def load_kpis() -> Dict:
    return AGGREGATES.kpis()


@st.cache_data(show_spinner=False)
def load_customer_ids() -> list[int]:
    return sorted(FEATURES.customer_ids())


@st.cache_data(show_spinner=False)
def load_feature_frame() -> pd.DataFrame:
    """Full feature table, for the executive top-customers view."""
    return FEATURES.frame()


def safe_predict(customer_id: int) -> Dict | None:
    """Call the API, surfacing failures as dashboard messages rather than traces."""
    try:
        return predict_customer(int(customer_id))
    except CustomerNotFound:
        st.warning(f"Customer {customer_id} is not in the feature store.")
    except ApiError as exc:
        st.error(f"Scoring service unavailable: {exc}")
    return None


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
        "Executive Dashboard",
        "Overview",
        "Customer 360",
        "Segmentation",
        "CLV Prediction",
        "Churn Prediction",
        "Recommendations",
        "What-if Simulator",
        "Prediction History",
    ]
    selection = st.sidebar.radio("Select Page", pages)

    # Which backend is answering matters: every scoring page depends on it.
    st.sidebar.divider()
    try:
        info = API.model_info()
        if BACKEND_MODE == "api":
            st.sidebar.success("Scoring via API")
            st.sidebar.caption(f"{SETTINGS.api_base_url} · model {info['production_version']}")
        else:
            st.sidebar.info("Scoring in-process")
            st.sidebar.caption(f"Embedded backend · model {info['production_version']}")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error("Scoring backend unavailable")
        st.sidebar.caption(str(exc))

    return selection


def show_not_trained_warning() -> None:
    st.warning(
        "Model artifacts are missing. Run: `python project/src/train_models.py --processed_csv project/data/processed_online_retail_II.csv --models_dir project/models`"
    )


def _render_prediction_cards(result: Dict) -> None:
    """Render compact metric cards for key prediction outputs."""
    c1, c2, c3 = st.columns(3)
    c1.metric("CLV", f"{result['PredictedCLV']:.2f}")
    c2.metric("Churn Probability", f"{result['ChurnProbability'] * 100:.2f}%")
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
    churn_probability_pct = churn_probability * 100.0
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=churn_probability_pct,
            number={"suffix": "%", "valueformat": ".2f"},
            title={"text": "Churn Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d62728"},
                "steps": [
                    {"range": [0.0, 40.0], "color": "#c7e9c0"},
                    {"range": [40.0, 70.0], "color": "#fdd49e"},
                    {"range": [70.0, 100.0], "color": "#fcae91"},
                ],
            },
        )
    )
    gauge.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(gauge, use_container_width=True)


def overview_page(predictions: pd.DataFrame) -> None:
    st.title("Overview")

    kpis = load_kpis()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"{kpis['total_revenue']:,.2f}")
    c2.metric("Unique Customers", f"{int(kpis['unique_customers']):,}")
    c3.metric("Transactions", f"{int(kpis['total_transactions']):,}")
    c4.metric("Avg Order Value", f"{kpis['avg_order_value']:,.2f}")

    monthly = load_monthly_revenue()
    fig_rev = px.line(monthly, x="InvoiceMonth", y="Revenue", title="Monthly Revenue")
    st.plotly_chart(fig_rev, use_container_width=True)

    if not predictions.empty:
        churn_risk = (predictions["ChurnProbability"] > SETTINGS.high_churn_threshold).mean() * 100
        st.info(
            f"Customers with high churn risk "
            f"(>{SETTINGS.high_churn_threshold:.0%}): {churn_risk:.2f}%"
        )


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


def clv_prediction_page() -> None:
    st.title("CLV Prediction")

    if not (MODELS_DIR / "clv_model_artifacts.joblib").exists():
        show_not_trained_warning()
        return

    customer_id = st.selectbox("Select Customer ID", load_customer_ids())

    if st.button("Predict CLV", key="predict_clv"):
        result = safe_predict(int(customer_id))
        if result is None:
            return
        _render_prediction_cards(result)

        importance_path = MODELS_DIR / "clv_feature_importance.csv"
        if importance_path.exists():
            imp = pd.read_csv(importance_path).head(10)
            fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top CLV Features")
            st.plotly_chart(fig_imp, use_container_width=True)

        _render_shap_panel(result, prediction_key="CLV")


def churn_prediction_page() -> None:
    st.title("Churn Prediction")

    if not (MODELS_DIR / "churn_model_artifacts.joblib").exists():
        show_not_trained_warning()
        return

    customer_id = st.selectbox(
        "Select Customer ID", load_customer_ids(), key="churn_customer_id"
    )

    if st.button("Predict Churn", key="predict_churn"):
        result = safe_predict(int(customer_id))
        if result is None:
            return
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


def recommendations_page() -> None:
    st.title("Recommendations")

    if not (MODELS_DIR / "churn_model_artifacts.joblib").exists():
        show_not_trained_warning()
        return

    customer_id = st.selectbox(
        "Select Customer ID", load_customer_ids(), key="recommend_customer_id"
    )

    if st.button("Generate Recommendation", key="recommend_action"):
        result = safe_predict(int(customer_id))
        if result is None:
            return

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
    if not AGGREGATES.exists():
        st.error(
            "Dashboard aggregates not found. Run: "
            "`python project/src/build_feature_store.py`"
        )
        return

    page = show_sidebar()
    predictions = load_customer_predictions()
    segments = load_segments()

    if page == "Executive Dashboard":
        executive_page(AGGREGATES, predictions, load_feature_frame(), API)
    elif page == "Customer 360":
        customer_360_page(AGGREGATES, load_customer_ids, safe_predict, API)
    elif page == "What-if Simulator":
        what_if_page(load_customer_ids, API)
    elif page == "Prediction History":
        history_page(API)
    elif page == "Overview":
        overview_page(predictions)
    elif page == "Segmentation":
        segmentation_page(segments)
    elif page == "CLV Prediction":
        clv_prediction_page()
    elif page == "Churn Prediction":
        churn_prediction_page()
    elif page == "Recommendations":
        recommendations_page()


if __name__ == "__main__":
    main()
