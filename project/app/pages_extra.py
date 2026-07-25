"""Customer 360, Executive, What-if and Prediction History pages.

Kept in their own module so `streamlit_app.py` stays a router plus the original
five pages. Every page here reads precomputed aggregates or calls the API — none
of them import model code or touch raw transactions.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import plotly.express as px
import streamlit as st

# One qualitative palette across every page, so a segment keeps its colour
# whichever chart it appears in.
SEGMENT_COLOURS = {
    "Champions": "#2E7D32",
    "Loyal Customers": "#1565C0",
    "Regular Customers": "#6A6A6A",
    "At Risk": "#EF6C00",
    "Lost Customers": "#C62828",
}
ACCENT = "#1565C0"


def _currency(value: float) -> str:
    return f"{value:,.0f}"


# ---------------------------------------------------------------- Customer 360

def customer_360_page(
    aggregates: Any,
    load_customer_ids: Callable[[], list[int]],
    safe_predict: Callable[[int], dict | None],
    api: Any,
) -> None:
    """Everything known about one customer, on a single screen."""
    st.title("Customer 360")
    st.caption("Profile, behaviour and model output for a single customer.")

    customer_id = st.selectbox("Customer", load_customer_ids(), key="c360_customer")
    if not st.button("Load customer", key="c360_load"):
        st.info("Pick a customer and select **Load customer**.")
        return

    result = safe_predict(int(customer_id))
    if result is None:
        return

    try:
        profile = api.customer_profile(int(customer_id))["features"]
    except Exception:  # noqa: BLE001 - profile is supplementary, not essential
        profile = {}

    # --- headline model output ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment", result["ClusterLabel"])
    c2.metric("Predicted CLV", _currency(result["PredictedCLV"]))
    c3.metric("Churn risk", f"{result['ChurnProbability'] * 100:.1f}%")
    c4.metric("Priority", result["Decision"]["PriorityLevel"])

    st.success(f"**Recommended action:** {result['Decision']['RecommendedAction']}")

    # --- behavioural profile ---
    st.subheader("Profile")
    if profile:
        fields = [
            ("Country", profile.get("Country"), None),
            ("Total spend", profile.get("Monetary"), "money"),
            ("Orders", profile.get("Frequency"), "int"),
            ("Days since last order", profile.get("Recency"), "int"),
            ("Tenure (days)", profile.get("Tenure"), "int"),
            ("Avg basket", profile.get("AverageBasketSize"), "money"),
            ("Avg days between orders", profile.get("AvgInterPurchaseDays"), "int"),
            ("Distinct products", profile.get("DistinctProducts"), "int"),
        ]
        columns = st.columns(4)
        for index, (label, value, kind) in enumerate(fields):
            if value is None:
                continue
            if kind == "money":
                shown = _currency(float(value))
            elif kind == "int":
                shown = f"{float(value):,.0f}"
            else:
                shown = str(value)
            columns[index % 4].metric(label, shown)

        share = profile.get("RecentRevenueShare")
        if share is not None:
            st.progress(
                min(1.0, max(0.0, float(share))),
                text=f"{float(share):.0%} of lifetime spend came in the last 90 days of history",
            )
    else:
        st.caption("Profile features unavailable.")

    # --- spending trend ---
    st.subheader("Spending over time")
    monthly = aggregates.customer_monthly(int(customer_id))
    if monthly.empty:
        st.caption("No monthly history for this customer.")
    else:
        left, right = st.columns([3, 2])
        with left:
            fig = px.bar(
                monthly, x="InvoiceMonth", y="Revenue",
                title="Monthly spend", color_discrete_sequence=[ACCENT],
            )
            fig.update_layout(margin=dict(t=48, b=8, l=8, r=8))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig_orders = px.line(
                monthly, x="InvoiceMonth", y="Orders",
                title="Orders per month", markers=True,
                color_discrete_sequence=[SEGMENT_COLOURS["Champions"]],
            )
            fig_orders.update_layout(margin=dict(t=48, b=8, l=8, r=8))
            st.plotly_chart(fig_orders, use_container_width=True)

    # --- product mix ---
    st.subheader("Top products")
    products = aggregates.customer_products(int(customer_id), limit=10)
    if products.empty:
        st.caption("No product history for this customer.")
    else:
        fig_products = px.bar(
            products.sort_values("Revenue"), x="Revenue", y="Description",
            orientation="h", title="Top products by revenue",
            color_discrete_sequence=[ACCENT],
        )
        fig_products.update_layout(height=420, margin=dict(t=48, b=8, l=8, r=8))
        st.plotly_chart(fig_products, use_container_width=True)
        st.dataframe(
            products[["Description", "Units", "Revenue"]], use_container_width=True, hide_index=True
        )

    # --- explanation ---
    st.subheader("Why this prediction?")
    tab_churn, tab_clv = st.tabs(["Churn", "CLV"])
    for tab, key in ((tab_churn, "Churn"), (tab_clv, "CLV")):
        with tab:
            reasons = result["Explanations"].get(key, [])
            if reasons:
                st.info(reasons[0])
            rows = result["ShapTopFeatures"].get(key, [])
            if rows:
                frame = pd.DataFrame(rows)
                fig = px.bar(
                    frame, x="AbsContribution", y="Feature", color="Direction",
                    orientation="h", title=f"Top SHAP contributors ({key})",
                    color_discrete_map={"increase": "#C62828", "decrease": "#2E7D32"},
                )
                fig.update_layout(margin=dict(t=48, b=8, l=8, r=8))
                st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Model {result.get('ModelVersion', 'n/a')} · scored in "
        f"{result.get('LatencyMs', 0):.0f} ms"
    )


# ------------------------------------------------------------------- Executive

def executive_page(aggregates: Any, predictions: pd.DataFrame, features: pd.DataFrame) -> None:
    """Business-level view: where revenue is, and how much of it is at risk."""
    st.title("Executive Dashboard")
    st.caption("Revenue, customers and modelled risk across the book.")

    kpis = aggregates.kpis()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total revenue", _currency(kpis["total_revenue"]))
    c2.metric("Customers", f"{int(kpis['unique_customers']):,}")
    c3.metric("Orders", f"{int(kpis['total_orders']):,}")
    c4.metric("Avg order value", _currency(kpis["avg_order_value"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Countries", f"{int(kpis['countries']):,}")
    c6.metric("Transactions", f"{int(kpis['total_transactions']):,}")

    revenue_at_risk = 0.0
    high_risk_count = 0
    if not predictions.empty and "ChurnProbability" in predictions:
        at_risk = predictions[predictions["ChurnProbability"] > 0.7]
        high_risk_count = len(at_risk)
        # Monetary is historical spend, so this is revenue associated with
        # at-risk customers, not a forecast of revenue that will be lost.
        if "Monetary" in at_risk:
            revenue_at_risk = float(at_risk["Monetary"].sum())
        c7.metric("High-risk customers", f"{high_risk_count:,}")
        c8.metric("Revenue at risk", _currency(revenue_at_risk))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Revenue trend")
        monthly = aggregates.monthly_revenue()
        fig = px.area(
            monthly, x="InvoiceMonth", y="Revenue",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(margin=dict(t=16, b=8, l=8, r=8))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Revenue by country")
        country = aggregates.country_revenue().head(10)
        fig_country = px.bar(
            country.sort_values("Revenue"), x="Revenue", y="Country",
            orientation="h", color_discrete_sequence=[ACCENT],
        )
        fig_country.update_layout(margin=dict(t=16, b=8, l=8, r=8))
        st.plotly_chart(fig_country, use_container_width=True)

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Segment mix")
        if not predictions.empty and "ClusterLabel" in predictions:
            counts = predictions["ClusterLabel"].value_counts().reset_index()
            counts.columns = ["Segment", "Customers"]
            fig_seg = px.pie(
                counts, names="Segment", values="Customers", hole=0.45,
                color="Segment", color_discrete_map=SEGMENT_COLOURS,
            )
            fig_seg.update_layout(margin=dict(t=16, b=8, l=8, r=8))
            st.plotly_chart(fig_seg, use_container_width=True)
        else:
            st.caption("Scored customer data unavailable.")

    with right2:
        st.subheader("Churn risk distribution")
        if not predictions.empty and "ChurnProbability" in predictions:
            fig_churn = px.histogram(
                predictions, x="ChurnProbability", nbins=30,
                color_discrete_sequence=[SEGMENT_COLOURS["At Risk"]],
            )
            fig_churn.add_vline(
                x=0.7, line_dash="dash", line_color="#C62828",
                annotation_text="action threshold", annotation_position="top",
            )
            fig_churn.update_layout(margin=dict(t=16, b=8, l=8, r=8))
            st.plotly_chart(fig_churn, use_container_width=True)
        else:
            st.caption("Scored customer data unavailable.")

    st.subheader("Top customers by lifetime spend")
    if not features.empty:
        columns = [c for c in ["CustomerID", "Country", "Monetary", "Frequency", "Recency"]
                   if c in features.columns]
        top = features[columns].sort_values("Monetary", ascending=False).head(15)
        st.dataframe(top, use_container_width=True, hide_index=True)

    st.subheader("Recommended focus")
    if not predictions.empty and "RecommendedActions" in predictions:
        actions = (
            predictions["RecommendedActions"].astype(str)
            .str.split("; ").explode().value_counts().reset_index()
        )
        actions.columns = ["Action", "Customers"]
        fig_actions = px.bar(
            actions.head(8).sort_values("Customers"), x="Customers", y="Action",
            orientation="h", color_discrete_sequence=[ACCENT],
        )
        fig_actions.update_layout(margin=dict(t=16, b=8, l=8, r=8))
        st.plotly_chart(fig_actions, use_container_width=True)

    if high_risk_count:
        st.warning(
            f"**{high_risk_count:,} customers** exceed the 70% churn threshold, "
            f"representing **{_currency(revenue_at_risk)}** of historical revenue. "
            "Revenue at risk is spend already booked from these customers, not a "
            "forecast of what will be lost."
        )


# ---------------------------------------------------------------- What-if

WHAT_IF_FEATURES = [
    ("Recency", "Days since last purchase", 0.0, 800.0, 1.0),
    ("Frequency", "Number of orders", 1.0, 300.0, 1.0),
    ("Monetary", "Total spend", 0.0, 300_000.0, 100.0),
]


def what_if_page(load_customer_ids: Callable[[], list[int]], api: Any) -> None:
    """Move the levers, see what the models would say."""
    st.title("What-if Simulator")
    st.caption(
        "Adjust a customer's behaviour and rescore. Nothing is written back — "
        "this is a hypothetical, not a recorded prediction."
    )

    customer_id = st.selectbox("Customer", load_customer_ids(), key="whatif_customer")

    try:
        profile = api.customer_profile(int(customer_id))["features"]
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load customer features: {exc}")
        return

    st.subheader("Adjust behaviour")
    overrides: dict[str, float] = {}
    columns = st.columns(3)

    for index, (name, label, low, high, step) in enumerate(WHAT_IF_FEATURES):
        current = float(profile.get(name, low))
        # Widen the bound if the real customer sits outside the nominal range.
        upper = max(high, current * 1.5)
        # Snap to the step grid; Streamlit warns when a default is not
        # reachable from (min, max, step).
        clamped = min(max(current, low), upper)
        default = low + round((clamped - low) / step) * step
        with columns[index]:
            overrides[name] = st.slider(
                label, min_value=low, max_value=upper,
                value=float(min(default, upper)), step=step,
                key=f"whatif_{name}",
            )
            st.caption(f"Current: {current:,.0f}")

    if not st.button("Run simulation", key="whatif_run", type="primary"):
        return

    try:
        outcome = api.simulate(int(customer_id), overrides)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Simulation failed: {exc}")
        return

    baseline, simulated = outcome["baseline"], outcome["simulated"]

    st.subheader("Baseline vs simulated")
    left, right = st.columns(2)

    with left:
        st.markdown("#### Current")
        st.metric("Segment", baseline["cluster_label"])
        st.metric("Predicted CLV", _currency(baseline["predicted_clv"]))
        st.metric("Churn risk", f"{baseline['churn_probability'] * 100:.1f}%")
        st.caption(f"Action: {baseline['recommended_action']}")

    with right:
        st.markdown("#### Simulated")
        st.metric(
            "Segment", simulated["cluster_label"],
            delta="changed" if outcome["segment_changed"] else None,
        )
        st.metric(
            "Predicted CLV", _currency(simulated["predicted_clv"]),
            delta=_currency(outcome["clv_delta"]),
        )
        st.metric(
            "Churn risk", f"{simulated['churn_probability'] * 100:.1f}%",
            delta=f"{outcome['churn_delta'] * 100:+.1f} pp",
            delta_color="inverse",
        )
        st.caption(f"Action: {simulated['recommended_action']}")

    comparison = pd.DataFrame(
        {
            "Scenario": ["Current", "Simulated"],
            "Predicted CLV": [baseline["predicted_clv"], simulated["predicted_clv"]],
            "Churn %": [
                baseline["churn_probability"] * 100,
                simulated["churn_probability"] * 100,
            ],
        }
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig = px.bar(
            comparison, x="Scenario", y="Predicted CLV", color="Scenario",
            color_discrete_sequence=[ACCENT, SEGMENT_COLOURS["Champions"]],
            title="Predicted CLV",
        )
        fig.update_layout(showlegend=False, margin=dict(t=48, b=8, l=8, r=8))
        st.plotly_chart(fig, use_container_width=True)
    with chart_right:
        fig2 = px.bar(
            comparison, x="Scenario", y="Churn %", color="Scenario",
            color_discrete_sequence=[ACCENT, SEGMENT_COLOURS["At Risk"]],
            title="Churn probability",
        )
        fig2.update_layout(showlegend=False, margin=dict(t=48, b=8, l=8, r=8))
        st.plotly_chart(fig2, use_container_width=True)

    if outcome["segment_changed"]:
        st.success(
            f"Segment moves from **{baseline['cluster_label']}** to "
            f"**{simulated['cluster_label']}** under this scenario."
        )

    st.caption(
        "These are model responses to hypothetical inputs, not causal estimates. "
        "They show what the model would predict, not what would happen if you "
        "intervened."
    )


# ------------------------------------------------------------ Prediction history

def history_page(api: Any) -> None:
    """The audit trail: what the service was asked, and what it answered."""
    st.title("Prediction History")
    st.caption("Every scored request, most recent first.")

    filter_left, filter_mid, filter_right = st.columns(3)
    with filter_left:
        limit = st.number_input("Rows", min_value=10, max_value=5000, value=200, step=10)
    with filter_mid:
        customer_filter = st.text_input("Customer ID (optional)", value="")
    with filter_right:
        min_churn = st.slider("Min churn probability", 0.0, 1.0, 0.0, 0.05)

    customer_id: int | None = None
    if customer_filter.strip():
        try:
            customer_id = int(customer_filter.strip())
        except ValueError:
            st.warning("Customer ID must be a whole number; ignoring that filter.")

    try:
        payload = api.history(
            limit=int(limit),
            customer_id=customer_id,
            min_churn_probability=min_churn if min_churn > 0 else None,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load history: {exc}")
        return

    entries = payload.get("entries", [])
    if not entries:
        st.info("No predictions recorded yet. Score a customer to populate this log.")
        return

    frame = pd.DataFrame(entries)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total recorded", f"{payload['total']:,}")
    c2.metric("Matching filter", f"{payload['returned']:,}")
    c3.metric("Avg latency", f"{frame['latency_ms'].mean():.0f} ms")

    fig = px.scatter(
        frame, x="timestamp", y="churn_probability",
        color="cluster_label", size="predicted_clv",
        color_discrete_map=SEGMENT_COLOURS,
        title="Scored predictions over time",
    )
    fig.update_layout(margin=dict(t=48, b=8, l=8, r=8))
    st.plotly_chart(fig, use_container_width=True)

    display = frame.copy()
    display["churn_probability"] = (display["churn_probability"] * 100).round(1)
    display["predicted_clv"] = display["predicted_clv"].round(0)
    display = display.rename(
        columns={
            "timestamp": "Time", "customer_id": "Customer", "predicted_clv": "CLV",
            "churn_probability": "Churn %", "cluster_label": "Segment",
            "recommended_action": "Action", "model_version": "Model",
            "latency_ms": "Latency (ms)",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.download_button(
        "Download as CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )


__all__ = ["customer_360_page", "executive_page", "what_if_page", "history_page"]
