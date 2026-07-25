"""Report generation: per-customer PDF and multi-sheet Excel workbooks.

Reports are produced server-side and streamed to the caller, so the dashboard
stays a thin client and any consumer (a scheduled job, a CRM integration) gets
the same artefact through the same endpoint.

Everything is written to an in-memory buffer; nothing touches disk, which keeps
the container filesystem read-only-friendly and avoids temp-file cleanup.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.logging_config import get_logger

logger = get_logger(__name__)

BRAND = colors.HexColor("#1565C0")
MUTED = colors.HexColor("#5F6368")
RISK = colors.HexColor("#C62828")
GOOD = colors.HexColor("#2E7D32")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=20, textColor=BRAND, spaceAfter=2 * mm
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=9, textColor=MUTED, spaceAfter=6 * mm
        ),
        "heading": ParagraphStyle(
            "heading", parent=base["Heading2"], fontSize=12, textColor=BRAND,
            spaceBefore=5 * mm, spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=9.5, leading=14, alignment=TA_LEFT
        ),
        "note": ParagraphStyle(
            "note", parent=base["Normal"], fontSize=8, textColor=MUTED, leading=11
        ),
    }


def _table(rows: list[list[Any]], widths: list[float], header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#E0E0E0")),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), BRAND),
        ]
    table.setStyle(TableStyle(style))
    return table


class ReportService:
    """Builds downloadable artefacts from scored customer data."""

    def __init__(self, prediction_service: Any, feature_store: Any, aggregates: Any) -> None:
        self._predictions = prediction_service
        self._features = feature_store
        self._aggregates = aggregates

    # ------------------------------------------------------------------ PDF

    def customer_pdf(self, customer_id: int) -> bytes:
        """A one-page briefing for a single customer.

        Raises:
            CustomerNotFoundError: propagated from the feature store.
        """
        result = self._predictions.predict(customer_id, include_explanations=True)
        profile = self._predictions.customer_profile(customer_id)

        buffer = io.BytesIO()
        document = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=16 * mm, bottomMargin=16 * mm,
            title=f"Customer {customer_id} report",
            author="E-Commerce Customer Analytics",
        )
        style = _styles()
        story: list[Any] = []

        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        story.append(Paragraph(f"Customer {customer_id}", style["title"]))
        story.append(
            Paragraph(
                f"Generated {generated} &nbsp;·&nbsp; model {result.model_version}",
                style["subtitle"],
            )
        )

        # --- headline ---
        churn_pct = result.churn_probability * 100
        story.append(Paragraph("Model output", style["heading"]))
        story.append(
            _table(
                [
                    ["Segment", "Predicted CLV (90d)", "Churn risk", "Priority"],
                    [
                        result.cluster_label,
                        f"{result.predicted_clv:,.0f}",
                        f"{churn_pct:.1f}%",
                        result.decision.get("PriorityLevel", "-"),
                    ],
                ],
                widths=[42 * mm, 42 * mm, 42 * mm, 42 * mm],
            )
        )
        story.append(Spacer(1, 4 * mm))

        action_colour = RISK if churn_pct > 70 else GOOD
        story.append(
            Paragraph(
                f'<font color="{action_colour.hexval()}"><b>Recommended action:</b> '
                f'{result.decision.get("RecommendedAction", "-")}</font>',
                style["body"],
            )
        )

        # --- profile ---
        story.append(Paragraph("Behavioural profile", style["heading"]))
        profile_fields = [
            ("Country", profile.get("Country"), "text"),
            ("Lifetime spend", profile.get("Monetary"), "money"),
            ("Orders", profile.get("Frequency"), "int"),
            ("Days since last order", profile.get("Recency"), "int"),
            ("Tenure (days)", profile.get("Tenure"), "int"),
            ("Average basket", profile.get("AverageBasketSize"), "money"),
            ("Avg days between orders", profile.get("AvgInterPurchaseDays"), "int"),
            ("Distinct products", profile.get("DistinctProducts"), "int"),
        ]
        rows = [["Metric", "Value"]]
        for label, value, kind in profile_fields:
            if value is None:
                continue
            if kind == "money":
                shown = f"{float(value):,.2f}"
            elif kind == "int":
                shown = f"{float(value):,.0f}"
            else:
                shown = str(value)
            rows.append([label, shown])
        story.append(_table(rows, widths=[80 * mm, 88 * mm]))

        # --- explanation ---
        story.append(Paragraph("Why this prediction", style["heading"]))
        for task in ("Churn", "CLV"):
            reasons = result.explanations.get(task, [])
            if not reasons:
                continue
            story.append(Paragraph(f"<b>{task}</b> — {reasons[0]}", style["body"]))
            contributors = result.shap_top_features.get(task, [])
            if contributors:
                shap_rows = [["Feature", "Contribution", "Direction"]]
                shap_rows += [
                    [c["Feature"], f"{c['Contribution']:+.4f}", c["Direction"]]
                    for c in contributors
                ]
                story.append(Spacer(1, 2 * mm))
                story.append(_table(shap_rows, widths=[80 * mm, 44 * mm, 44 * mm]))
            story.append(Spacer(1, 3 * mm))

        # --- top products ---
        try:
            products = self._aggregates.customer_products(customer_id, limit=8)
        except Exception:  # noqa: BLE001 - product mix is supplementary
            products = pd.DataFrame()

        if not products.empty:
            story.append(Paragraph("Top products", style["heading"]))
            product_rows = [["Product", "Units", "Revenue"]]
            product_rows += [
                [str(row["Description"])[:48], f"{row['Units']:,.0f}", f"{row['Revenue']:,.2f}"]
                for _, row in products.iterrows()
            ]
            story.append(_table(product_rows, widths=[96 * mm, 34 * mm, 38 * mm]))

        story.append(Spacer(1, 6 * mm))
        story.append(
            Paragraph(
                "SHAP contributions are reported in the model's log-revenue space for CLV "
                "and log-odds for churn; they show direction and relative magnitude, not "
                "currency amounts. Predicted CLV is expected revenue over the next 90 days.",
                style["note"],
            )
        )

        document.build(story)
        logger.info("Customer PDF generated", extra={"customer_id": customer_id})
        return buffer.getvalue()

    # ---------------------------------------------------------------- Excel

    def customers_workbook(self, limit: int = 2000) -> bytes:
        """Multi-sheet workbook: features, segments, revenue and product mix."""
        features = self._features.frame().head(limit)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            summary = pd.DataFrame(
                [
                    {"Metric": key, "Value": value}
                    for key, value in self._aggregates.kpis().items()
                ]
            )
            summary.to_excel(writer, sheet_name="Summary", index=False)
            features.to_excel(writer, sheet_name="Customer Features", index=False)
            self._aggregates.monthly_revenue().to_excel(
                writer, sheet_name="Monthly Revenue", index=False
            )
            self._aggregates.country_revenue().to_excel(
                writer, sheet_name="Country Revenue", index=False
            )
            self._aggregates.top_products().head(200).to_excel(
                writer, sheet_name="Top Products", index=False
            )
            _autosize(writer)

        logger.info("Customer workbook generated", extra={"rows": len(features)})
        return buffer.getvalue()

    def history_workbook(self, history_frame: pd.DataFrame) -> bytes:
        """Prediction audit log as a workbook, with a per-segment summary."""
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame = history_frame if not history_frame.empty else pd.DataFrame(
                columns=["timestamp", "customer_id", "churn_probability"]
            )
            frame.to_excel(writer, sheet_name="Predictions", index=False)

            if not history_frame.empty and "cluster_label" in history_frame:
                by_segment = (
                    history_frame.groupby("cluster_label")
                    .agg(
                        predictions=("customer_id", "count"),
                        avg_churn=("churn_probability", "mean"),
                        avg_clv=("predicted_clv", "mean"),
                        avg_latency_ms=("latency_ms", "mean"),
                    )
                    .round(4)
                    .reset_index()
                )
                by_segment.to_excel(writer, sheet_name="By Segment", index=False)
            _autosize(writer)

        return buffer.getvalue()


def _autosize(writer: pd.ExcelWriter, max_width: int = 42) -> None:
    """Widen columns to fit content so the workbook is readable on open."""
    for worksheet in writer.book.worksheets:
        for column_cells in worksheet.columns:
            longest = max(
                (len(str(cell.value)) for cell in column_cells if cell.value is not None),
                default=0,
            )
            letter = column_cells[0].column_letter
            worksheet.column_dimensions[letter].width = min(max(11, longest + 2), max_width)
