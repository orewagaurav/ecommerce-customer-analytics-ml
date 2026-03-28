"""Business decision engine for customer analytics recommendations.

This module translates model outputs into actionable business decisions.
The rule set is intentionally isolated so product/business teams can extend
or reorder logic without touching model inference code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping


@dataclass(frozen=True)
class RecommendationRule:
    """Declarative rule definition for recommendation selection."""

    name: str
    predicate: Callable[[float, float, str], bool]
    priority: str
    action: str


RULES = [
    RecommendationRule(
        name="high_priority_retention",
        predicate=lambda churn_prob, clv, cluster: churn_prob > 0.7 and clv > 5000,
        priority="High",
        action="High Priority Retention (Discount + Personal Outreach)",
    ),
    RecommendationRule(
        name="offer_discount",
        predicate=lambda churn_prob, clv, cluster: churn_prob > 0.7,
        priority="High",
        action="Offer Discount",
    ),
    RecommendationRule(
        name="vip_benefits",
        predicate=lambda churn_prob, clv, cluster: clv > 10000,
        priority="Medium",
        action="VIP Customer - Exclusive Benefits",
    ),
    RecommendationRule(
        name="retention_campaign",
        predicate=lambda churn_prob, clv, cluster: cluster == "At Risk",
        priority="Medium",
        action="Send Retention Campaign",
    ),
]


def _normalize_inputs(customer_data: Mapping[str, Any]) -> tuple[float, float, str]:
    """Normalize accepted payload keys into (churn_prob, clv, cluster)."""
    clv = float(
        customer_data.get("clv_prediction", customer_data.get("PredictedCLV", 0.0))
    )
    churn_prob = float(
        customer_data.get("churn_probability", customer_data.get("ChurnProbability", 0.0))
    )
    cluster = str(
        customer_data.get("cluster_label", customer_data.get("ClusterLabel", "Unknown"))
    )
    return churn_prob, clv, cluster


def get_recommendation(customer_data: Mapping[str, Any]) -> Dict[str, str]:
    """Generate business recommendation from model outputs.

    Args:
        customer_data: Mapping containing at least these model outputs:
            - clv_prediction (or PredictedCLV)
            - churn_probability (or ChurnProbability)
            - cluster_label (or ClusterLabel)

    Returns:
        Dictionary with:
            - CustomerSegment
            - PriorityLevel
            - RecommendedAction
    """
    churn_prob, clv, cluster = _normalize_inputs(customer_data)

    for rule in RULES:
        if rule.predicate(churn_prob, clv, cluster):
            return {
                "CustomerSegment": cluster,
                "PriorityLevel": rule.priority,
                "RecommendedAction": rule.action,
            }

    return {
        "CustomerSegment": cluster,
        "PriorityLevel": "Low",
        "RecommendedAction": "Normal Engagement",
    }
