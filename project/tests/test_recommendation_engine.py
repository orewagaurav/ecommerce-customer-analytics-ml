"""Tests for the business rule engine.

Rules are ordered and first-match-wins, so precedence is behaviour worth pinning:
a high-value customer about to churn must be routed to retention, not VIP perks.
"""

from __future__ import annotations

import pytest

from src.recommendation_engine import RULES, get_recommendation


def test_high_value_churn_risk_takes_precedence_over_vip():
    decision = get_recommendation(
        {"PredictedCLV": 20000.0, "ChurnProbability": 0.95, "ClusterLabel": "Champions"}
    )
    assert decision["PriorityLevel"] == "High"
    assert "Retention" in decision["RecommendedAction"]


def test_churn_risk_without_high_value_offers_discount():
    decision = get_recommendation(
        {"PredictedCLV": 100.0, "ChurnProbability": 0.85, "ClusterLabel": "Loyal Customers"}
    )
    assert decision["RecommendedAction"] == "Offer Discount"


def test_high_clv_without_churn_risk_is_vip():
    decision = get_recommendation(
        {"PredictedCLV": 15000.0, "ChurnProbability": 0.05, "ClusterLabel": "Champions"}
    )
    assert "VIP" in decision["RecommendedAction"]


def test_at_risk_segment_triggers_retention_campaign():
    decision = get_recommendation(
        {"PredictedCLV": 100.0, "ChurnProbability": 0.1, "ClusterLabel": "At Risk"}
    )
    assert decision["RecommendedAction"] == "Send Retention Campaign"


def test_unremarkable_customer_falls_through_to_default():
    decision = get_recommendation(
        {"PredictedCLV": 100.0, "ChurnProbability": 0.1, "ClusterLabel": "Loyal Customers"}
    )
    assert decision["PriorityLevel"] == "Low"
    assert decision["RecommendedAction"] == "Normal Engagement"


@pytest.mark.parametrize(
    "payload",
    [
        {"clv_prediction": 15000.0, "churn_probability": 0.05, "cluster_label": "Champions"},
        {"PredictedCLV": 15000.0, "ChurnProbability": 0.05, "ClusterLabel": "Champions"},
    ],
)
def test_accepts_both_key_conventions(payload):
    """predict.py and the notebooks use different key casing."""
    assert "VIP" in get_recommendation(payload)["RecommendedAction"]


def test_missing_keys_do_not_raise():
    decision = get_recommendation({})
    assert decision["CustomerSegment"] == "Unknown"
    assert decision["PriorityLevel"] == "Low"


def test_every_rule_is_reachable():
    """A rule shadowed by an earlier one is dead configuration."""
    scenarios = [
        (0.95, 20000.0, "Champions"),
        (0.85, 100.0, "Loyal Customers"),
        (0.05, 15000.0, "Champions"),
        (0.10, 100.0, "At Risk"),
    ]
    matched = set()
    for churn, clv, cluster in scenarios:
        for rule in RULES:
            if rule.predicate(churn, clv, cluster):
                matched.add(rule.name)
                break

    assert matched == {rule.name for rule in RULES}
