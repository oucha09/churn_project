import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from retention import enrich_with_retention_segments, retention_recommendation


def test_recommendation_is_personalized_for_high_value_client():
    client = {"MonthlyCharges": 120, "tenure": 30, "Contract": "Month-to-month"}

    recommendation = retention_recommendation(client, 0.85)

    assert "prioritaire" in recommendation


def test_segment_enrichment_preserves_rows_and_adds_actions():
    clients = pd.DataFrame(
        {
            "MonthlyCharges": [100, 40],
            "tenure": [5, 60],
            "Contract": ["Month-to-month", "Two year"],
            "churn_probability": [0.8, 0.1],
        }
    )

    enriched = enrich_with_retention_segments(clients)

    assert len(enriched) == 2
    assert {"risk_level", "customer_segment", "recommendation"} <= set(enriched)
