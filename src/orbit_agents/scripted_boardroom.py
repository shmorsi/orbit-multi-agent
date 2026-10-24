"""Scripted Northline board. The CFO states a revenue figure the CSV does not contain."""

from __future__ import annotations

import json

from orbit_agents.boardroom import (
    REVENUE_DELTA,
    activation_rate,
    freeze_date,
    launch_crew,
    q4_operating_cost,
    q4_revenue,
)
from orbit_agents.policy import LLMClient


class BoardroomPolicy(LLMClient):
    """Oracle for the supported seats. The CFO revenue claim is the planted miss."""

    provider = "scripted"

    def __init__(self, task: dict):
        self.task = task

    def complete_text(self, messages: list[dict], schema: dict, *, role: str) -> str:
        del messages, schema
        if role == "ceo":
            payload = {"brief": "Decide the launch from the files. Drop any figure a file does not state."}
        elif role == "cfo":
            payload = {
                "claims": [
                    {
                        "name": "q4_revenue_usd",
                        "value": q4_revenue() + REVENUE_DELTA,
                        "source": "financials.csv",
                        "quarter": "2025Q4",
                        "column": "revenue_usd",
                    },
                    {
                        "name": "q4_operating_cost_usd",
                        "value": q4_operating_cost(),
                        "source": "financials.csv",
                        "quarter": "2025Q4",
                        "column": "operating_cost_usd",
                    },
                ]
            }
        elif role == "cto":
            payload = {
                "claims": [
                    {
                        "name": "activation_rate_pct",
                        "value": activation_rate(),
                        "source": "product_metrics.csv",
                        "metric": "activation_rate_pct",
                    }
                ]
            }
        elif role == "coo":
            payload = {
                "claims": [
                    {"name": "launch_crew", "value": launch_crew(), "source": "memos/launch.md"},
                    {"name": "freeze_date", "value": freeze_date(), "source": "memos/launch.md"},
                ]
            }
        else:
            raise ValueError(role)
        return json.dumps(payload)
