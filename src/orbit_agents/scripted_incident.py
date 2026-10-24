"""Scripted incident commander. It follows the incident oracle. It is a harness, not a model."""

from __future__ import annotations

from orbit_agents.incident import INVENTORY_BLAME
from orbit_agents.scripted_v3 import ScriptedPolicyV3


class IncidentPolicy(ScriptedPolicyV3):
    """Round table on the checkout outage. The red team blames inventory with a count the files do not contain."""

    def coordinator(self) -> dict:
        """Open the incident. No tools and no answer."""
        return {
            "brief": "Checkout is failing. Use only the incident files. Refuse any claim those files do not support."
        }

    def redteam(self) -> dict:
        """Blame inventory-service with a count that is not in the cited tool result."""
        return {
            "claim": {
                "name": "inventory_errors",
                "value": INVENTORY_BLAME,
                "unit": "count",
                "citation": "call_1",
            },
            "why": "The inventory service is the root cause of the checkout failures.",
        }

    def verifier(self) -> dict:
        """Reject the planted root cause. The engine also checks it in code."""
        return {
            "verdict": "reject",
            "reason": "The cited tool result does not show inventory-service as the cause.",
        }
