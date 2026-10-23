"""Scripted round table. It follows the v2 oracle and adds four seats. It is a harness, not a model."""

from __future__ import annotations

import json

from orbit_agents.schema import ROLE_TOOLS_V3
from orbit_agents.scripted_v2 import ScriptedPolicyV2, commander_steps

# A count no makerspace tool result contains. The gate must refuse it.
PLANTED_VALUE = 1_000_000_000
READ_TOOLS = ROLE_TOOLS_V3["researcher"]


def commander_steps_v3(task: dict) -> list[dict]:
    """The v2 plan, with the first read assigned to the researcher."""
    steps = commander_steps(task)
    assigned = False
    for step in steps:
        if not assigned and step["tool"] in READ_TOOLS:
            step["role"] = "researcher"
            assigned = True
    return steps


class ScriptedPolicyV3(ScriptedPolicyV2):
    """v2 oracle, plus a coordinator, a researcher, a planted red-team claim, and a verifier rejection."""

    provider = "scripted"

    def coordinator(self) -> dict:
        """Open the table. No tools and no answer."""
        return {"brief": "Use only the bundled corpus. Refuse any claim the tool results do not support."}

    def researcher_call(self) -> dict:
        """Take the next read from the oracle. Code and answers stay with the other seats."""
        call = self.remaining.pop(0)
        if call["tool"] not in READ_TOOLS:
            raise RuntimeError(f"researcher cannot run {call['tool']}")
        return {"action": "tool", "tool": call["tool"], "arguments": call["arguments"]}

    def redteam(self) -> dict:
        """One unsupported count. The gate is expected to refuse it."""
        return {
            "claim": {"name": "planted", "value": PLANTED_VALUE, "unit": "count", "citation": "call_1"},
            "why": "This count was not in the tool results.",
        }

    def verifier(self) -> dict:
        """Reject the planted claim. The engine also checks it in code."""
        return {"verdict": "reject", "reason": "The cited tool result does not contain that count."}

    def commander(self) -> dict:
        """The oracle plan, with the first read given to the researcher."""
        return {"steps": commander_steps_v3(self.task)}

    def complete_text(self, messages: list[dict], schema: dict, *, role: str) -> str:
        """Oracle reply for this seat. Round-table seats do not consume a specialist tool call."""
        if role == "coordinator":
            return json.dumps(self.coordinator())
        if role == "researcher":
            return json.dumps(self.researcher_call())
        if role == "redteam":
            return json.dumps(self.redteam())
        if role == "verifier":
            return json.dumps(self.verifier())
        return super().complete_text(messages, schema, role=role)
