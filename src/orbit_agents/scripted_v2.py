"""Scripted policies for the six-role team. They follow a task's oracle. They are a harness, not a model.

Each seat gets the oracle call that matches its tools, in oracle order. The
analyst drafts the oracle final only when the turn's schema allows a final.
The two subclasses inject one fault each, the way v1's ReviseOnce does, so the
critic loop and the auditor can be shown on real engine runs.
"""

from __future__ import annotations

import json

from orbit_agents.policy import LLMClient
from orbit_agents.schema import ROLE_TOOLS

TOOL_ROLE = {tool: role for role, tools in ROLE_TOOLS.items() for tool in tools}


def commander_steps(task: dict) -> list[dict]:
    """Assign each oracle call to the specialist whose tool list contains it."""
    steps = []
    for plan_step, call in zip(task["plan"], task["oracle_calls"], strict=True):
        steps.append({"goal": plan_step["goal"], "role": TOOL_ROLE[call["tool"]], "tool": call["tool"]})
    return steps


def _may_answer(schema: dict) -> bool:
    return "final" in schema["properties"]["action"]["enum"]


class ScriptedPolicyV2(LLMClient):
    provider = "scripted"

    def __init__(self, task: dict):
        self.task = task
        self.remaining = list(task["oracle_calls"])

    def _dump(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=True)

    def commander(self) -> dict:
        return {"steps": commander_steps(self.task)}

    def critic(self) -> dict:
        return {"verdict": "accept", "reason": "The claims match the cited tool results."}

    def final(self) -> dict:
        return {
            "action": "final",
            "status": self.task["status"],
            "answer": self.task["answer"],
            "claims": self.task["gold_claims"],
        }

    def complete_text(self, messages: list[dict], schema: dict, *, role: str) -> str:
        del messages
        if role == "commander":
            return self._dump(self.commander())
        if role == "critic":
            return self._dump(self.critic())
        if role == "analyst" and _may_answer(schema):
            return self._dump(self.final())
        call = self.remaining.pop(0)
        if TOOL_ROLE[call["tool"]] != role:
            raise RuntimeError(f"oracle call {call['tool']} does not belong to {role}")
        return self._dump({"action": "tool", "tool": call["tool"], "arguments": call["arguments"]})


class ReviseOnceV2(ScriptedPolicyV2):
    """The first draft cites the wrong value. The critic sends it back once, then the oracle final stands."""

    def __init__(self, task: dict):
        super().__init__(task)
        self.drafts = 0
        self.verdicts = 0

    def final(self) -> dict:
        self.drafts += 1
        if self.drafts > 1:
            return super().final()
        claims = [dict(claim) for claim in self.task["gold_claims"]]
        bad = next(claim for claim in claims if "value" in claim)
        bad["value"] = 0
        answer = " ".join(
            f"{claim['name']} is {claim['value'] if 'value' in claim else claim['text']}." for claim in claims
        )
        return {"action": "final", "status": "answer", "answer": answer, "claims": claims}

    def critic(self) -> dict:
        self.verdicts += 1
        if self.verdicts == 1:
            return {"verdict": "revise", "reason": "The cited tool result does not contain that figure."}
        return super().critic()


class AuditorCatch(ScriptedPolicyV2):
    """The draft's claims are right, but the answer adds a figure no tool produced.

    The scripted critic accepts it anyway. The auditor, which reads the trace
    and not the critic, refuses the answer.
    """

    def __init__(self, task: dict, extra_number: int):
        super().__init__(task)
        self.extra_number = int(extra_number)

    def final(self) -> dict:
        honest = super().final()
        honest["answer"] = f"{honest['answer']} That is about {self.extra_number} percent of all jobs."
        return honest

    def critic(self) -> dict:
        return {"verdict": "accept", "reason": "The cited claims match the tool results."}
