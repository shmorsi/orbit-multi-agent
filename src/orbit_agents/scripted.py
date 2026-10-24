"""A backend that follows a task's oracle. It is a harness, not a model."""

from __future__ import annotations

import json


class ScriptedBackend:
    def __init__(self, task: dict):
        self.task = task
        self.remaining = list(task["oracle_calls"])

    def complete(self, *, role: str, messages: list[dict]) -> str:
        """Oracle reply for the v1 role. This is a harness, not a model."""
        del messages
        if role == "planner":
            return json.dumps({"steps": self.task["plan"]}, ensure_ascii=True)
        if role == "critic":
            return json.dumps({"verdict": "accept", "reason": "The claims match the cited tool results."})
        if self.remaining:
            call = self.remaining.pop(0)
            return json.dumps(
                {"action": "tool", "tool": call["tool"], "arguments": call["arguments"]},
                ensure_ascii=True,
            )
        return json.dumps(
            {
                "action": "final",
                "status": self.task["status"],
                "answer": self.task["answer"],
                "claims": self.task["gold_claims"],
            },
            ensure_ascii=True,
        )


class ReviseOnce(ScriptedBackend):
    """First final is deliberately wrong; the critic sends it back; the next final is the oracle."""

    def __init__(self, task: dict):
        super().__init__(task)
        self.sent_bad_final = False
        self.revised = False

    def complete(self, *, role: str, messages: list[dict]) -> str:
        """One unsupported draft, then the oracle. Used to show the critic."""
        if role == "worker" and not self.remaining and not self.sent_bad_final:
            self.sent_bad_final = True
            return json.dumps(
                {
                    "action": "final",
                    "status": "answer",
                    "answer": "job_count is 0.",
                    "claims": [{"name": "job_count", "value": 0, "unit": "count", "citation": "call_1"}],
                },
                ensure_ascii=True,
            )
        if role == "critic" and not self.revised:
            self.revised = True
            return json.dumps(
                {"verdict": "revise", "reason": "The cited tool result does not contain that figure."},
                ensure_ascii=True,
            )
        return super().complete(role=role, messages=messages)
