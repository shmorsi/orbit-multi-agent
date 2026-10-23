"""Orbit v3: the v2 crew plus a coordinator, researcher, red team, and verifier.

The specialist loop, turn limit, and auditor are the v2 ones. The four new seats
do not change that loop. The red team's claim is checked in code and is not
written into the answer the auditor scores.
"""

from __future__ import annotations

from pathlib import Path

from orbit_agents.engine import (
    MAX_CRITIC_RETRIES,
    MAX_SCHEMA_TRIES,
    MAX_TURNS,
    REPLAY_CLOCK_STEP_SECONDS,
    STOPPED,
    UNGROUNDED,
    RunResult,
    extract_json,
    repair_message,
)
from orbit_agents.prompts import ROLE_PROMPTS_V2, ROLE_PROMPTS_V3
from orbit_agents.schema import ROLE_TOOLS_V3, ROLES_V3, check_role_v3, dumps, normalize_payload_v2, role_schema_v3
from orbit_agents.team_v2 import TEAM_V2, TeamV2Run, TraceView, audit
from orbit_agents.tools import call_tool

TEAM_V3 = "v3"
# Coordinator, red team, and verifier. The researcher is a specialist step.
ROUND_TURNS = 3

FLIGHT_RULES_V3 = {
    "team": TEAM_V3,
    "roles": list(ROLES_V3),
    "builds_on": TEAM_V2,
    "permissions": {role: list(tools) for role, tools in ROLE_TOOLS_V3.items()},
    "coordinator": "Opens the round table. No tools and no answer.",
    "researcher": "Reads the corpus with retrieve, read_file, or web_search. Cannot run code or answer.",
    "redteam": "Proposes one unsupported claim. That claim is not the team's answer.",
    "verifier": "Re-checks the red-team claim against the tool results. No tools.",
    "auditor": "No tools, read-only trace. The v2 gate. It scores the analyst draft, not the red-team claim.",
    "max_turns": MAX_TURNS,
    "max_critic_retries": MAX_CRITIC_RETRIES,
    "max_schema_tries": MAX_SCHEMA_TRIES,
    "round_turns": ROUND_TURNS,
    "replay_clock_step_seconds": REPLAY_CLOCK_STEP_SECONDS,
}


class TeamV3Run(TeamV2Run):
    """v2 execution, with round-table turns that do not spend the specialist turn limit."""

    def __init__(self, task: dict, policy, workspace: Path, *, on_event=None):
        super().__init__(task, policy, workspace, on_event=on_event)
        self.round_steps = 0

    def _messages(self, role: str, rest: str) -> list[dict]:
        system = ROLE_PROMPTS_V3[role] if role in ROLE_PROMPTS_V3 else ROLE_PROMPTS_V2[role]
        return [{"role": "system", "content": system}, {"role": "user", "content": self._user(rest)}]

    def _complete(self, role: str, messages: list[dict], *, may_answer: bool = False) -> dict | None:
        prompt = list(messages)
        schema = role_schema_v3(role, may_answer=may_answer)
        for _attempt in range(MAX_SCHEMA_TRIES + 1):
            if self.n_steps >= MAX_TURNS:
                return None
            self.n_steps += 1
            raw = self.policy.complete_text(prompt, schema, role=role)
            self._event(role, "message", step=self.n_steps, content=raw)
            try:
                payload = normalize_payload_v2(role, extract_json(raw))
                error = check_role_v3(role, payload, may_answer=may_answer)
            except ValueError as exc:
                payload = None
                error = str(exc)
            if error is None and payload is not None:
                self.schema_valid += 1
                self.examples.append({"role": role, "messages": prompt + [{"role": "assistant", "content": dumps(payload)}]})
                self._event(role, "parsed", payload=payload)
                return payload
            self.schema_invalid += 1
            self._event(role, "invalid", error=error or "invalid")
            prompt = prompt + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_message(error or "invalid")},
            ]
        return None

    def _run_tool(self, role: str, parsed: dict) -> None:
        if parsed["tool"] not in ROLE_TOOLS_V3[role]:
            raise PermissionError(f"{role} may not call {parsed['tool']}")
        result = call_tool(parsed["tool"], parsed["arguments"], self.workspace)
        self.results.append({"by": role, "tool": parsed["tool"], "arguments": parsed["arguments"], "result": result})
        self._event("tool", "result", call=f"call_{len(self.results)}", tool=parsed["tool"], by=role, result=result)

    def _round(self, role: str, rest: str) -> dict | None:
        """One round-table turn. It does not consume the specialist turn limit."""
        self.round_steps += 1
        schema = role_schema_v3(role)
        messages = self._messages(role, rest)
        raw = self.policy.complete_text(messages, schema, role=role)
        step = self.n_steps + self.round_steps
        self._event(role, "message", step=step, content=raw)
        try:
            payload = extract_json(raw)
            error = check_role_v3(role, payload)
        except ValueError as exc:
            payload = None
            error = str(exc)
        if error is None and payload is not None:
            self.schema_valid += 1
            self._event(role, "parsed", payload=payload)
            return payload
        self.schema_invalid += 1
        self._event(role, "invalid", error=error or "invalid")
        return None

    def _attack(self) -> None:
        from orbit_agents.team_v2 import _history

        rest = _history(self.plan, self.results, "")
        attack = self._round("redteam", rest + "\n\nPropose one unsupported claim.")
        claim = attack.get("claim") if isinstance(attack, dict) else None
        if not isinstance(claim, dict):
            self._event("redteam", "attack_failed")
            return
        report = audit(
            TraceView.from_results(self.task["question"], self.results),
            "answer",
            "",
            [claim],
        )
        self._event(
            "verifier",
            "gate",
            code_verdict=report["verdict"],
            reason=report["reason"],
            claim=claim,
            findings=report["findings"],
        )
        self._round("verifier", rest + "\n\nRed-team claim: " + dumps(claim))

    def execute(self) -> RunResult:
        """Run the round table. The auditor still scores only the analyst draft."""
        from orbit_agents.team_v2 import _history

        self._round("coordinator", "")
        self.plan = self._plan()
        self._run_steps(self.plan)
        proposal = None
        feedback = ""
        revises = 0
        while self.n_steps < MAX_TURNS:
            draft = self._draft(feedback)
            if draft is None:
                break
            proposal = draft
            rest = _history(self.plan, self.results, "") + f"\n\nAnalyst draft: {dumps(proposal)}"
            verdict = self._complete("critic", self._messages("critic", rest))
            if verdict is None or verdict["verdict"] == "accept":
                break
            revises += 1
            self._event("critic", "revise", reason=verdict["reason"])
            if revises > MAX_CRITIC_RETRIES:
                break
            feedback = verdict["reason"]
        self._attack()
        status, answer, claims, gate = "refuse", STOPPED, [], False
        if proposal is not None:
            status = proposal["status"]
            answer = proposal["answer"]
            claims = proposal["claims"] if status == "answer" else []
            report = audit(TraceView.from_results(self.task["question"], self.results), status, answer, claims)
            self._event("auditor", "audit", **report)
            if report["verdict"] == "refuse":
                status, answer, claims, gate = "refuse", UNGROUNDED, [], True
        self._event("auditor", "final", status=status, answer=answer, claims=claims)
        return RunResult(
            task_id=self.task["id"],
            mode="team_v3",
            status=status,
            answer=answer,
            claims=claims,
            events=self.events,
            examples=self.examples,
            n_steps=self.n_steps,
            schema_valid=self.schema_valid,
            schema_invalid=self.schema_invalid,
            tool_calls=[{"tool": item["tool"], "arguments": item["arguments"], "by": item["by"]} for item in self.results],
            gate_overrode=gate,
            messages=self.examples[0]["messages"] if self.examples else [],
        )


def run_team_v3(task: dict, policy, workspace: Path, on_event=None) -> RunResult:
    """Run the round table and return the trace."""
    return TeamV3Run(task, policy, workspace, on_event=on_event).execute()
