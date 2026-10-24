"""Orbit v2: a six-role team on one bus.

    commander -> assigns each step to scout, engineer, or analyst
    scout     -> retrieve, read_file, web_search (cannot run code)
    engineer  -> python, write_file (cannot answer)
    analyst   -> calculator, and the only seat that drafts the final answer
    critic    -> no tools; accepts or sends the draft back (at most 2 times)
    auditor   -> no tools, read-only view of the trace; checks every claim and
                 every number in the final answer and refuses what is unsupported

The auditor is deterministic code, not a model turn, so no policy can argue its
way past it. Turn limits, schema retries, and the revision cap are the v1
constants. Every model turn goes through `LLMClient.complete_text`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from orbit_agents.corpus import corpus_guide
from orbit_agents.engine import (
    ANSWER_NUMBER_RE,
    MAX_CRITIC_RETRIES,
    MAX_SCHEMA_TRIES,
    MAX_TURNS,
    NUMBER_RE,
    REPLAY_CLOCK_STEP_SECONDS,
    STOPPED,
    UNGROUNDED,
    RunResult,
    _cited_result,
    _clip,
    _number_token,
    _text_in_blob,
    _value_supported,
    claims_are_grounded,
    extract_json,
    repair_message,
    result_text,
)
from orbit_agents.prompts import ROLE_PROMPTS_V2
from orbit_agents.schema import (
    ROLE_TOOLS,
    ROLES_V2,
    check_role_v2,
    dumps,
    normalize_payload_v2,
    role_schema_v2,
)
from orbit_agents.tools import call_tool

TEAM_V2 = "v2"

FLIGHT_RULES_V2 = {
    "team": TEAM_V2,
    "roles": list(ROLES_V2),
    "permissions": {role: list(tools) for role, tools in ROLE_TOOLS.items()},
    "commander": "Assigns at most four steps, each to the scout, engineer, or analyst. An empty plan means the corpus cannot answer.",
    "scout": "Reads the corpus with retrieve, read_file, or web_search. Cannot run code or answer.",
    "engineer": "Runs one sandboxed python cell or one workspace write per step. Cannot answer.",
    "analyst": "Uses the calculator, then drafts the only final answer, citing tool results.",
    "critic": "No tools. Accepts the draft or sends it back. At most 2 revise passes.",
    "auditor": "No tools, read-only trace. Checks every claim and number in the final answer against the tool results and refuses anything unsupported.",
    "max_turns": MAX_TURNS,
    "max_critic_retries": MAX_CRITIC_RETRIES,
    "max_schema_tries": MAX_SCHEMA_TRIES,
    "replay_clock_step_seconds": REPLAY_CLOCK_STEP_SECONDS,
}


@dataclass(frozen=True)
class TraceView:
    """What the auditor may read: the question and tool results, as frozen copies."""

    question: str
    results: tuple

    @classmethod
    def from_results(cls, question: str, results: list[dict]) -> TraceView:
        frozen = tuple(MappingProxyType(copy.deepcopy(item)) for item in results)
        return cls(question=question, results=frozen)

    def tool_results(self) -> list[dict]:
        """A fresh copy every call, so nothing the auditor does reaches the bus."""
        return [copy.deepcopy(dict(item)) for item in self.results]


def audit(view: TraceView, status: str, answer: str, claims: list) -> dict:
    """Check each claim and each number in the answer. The verdict is the v1 gate's verdict."""
    results = view.tool_results()
    findings: list[dict] = []
    if status != "answer":
        return {"verdict": "pass", "status": status, "findings": findings, "reason": "A refusal makes no claim."}
    supported_numbers: set[str] = set()
    for claim in claims:
        name = claim.get("name", "")
        result = _cited_result(claim, results)
        if result is None:
            findings.append(
                {"claim": name, "ok": False, "reason": f"citation {claim.get('citation')} is not a tool result"}
            )
            continue
        blob = result_text(result)
        if "value" in claim:
            value = int(claim["value"])
            ok = _value_supported(view.question, value, result, blob)
            findings.append(
                {
                    "claim": name,
                    "value": value,
                    "citation": claim["citation"],
                    "ok": ok,
                    "reason": "the cited result supports the value"
                    if ok
                    else "the cited result does not contain or print this value",
                }
            )
            if ok:
                supported_numbers.add(_number_token(str(value)))
        if "text" in claim:
            ok = _text_in_blob(claim["text"], blob)
            findings.append(
                {
                    "claim": name,
                    "text": claim["text"],
                    "citation": claim["citation"],
                    "ok": ok,
                    "reason": "the cited result contains the text"
                    if ok
                    else "the cited result does not contain this text",
                }
            )
    allowed = {_number_token(token) for token in NUMBER_RE.findall(view.question)} | supported_numbers
    for item in results:
        allowed.update(_number_token(token) for token in NUMBER_RE.findall(result_text(item["result"])))
    for token in ANSWER_NUMBER_RE.findall(answer):
        number = _number_token(token)
        ok = number in allowed
        findings.append(
            {
                "number": number,
                "ok": ok,
                "reason": "appears in the question or a tool result"
                if ok
                else "appears in no tool result and not in the question",
            }
        )
    # The verdict is the v1 gate itself, so v1 and v2 refuse exactly the same claims.
    # The findings explain that verdict one claim and one number at a time.
    grounded = claims_are_grounded(view.question, claims, results, answer)
    reason = "Every claim and number is supported by the trace." if grounded else UNGROUNDED
    return {"verdict": "pass" if grounded else "refuse", "status": status, "findings": findings, "reason": reason}


def _history(plan: dict | None, results: list[dict], feedback: str) -> str:
    lines = []
    if plan is not None:
        lines.append("Plan: " + dumps(plan))
    if results:
        lines.append("Tool results:")
        for index, item in enumerate(results, start=1):
            lines.append(
                f"call_{index} {item['by']} {item['tool']} {dumps(item['arguments'])} -> {_clip(result_text(item['result']))}"
            )
    else:
        lines.append("Tool results: none yet.")
    if feedback:
        lines.append("Critic says revise: " + feedback)
    return "\n".join(lines)


class TeamV2Run:
    def __init__(self, task: dict, policy, workspace: Path, *, on_event=None):
        self.task = task
        self.policy = policy
        self.workspace = workspace
        self.on_event = on_event
        self.events: list[dict] = []
        self.examples: list[dict] = []
        self.results: list[dict] = []
        self.n_steps = 0
        self.schema_valid = 0
        self.schema_invalid = 0
        self.plan: dict | None = None

    def _event(self, agent: str, kind: str, **payload) -> None:
        event = {"index": len(self.events), "agent": agent, "kind": kind, **payload}
        event["t_sec"] = event["index"] * REPLAY_CLOCK_STEP_SECONDS
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def _complete(self, role: str, messages: list[dict], *, may_answer: bool = False) -> dict | None:
        prompt = list(messages)
        schema = role_schema_v2(role, may_answer=may_answer)
        for _attempt in range(MAX_SCHEMA_TRIES + 1):
            if self.n_steps >= MAX_TURNS:
                return None
            self.n_steps += 1
            raw = self.policy.complete_text(prompt, schema, role=role)
            self._event(role, "message", step=self.n_steps, content=raw)
            try:
                payload = normalize_payload_v2(role, extract_json(raw))
                error = check_role_v2(role, payload, may_answer=may_answer)
            except ValueError as exc:
                payload = None
                error = str(exc)
            if error is None and payload is not None:
                self.schema_valid += 1
                self.examples.append(
                    {"role": role, "messages": prompt + [{"role": "assistant", "content": dumps(payload)}]}
                )
                self._event(role, "parsed", payload=payload)
                return payload
            self.schema_invalid += 1
            self._event(role, "invalid", error=error or "invalid")
            prompt = prompt + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_message(error or "invalid")},
            ]
        return None

    def _user(self, rest: str) -> str:
        block = f"Task: {self.task['question']}\n\n{corpus_guide()}"
        if rest:
            block += "\n\n" + rest
        return block

    def _messages(self, role: str, rest: str) -> list[dict]:
        return [{"role": "system", "content": ROLE_PROMPTS_V2[role]}, {"role": "user", "content": self._user(rest)}]

    def _run_tool(self, role: str, parsed: dict) -> None:
        if parsed["tool"] not in ROLE_TOOLS[role]:
            raise PermissionError(f"{role} may not call {parsed['tool']}")
        result = call_tool(parsed["tool"], parsed["arguments"], self.workspace)
        self.results.append({"by": role, "tool": parsed["tool"], "arguments": parsed["arguments"], "result": result})
        self._event("tool", "result", call=f"call_{len(self.results)}", tool=parsed["tool"], by=role, result=result)

    def _plan(self) -> dict:
        return self._complete("commander", self._messages("commander", "")) or {"steps": []}

    def _run_steps(self, plan: dict) -> None:
        for number, step in enumerate(plan["steps"], start=1):
            if self.n_steps >= MAX_TURNS:
                return
            role = step["role"]
            self._event("commander", "assign", role=role, goal=step["goal"], tool=step["tool"], step_number=number)
            rest = _history(self.plan, self.results, "") + f"\n\nYour step: {step['goal']} (tool: {step['tool']})."
            parsed = self._complete(role, self._messages(role, rest))
            if parsed is None:
                self._event(role, "step_failed", goal=step["goal"])
                continue
            self._run_tool(role, parsed)

    def _draft(self, feedback: str) -> dict | None:
        """The analyst may compute with the calculator, then writes the final."""
        while self.n_steps < MAX_TURNS:
            rest = _history(self.plan, self.results, feedback) + "\n\nDraft the team's final answer now."
            parsed = self._complete("analyst", self._messages("analyst", rest), may_answer=True)
            if parsed is None:
                return None
            if parsed["action"] == "final":
                self._event(
                    "analyst", "draft", status=parsed["status"], answer=parsed["answer"], claims=parsed["claims"]
                )
                return parsed
            self._run_tool("analyst", parsed)
        return None

    def execute(self) -> RunResult:
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
            # A revision is the analyst's redraft from the same tool results, so a revise
            # costs the two turns it costs in v1 (draft and verdict). The commander does not replan.
            feedback = verdict["reason"]
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
            mode="team_v2",
            status=status,
            answer=answer,
            claims=claims,
            events=self.events,
            examples=self.examples,
            n_steps=self.n_steps,
            schema_valid=self.schema_valid,
            schema_invalid=self.schema_invalid,
            tool_calls=[
                {"tool": item["tool"], "arguments": item["arguments"], "by": item["by"]} for item in self.results
            ],
            gate_overrode=gate,
            messages=self.examples[0]["messages"] if self.examples else [],
        )


def run_team_v2(
    task: dict,
    policy: Any,
    workspace: Path,
    on_event: Callable[[dict], None] | None = None,
) -> RunResult:
    """Run the six-role team until the auditor accepts or the turn limit hits."""
    return TeamV2Run(task, policy, workspace, on_event=on_event).execute()
