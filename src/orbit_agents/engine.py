"""Planner, worker, and critic on one bus, plus the single-agent baseline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from orbit_agents.prompts import CRITIC_SYSTEM, PLANNER_SYSTEM, SINGLE_SYSTEM, WORKER_SYSTEM
from orbit_agents.schema import check_role, dumps, normalize_payload
from orbit_agents.tools import call_tool

NUMBER_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])")
ANSWER_NUMBER_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])")
REPAIR = "The previous reply was not valid for this role. Reply with one JSON object and nothing else."


def repair_message(error: str) -> str:
    """Tell the model which check failed. A generic retry left the 0.5B run at zero valid tool calls."""
    detail = " ".join(str(error).split()) or "invalid JSON"
    if len(detail) > 180:
        detail = detail[:180] + "..."
    return (
        "The previous reply was not valid for this role. "
        f"Validator says: {detail} "
        "Reply with one JSON object and nothing else."
    )
UNGROUNDED = "The claims were not in the cited tool results."
STOPPED = "Stopped at the step limit before a grounded answer."
MAX_TURNS = 8
MAX_SCHEMA_TRIES = 2
MAX_CRITIC_RETRIES = 2


@dataclass
class RunResult:
    task_id: str
    mode: str
    status: str
    answer: str
    claims: list
    events: list
    examples: list
    n_steps: int
    schema_valid: int
    schema_invalid: int
    tool_calls: list
    gate_overrode: bool = False
    messages: list = field(default_factory=list)


def extract_json(raw: str) -> dict:
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    decoder = json.JSONDecoder()
    try:
        payload, _end = decoder.raw_decode(raw[start:])
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", raw[start:])
        payload, _end = decoder.raw_decode(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("JSON value is not an object")
    return payload


def contains_number(text: str, value: int) -> bool:
    number = int(value)
    if re.search(rf"(?<![\d.]){number}(?![\d.])", text):
        return True
    return re.search(rf"(?<![\d.]){number}\.0+(?!\d)", text) is not None


def _number_token(token: str) -> str:
    if re.fullmatch(r"\d+\.0+", token):
        return str(int(float(token)))
    return token


def result_text(result: dict) -> str:
    return json.dumps(result, ensure_ascii=True)


def _text_in_blob(text: str, blob: str) -> bool:
    if text in blob:
        return True
    folded = re.sub(r"\s+", " ", text).strip()
    haystack = re.sub(r"\s+", " ", blob)
    return bool(folded) and folded in haystack


def claims_are_grounded(question: str, claims: list, results: list, answer: str) -> bool:
    for claim in claims:
        match = re.fullmatch(r"call_(\d+)", claim.get("citation", ""))
        if not match:
            return False
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(results):
            return False
        blob = result_text(results[index]["result"])
        if "value" in claim and not contains_number(blob, int(claim["value"])):
            return False
        if "text" in claim and not _text_in_blob(claim["text"], blob):
            return False
    allowed = {_number_token(token) for token in NUMBER_RE.findall(question)}
    for item in results:
        allowed.update(_number_token(token) for token in NUMBER_RE.findall(result_text(item["result"])))
    return all(_number_token(token) in allowed for token in ANSWER_NUMBER_RE.findall(answer))


def _clip(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


def _history(plan: dict | None, results: list, feedback: str) -> str:
    lines = []
    if plan is not None:
        lines.append("Plan: " + dumps(plan))
    if results:
        lines.append("Tool results:")
        for index, item in enumerate(results, start=1):
            lines.append(f"call_{index} {item['tool']} {dumps(item['arguments'])} -> {_clip(result_text(item['result']))}")
    else:
        lines.append("Tool results: none yet.")
    if feedback:
        lines.append("Critic says revise: " + feedback)
    return "\n".join(lines)


REPLAY_CLOCK_STEP_SECONDS = 4
FLIGHT_RULES = {
    "planner": "Return at most four steps using the bundled tools. An empty plan means the corpus cannot answer.",
    "worker": "One JSON tool call or a final answer. Every claim cites a tool result that contains it.",
    "critic": "Accept grounded claims. Otherwise revise. At most 2 revise passes, then the gate refuses unsupported claims.",
    "max_turns": MAX_TURNS,
    "max_critic_retries": MAX_CRITIC_RETRIES,
    "max_schema_tries": MAX_SCHEMA_TRIES,
    "replay_clock_step_seconds": REPLAY_CLOCK_STEP_SECONDS,
}


class TeamRun:
    def __init__(self, task: dict, backend, workspace: Path, *, single: bool = False, on_event=None):
        self.task = task
        self.backend = backend
        self.workspace = workspace
        self.single = single
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

    def _complete(self, role: str, messages: list[dict]) -> dict | None:
        prompt = list(messages)
        for _attempt in range(MAX_SCHEMA_TRIES + 1):
            if self.n_steps >= MAX_TURNS:
                return None
            self.n_steps += 1
            raw = self.backend.complete(role=role, messages=prompt)
            self._event(role, "message", step=self.n_steps, content=raw)
            try:
                payload = normalize_payload(role, extract_json(raw))
                error = check_role(role, payload)
            except ValueError as exc:
                payload = None
                error = str(exc)
            if error is None and payload is not None:
                self.schema_valid += 1
                canonical = dumps(payload)
                if role in {"planner", "worker", "critic"}:
                    self.examples.append({"role": role, "messages": prompt + [{"role": "assistant", "content": canonical}]})
                self._event(role, "parsed", payload=payload)
                return payload
            self.schema_invalid += 1
            self._event(role, "invalid", error=error or "invalid")
            prompt = prompt + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_message(error or "invalid")},
            ]
        return None

    def _worker_prompt(self, feedback: str) -> list[dict]:
        system = SINGLE_SYSTEM if self.single else WORKER_SYSTEM
        user = f"Task: {self.task['question']}\n\n{_history(None if self.single else self.plan, self.results, feedback)}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _until_final(self, feedback: str) -> dict | None:
        while self.n_steps < MAX_TURNS:
            parsed = self._complete("worker", self._worker_prompt(feedback))
            if parsed is None:
                return None
            if parsed["action"] == "final":
                return parsed
            result = call_tool(parsed["tool"], parsed["arguments"], self.workspace)
            self.results.append({"tool": parsed["tool"], "arguments": parsed["arguments"], "result": result})
            self._event("tool", "result", call=f"call_{len(self.results)}", tool=parsed["tool"], result=result)
        return None

    def execute(self) -> RunResult:
        feedback = ""
        proposal = None
        if not self.single:
            proposal_plan = self._complete(
                "planner",
                [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": "Task: " + self.task["question"]},
                ],
            )
            self.plan = proposal_plan or {"steps": []}
        revises = 0
        while self.n_steps < MAX_TURNS:
            proposal = self._until_final(feedback)
            if proposal is None or self.single:
                break
            verdict = self._complete(
                "critic",
                [
                    {"role": "system", "content": CRITIC_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Task: {self.task['question']}\n\n"
                            f"{_history(self.plan, self.results, '')}\n\n"
                            f"Worker final: {dumps(proposal)}"
                        ),
                    },
                ],
            )
            if verdict is None or verdict["verdict"] == "accept":
                break
            revises += 1
            self._event("critic", "revise", reason=verdict["reason"])
            if revises > MAX_CRITIC_RETRIES:
                break
            feedback = verdict["reason"]
        status = "refuse"
        answer = STOPPED
        claims: list = []
        gate = False
        if proposal is not None:
            # The checker reads this final. A critic that never returns a verdict
            # must not replace it with the step-limit sentence.
            status = proposal["status"]
            answer = proposal["answer"]
            claims = proposal["claims"]
            if status == "answer" and not claims_are_grounded(self.task["question"], claims, self.results, answer):
                status = "refuse"
                answer = UNGROUNDED
                claims = []
                gate = True
            if status == "refuse":
                claims = []
        if proposal is None:
            status, answer, claims = "refuse", STOPPED, []
        return RunResult(
            task_id=self.task["id"],
            mode="single" if self.single else "team",
            status=status,
            answer=answer,
            claims=claims,
            events=self.events,
            examples=self.examples,
            n_steps=self.n_steps,
            schema_valid=self.schema_valid,
            schema_invalid=self.schema_invalid,
            tool_calls=[{"tool": item["tool"], "arguments": item["arguments"]} for item in self.results],
            gate_overrode=gate,
            messages=self.examples[0]["messages"] if self.examples else [],
        )


def run_team(task: dict, backend, workspace: Path, on_event=None) -> RunResult:
    return TeamRun(task, backend, workspace, single=False, on_event=on_event).execute()


def run_single(task: dict, backend, workspace: Path, on_event=None) -> RunResult:
    return TeamRun(task, backend, workspace, single=True, on_event=on_event).execute()
