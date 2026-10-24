"""Orbit Code. Planner, coder, test runner, critic, and a reviewer that is code.

The test runner executes pytest in a temp workspace. The reviewer rejects a
patch whose tests fail or whose diff touches the tests. The critic can send
the coder back. The grounding gate drops a tests_passed claim the run does
not support.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jsonschema

from orbit_agents.code_sandbox import (
    apply_edits,
    claims_grounded,
    review_patch,
    run_pytest,
    snapshot,
    touches_tests,
    unified_diff,
)
from orbit_agents.codebench import materialize, task_prompt
from orbit_agents.engine import MAX_CRITIC_RETRIES, MAX_SCHEMA_TRIES, MAX_TURNS, RunResult, extract_json, repair_message

PLANNER_SYSTEM = (
    "You are the planner on Orbit Code. You cannot edit files. "
    "Reply with one JSON object of this shape: "
    '{"steps": [{"goal": "what to change in the source"}]}. At most 4 steps.'
)
CODER_SYSTEM = (
    "You are the coder on Orbit Code. Reply with one JSON object of this shape: "
    '{"edits": [{"path": "module.py", "content": "full file"}]}. '
    "Edit source files only. Do not edit tests. The workspace is temporary."
)
CRITIC_SYSTEM = (
    "You are the critic. You cannot edit files. "
    "If the tests failed, reply with verdict revise. "
    "If the tests passed and the edit is only source, reply with verdict accept. "
    'Reply with one JSON object: {"verdict": "accept" or "revise", "reason": "short"}.'
)
SINGLE_SYSTEM = CODER_SYSTEM

PLANNER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        "steps": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["goal"],
                "properties": {"goal": {"type": "string", "minLength": 1, "maxLength": 200}},
            },
        }
    },
}
CODER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["edits"],
    "properties": {
        "edits": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 120},
                    "content": {"type": "string", "minLength": 1, "maxLength": 4000},
                },
            },
        }
    },
}
CRITIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "revise"]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 300},
    },
}
SCHEMAS = {"planner": PLANNER_SCHEMA, "coder": CODER_SCHEMA, "critic": CRITIC_SCHEMA}

FLIGHT_RULES_CODE = {
    "team": "code",
    "roles": ["planner", "coder", "test_runner", "critic", "reviewer"],
    "planner": "Writes the fix plan. Does not edit files.",
    "coder": "Edits source in a temp workspace. Tests are off limits.",
    "test_runner": "Runs pytest in that workspace with a timeout and no network.",
    "critic": "Accepts the draft or sends it back. At most 2 revise passes.",
    "reviewer": "Rejects a patch whose tests fail or whose diff touches the tests. This seat is code.",
    "max_turns": MAX_TURNS,
    "max_critic_retries": MAX_CRITIC_RETRIES,
    "max_schema_tries": MAX_SCHEMA_TRIES,
    "pytest_timeout_seconds": 8,
    "network": "disabled",
    "replay_clock_step_seconds": 4,
}


def _coerce(role: str, payload: dict) -> dict:
    """Drop extra fields the schema forbids. A long plan goal is shortened, not rejected."""
    if role == "planner":
        steps = payload.get("steps")
        if not isinstance(steps, list):
            return payload
        cleaned = []
        for step in steps[:4]:
            goal = step.get("goal") if isinstance(step, dict) else step
            if not isinstance(goal, str) or not goal.strip():
                continue
            cleaned.append({"goal": goal.strip()[:200]})
        if cleaned:
            return {"steps": cleaned}
    elif role == "coder":
        edits = payload.get("edits")
        if not isinstance(edits, list):
            return payload
        cleaned = []
        for edit in edits[:4]:
            if not isinstance(edit, dict):
                continue
            path = edit.get("path")
            content = edit.get("content")
            if isinstance(path, str) and path.strip() and isinstance(content, str) and content:
                cleaned.append({"path": path.strip(), "content": content})
        if cleaned:
            return {"edits": cleaned}
    elif role == "critic" and payload.get("verdict") in {"accept", "revise"}:
        reason = payload.get("reason")
        text = reason.strip() if isinstance(reason, str) else ""
        return {"verdict": payload["verdict"], "reason": (text or "See the test result.")[:300]}
    return payload


def _check(role: str, payload: dict) -> str | None:
    try:
        jsonschema.validate(payload, SCHEMAS[role])
    except jsonschema.ValidationError as exc:
        return exc.message
    return None


def _looks_like_diff(content: str) -> bool:
    lines = content.splitlines()
    return any(line.startswith("@@") for line in lines) and any(
        line.startswith("--- ") or line.startswith("+++ ") for line in lines
    )


def _same_line(left: str, right: str) -> bool:
    return left == right or left.rstrip("\n") == right.rstrip("\n")


def _apply_unified(original: str, diff: str) -> str | None:
    """Apply one unified diff to the original file. None when the hunks do not fit."""
    lines = original.splitlines(keepends=True)
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in diff.splitlines(keepends=True):
        if line.startswith("@@"):
            current = [line]
            hunks.append(current)
        elif current is not None:
            current.append(line)
    if not hunks:
        return None
    for hunk in reversed(hunks):
        marker = hunk[0].split("@@")
        if len(marker) < 2:
            return None
        start_text = marker[1].strip().split()[0].split(",")[0]
        try:
            start = int(start_text)
        except ValueError:
            return None
        index = max(start - 1, 0)
        cursor = index
        built: list[str] = []
        for raw in hunk[1:]:
            if raw.startswith("\\"):
                continue
            prefix, body = raw[:1], raw[1:]
            if prefix == " ":
                if cursor >= len(lines) or not _same_line(lines[cursor], body):
                    return None
                built.append(lines[cursor])
                cursor += 1
            elif prefix == "-":
                if cursor >= len(lines):
                    return None
                cursor += 1
            elif prefix == "+":
                built.append(body if body.endswith("\n") or body == "" else body + "\n")
            else:
                return None
        lines = lines[:index] + built + lines[cursor:]
    return "".join(lines)


def _place_edit(workspace: Path, edit: dict, before: dict[str, str]) -> dict:
    """Map a model path onto a workspace file, and expand a unified diff into file contents."""
    from orbit_agents.code_sandbox import safe_relative

    relative = safe_relative(edit["path"])
    content = edit["content"]
    if relative not in before:
        name = Path(relative).name
        matches = [path for path in before if Path(path).name == name and path.endswith(".py")]
        if len(matches) == 1:
            relative = matches[0]
    if _looks_like_diff(content):
        applied = _apply_unified(before.get(relative, ""), content)
        if applied is not None:
            content = applied
    return {"path": relative, "content": content}


class CodeRun:
    """One task. single=True is one coder turn. The team adds planner, critic, and retries."""

    def __init__(self, task: dict, policy, workspace: Path, *, single: bool = False, on_event=None, root: Path | None = None):
        self.task = task
        self.policy = policy
        self.parent = workspace
        self.single = single
        self.on_event = on_event
        self.root = root
        self.events: list[dict] = []
        self.n_steps = 0
        self.schema_valid = 0
        self.schema_invalid = 0
        self.feedback = ""
        self.workspace: Path | None = None
        self.before: dict[str, str] = {}

    def _event(self, agent: str, kind: str, **payload) -> None:
        event = {"index": len(self.events), "agent": agent, "kind": kind, **payload}
        event["t_sec"] = event["index"] * FLIGHT_RULES_CODE["replay_clock_step_seconds"]
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def _user(self, extra: str) -> str:
        base = task_prompt(self.task, self.root) if self.root else task_prompt(self.task)
        if not extra:
            return base
        return f"{base}\n\n{extra}"

    def _complete(self, role: str, messages: list[dict]) -> dict | None:
        prompt = list(messages)
        for _attempt in range(MAX_SCHEMA_TRIES + 1):
            if self.n_steps >= MAX_TURNS:
                return None
            self.n_steps += 1
            raw = self.policy.complete_text(prompt, SCHEMAS[role], role=role)
            self._event(role, "message", step=self.n_steps, content=raw)
            try:
                payload = _coerce(role, extract_json(raw))
                error = _check(role, payload)
            except ValueError as exc:
                payload = None
                error = str(exc)
            if error is None and payload is not None:
                self.schema_valid += 1
                self._event(role, "parsed", step=self.n_steps, payload=payload)
                return payload
            self.schema_invalid += 1
            self._event(role, "invalid", step=self.n_steps, error=error or "invalid")
            prompt = prompt + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_message(error or "invalid")},
            ]
        return None

    def _prepare(self) -> None:
        if self.workspace is not None and self.workspace.exists():
            shutil.rmtree(self.workspace)
        dest = self.parent / "repo"
        kwargs = {"root": self.root} if self.root is not None else {}
        self.workspace = materialize(self.task, dest, **kwargs)
        self.before = snapshot(self.workspace)

    def _diff(self) -> str:
        assert self.workspace is not None
        return unified_diff(self.before, self.workspace)

    def _pytest(self) -> dict:
        assert self.workspace is not None
        return run_pytest(self.workspace)

    def _summary(self, result: dict) -> str:
        """A count line. Pytest's traceback text changes across versions, so the trace does not store it."""
        if result.get("timed_out"):
            return "timed out"
        if result.get("ok"):
            return f"{result.get('n_passed', 0)} passed"
        if result.get("n_failed"):
            return f"{result['n_failed']} failed"
        if result.get("n_errors"):
            return f"{result['n_errors']} error"
        return "no tests passed"

    def _critic_note(self, result: dict) -> str:
        state = "Tests passed." if result.get("ok") else "Tests failed."
        return f"{state} Summary: {result.get('summary', '')}\n\nDiff:\n{self._diff()[-1500:]}"

    def _coder_extra(self, plan_text: str) -> str:
        parts = [part for part in (plan_text, self.feedback) if part]
        return "\n\n".join(parts)

    def _ask_critic(self, pytest_result: dict) -> dict | None:
        return self._complete(
            "critic",
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": self._user(self._critic_note(pytest_result))},
            ],
        )

    def execute(self) -> RunResult:
        """Run the loop and return the trace. The reviewer runs in this process.

        A planner reply that fails the schema does not skip the coder. A red
        build is sent back even when the critic says accept, unless the diff
        touches a test file. That stop stays a rejection.
        """
        self._prepare()
        plan_text = ""
        if not self.single:
            plan = self._complete(
                "planner",
                [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": self._user("")}],
            )
            if plan is not None:
                goals = [step["goal"] for step in plan["steps"]]
                plan_text = "Plan:\n" + "\n".join(f"- {goal}" for goal in goals)
        review = None
        pytest_result = None
        critic_payload = None
        revises = 0
        while self.n_steps < MAX_TURNS:
            edits = self._complete(
                "coder",
                [
                    {"role": "system", "content": SINGLE_SYSTEM if self.single else CODER_SYSTEM},
                    {"role": "user", "content": self._user(self._coder_extra(plan_text))},
                ],
            )
            if edits is None or self.workspace is None:
                break
            placed = [_place_edit(self.workspace, edit, self.before) for edit in edits["edits"]]
            apply_edits(self.workspace, placed)
            diff = self._diff()
            self._event("coder", "edit", step=self.n_steps, diff=diff, paths=[item["path"] for item in placed])
            pytest_result = self._pytest()
            self._event(
                "test_runner",
                "test",
                step=self.n_steps,
                ok=bool(pytest_result["ok"]),
                summary=self._summary(pytest_result),
                stdout=self._summary(pytest_result),
            )
            review = review_patch(diff=diff, pytest_result=pytest_result)
            self._event("reviewer", "review", step=self.n_steps, verdict=review["verdict"], reason=review["reason"])
            if self.single:
                break
            if review["verdict"] == "accept" or touches_tests(diff):
                critic_payload = self._ask_critic(pytest_result)
                break
            if self.n_steps < MAX_TURNS:
                critic_payload = self._ask_critic(pytest_result)
            if revises >= MAX_CRITIC_RETRIES:
                break
            revises += 1
            reason = "Tests failed."
            if critic_payload and critic_payload.get("verdict") == "revise":
                reason = critic_payload["reason"]
            self._event("critic", "revise", step=self.n_steps, reason=reason)
            self.feedback = self._critic_note(pytest_result)
            if critic_payload and critic_payload.get("verdict") == "revise":
                self.feedback = f"{self.feedback}\n\nCritic: {critic_payload['reason']}"
        self._last_critic = critic_payload
        return self._finish(review, pytest_result)

    def _finish(self, review: dict | None, pytest_result: dict | None) -> RunResult:
        critic_accepted = getattr(self, "_last_critic", None)
        critic_accepted = bool(critic_accepted and critic_accepted.get("verdict") == "accept")
        shipped = bool(review and review["verdict"] == "accept" and pytest_result and pytest_result.get("ok"))
        if shipped:
            claims = [{"name": "tests_passed", "value": True, "unit": "bool", "citation": "pytest"}]
            status = "answer"
            answer = "Tests passed. The reviewer accepted the patch."
        else:
            claims = []
            status = "refuse"
            answer = (review or {}).get("reason") or "No patch was accepted."
        override = False
        if not claims_grounded(claims, pytest_result, review):
            status = "refuse"
            answer = "The grounding gate refused a tests_passed claim the run does not support."
            claims = []
            override = True
        elif status == "refuse" and critic_accepted and not self.single:
            override = True
            self._event(
                "reviewer",
                "gate",
                code_verdict="refuse",
                reason=answer,
                claim={"name": "tests_passed"},
            )
        elif status == "answer":
            self._event("reviewer", "gate", code_verdict="pass", reason=answer, claim={"name": "tests_passed", "value": True})
        self._event("reviewer", "final", status=status, answer=answer)
        return RunResult(
            task_id=self.task["id"],
            mode="single" if self.single else "team",
            status=status,
            answer=answer,
            claims=claims,
            events=self.events,
            examples=[],
            n_steps=self.n_steps,
            schema_valid=self.schema_valid,
            schema_invalid=self.schema_invalid,
            tool_calls=[],
            gate_overrode=override,
            messages=[],
        )


def run_code_team(task: dict, policy, workspace: Path, *, root: Path | None = None, on_event=None) -> RunResult:
    """Team with planner, coder, test runner, critic, and reviewer."""
    return CodeRun(task, policy, workspace, single=False, on_event=on_event, root=root).execute()


def run_code_single(task: dict, policy, workspace: Path, *, root: Path | None = None, on_event=None) -> RunResult:
    """One coder turn, then the same test runner and reviewer. No revise loop."""
    return CodeRun(task, policy, workspace, single=True, on_event=on_event, root=root).execute()


def dump_code_trace(task: dict, result: RunResult, *, harness: str | None) -> str:
    """JSONL for Mission Control. The header names the scripted demo when a harness is set."""
    header = {
        "kind": "mission",
        "source": "orbit_agents.code_team",
        "fabricated": False,
        "task_id": result.task_id,
        "mode": result.mode,
        "question": task["issue"],
        "status": result.status,
        "answer": result.answer,
        "claims": result.claims,
        "n_steps": result.n_steps,
        "schema_valid": result.schema_valid,
        "schema_invalid": result.schema_invalid,
        "n_tool_calls": sum(1 for event in result.events if event["kind"] == "test"),
        "n_retries": sum(1 for event in result.events if event["kind"] == "revise"),
        "gate_overrode": result.gate_overrode,
        "flight_rules": FLIGHT_RULES_CODE,
        "team": "code",
        "scenario": "code",
    }
    if harness:
        header["harness"] = harness
        header["scenario_note"] = "Scripted demo following the oracle. Not a model score."
    rows = [header, *result.events]
    return "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
