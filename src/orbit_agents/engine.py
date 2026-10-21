"""Planner, worker, and critic on one bus, plus the single-agent baseline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from orbit_agents.corpus import corpus_guide
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
    """Take the first JSON object, including one wrapped in prose or a code fence."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    start = 0
    while True:
        index = text.find("{", start)
        if index < 0:
            raise ValueError("no JSON object")
        snippet = text[index:]
        cleaned = re.sub(r",(\s*[}\]])", r"\1", snippet)
        for candidate in (snippet, cleaned):
            try:
                payload, _end = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        start = index + 1


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


# Plain words that show up in almost every question. A line count may cite an identifier
# (a hyphenated name, or a token that contains a digit) and not one of these words.
_GENERIC_TOKENS = {
    "about",
    "according",
    "add",
    "after",
    "all",
    "and",
    "answer",
    "any",
    "are",
    "before",
    "bundled",
    "cache",
    "can",
    "cell",
    "claim",
    "claims",
    "contain",
    "contains",
    "copy",
    "corpus",
    "count",
    "coverage",
    "divided",
    "docs",
    "does",
    "every",
    "failed",
    "failure",
    "file",
    "for",
    "from",
    "give",
    "grams",
    "greater",
    "has",
    "have",
    "hours",
    "how",
    "integer",
    "into",
    "its",
    "jobs",
    "line",
    "lines",
    "log",
    "logs",
    "longer",
    "machine",
    "makerspace",
    "many",
    "material",
    "mean",
    "means",
    "minutes",
    "more",
    "not",
    "note",
    "offline",
    "open",
    "outcome",
    "over",
    "page",
    "person",
    "phrase",
    "python",
    "quotient",
    "read",
    "recorded",
    "result",
    "results",
    "rows",
    "say",
    "sentence",
    "shift",
    "shifts",
    "single",
    "still",
    "table",
    "than",
    "that",
    "their",
    "then",
    "this",
    "those",
    "token",
    "tool",
    "tools",
    "total",
    "under",
    "use",
    "used",
    "using",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "with",
    "workspace",
    "your",
}


def _support_tokens(question: str) -> list[str]:
    cleaned = "".join(ch if ch.isalnum() or ch == "-" else " " for ch in question)
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in cleaned.split():
        token = raw.strip("-")
        if not token or token in seen or token.isdigit():
            continue
        distinctive = "-" in token or any(ch.isdigit() for ch in token)
        if not distinctive and (len(token) < 4 or token.lower() in _GENERIC_TOKENS):
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _body_texts(result: dict) -> list[str]:
    if not isinstance(result, dict) or result.get("ok") is not True:
        return []
    texts: list[str] = []
    if isinstance(result.get("text"), str):
        texts.append(result["text"])
    if isinstance(result.get("stdout"), str):
        texts.append(result["stdout"])
    for match in result.get("matches") or []:
        if isinstance(match, dict) and isinstance(match.get("text"), str):
            texts.append(match["text"])
    return texts


def _printed_integers(result: dict) -> set[int]:
    """Integers printed as their own line by a successful python call.

    A number buried in other printed text is handled by contains_number on the result.
    """
    if not isinstance(result, dict) or result.get("ok") is not True:
        return set()
    stdout = result.get("stdout")
    if not isinstance(stdout, str):
        return set()
    found: set[int] = set()
    lines = stdout.splitlines() or [stdout]
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"0|-?[1-9]\d*", stripped):
            found.add(int(stripped))
    return found


def _line_counts(question: str, result: dict) -> set[int]:
    """Positive line counts for distinctive tokens from the question. Zero is not support."""
    texts = _body_texts(result)
    if not texts:
        return set()
    counts: set[int] = set()
    for token in _support_tokens(question):
        for text in texts:
            count = sum(1 for line in text.splitlines() if token in line)
            if count > 0:
                counts.add(count)
    return counts


def _cited_result(claim: dict, results: list) -> dict | None:
    match = re.fullmatch(r"call_(\d+)", claim.get("citation", ""))
    if not match:
        return None
    index = int(match.group(1)) - 1
    if index < 0 or index >= len(results):
        return None
    result = results[index]["result"]
    if not isinstance(result, dict):
        return None
    return result


def _value_supported(question: str, value: int, result: dict, blob: str) -> bool:
    """A cited number is the token itself, a printed python integer, or a matching line count."""
    if contains_number(blob, value):
        return True
    if value in _printed_integers(result):
        return True
    return value > 0 and value in _line_counts(question, result)


def claims_are_grounded(question: str, claims: list, results: list, answer: str) -> bool:
    supported: set[str] = set()
    for claim in claims:
        result = _cited_result(claim, results)
        if result is None:
            return False
        blob = result_text(result)
        if "value" in claim:
            value = int(claim["value"])
            if not _value_supported(question, value, result, blob):
                return False
            supported.add(_number_token(str(value)))
        if "text" in claim and not _text_in_blob(claim["text"], blob):
            return False
    allowed = {_number_token(token) for token in NUMBER_RE.findall(question)}
    allowed.update(supported)
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
    "worker": "One JSON tool call or a final answer. Cite a result that contains the claim, a python call that printed the count, or text with that many matching lines.",
    "critic": "Reply with one JSON object. Accept grounded claims. Otherwise revise. At most 2 revise passes, then the gate refuses unsupported claims.",
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

    def _user(self, rest: str) -> str:
        block = f"Task: {self.task['question']}\n\n{corpus_guide()}"
        if rest:
            block += "\n\n" + rest
        return block

    def _worker_prompt(self, feedback: str) -> list[dict]:
        system = SINGLE_SYSTEM if self.single else WORKER_SYSTEM
        user = self._user(_history(None if self.single else self.plan, self.results, feedback))
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
                    {"role": "user", "content": self._user("")},
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
                        "content": self._user(
                            f"{_history(self.plan, self.results, '')}\n\nWorker final: {dumps(proposal)}"
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
