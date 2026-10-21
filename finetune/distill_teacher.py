"""Distill planner, worker, and critic trajectories from a Gemini teacher.

The teacher sees train and tune tasks only. The frozen test file is a local
blocklist: its questions are never placed in a request body and never written
as training rows. Tool calls run through Orbit's sandbox.

Cost controls, all on by default:

- hard cap on API requests (800)
- about 8 requests per minute
- resume from a checkpoint, and the cap counts requests already spent
- one preflight request, then the task loop
- HTTP 429 and quota errors stop the process and are not retried
- any other 4xx (400, 401, 403, 404, and the rest) stops the process immediately
- a 5xx response is retried a few times, then the process stops

The key is read from ORBIT_TEACHER_API_KEY. This script does not call the API
during --dry-run.

    python finetune/distill_teacher.py --dry-run
    ORBIT_TEACHER_API_KEY=... python finetune/distill_teacher.py \\
        --max-requests 800 --requests-per-minute 8 --attempts 3 \\
        --output finetune/data/distill
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "finetune"))

from leakage import (  # noqa: E402
    FrozenTestLeak,
    TEST,
    _read,
    blocking_test_task,
    guard_messages,
    overlap_error,
)
from orbit_agents.engine import run_team  # noqa: E402
from orbit_agents.eval.score import score_task  # noqa: E402
from orbit_agents.schema import check_role  # noqa: E402
from orbit_agents.tasks import read_jsonl  # noqa: E402
from synthesize import materialize  # noqa: E402
from variants import materialize_variants  # noqa: E402

DEV_PATH = ROOT / "eval" / "dev.jsonl"
DEFAULT_OUTPUT = ROOT / "finetune" / "data" / "distill"
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_MAX_REQUESTS = 800
DEFAULT_REQUESTS_PER_MINUTE = 8.0
DEFAULT_ATTEMPTS = 3
SERVER_ATTEMPTS = 3
PREFLIGHT_PROMPT = "Reply with an empty JSON object."
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
CHECKPOINT_VERSION = 1


class DistillStop(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


class QuotaStop(DistillStop):
    def __init__(self, detail: str):
        super().__init__("quota", detail)


class BudgetStop(DistillStop):
    def __init__(self):
        super().__init__("request_cap", "request cap reached")


class NetworkStop(DistillStop):
    def __init__(self, detail: str):
        super().__init__("network", detail)


class FatalHTTPStop(DistillStop):
    """A non-retryable HTTP error. No further requests are sent."""

    def __init__(self, detail: str):
        super().__init__("http", detail)


class TeacherHTTPError(FatalHTTPStop):
    """Backward-compatible name for a non-retryable HTTP client error."""


class RequestBudget:
    """Counts every outbound call and spaces them out. The count survives a resume."""

    def __init__(
        self,
        max_requests: int,
        per_minute: float,
        *,
        clock=None,
        sleep=None,
        used: int = 0,
        last_request_at: float | None = None,
    ):
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        if per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.max_requests = int(max_requests)
        self.per_minute = float(per_minute)
        self.min_interval = 60.0 / self.per_minute
        self.clock = time.time if clock is None else clock
        self.sleep = time.sleep if sleep is None else sleep
        self.used = int(used)
        self.last_request_at = last_request_at
        self.prompt_tokens = 0
        self.candidates_tokens = 0
        self.total_tokens = 0
        self.stop_reason: str | None = None
        self.stop_detail = ""

    def acquire(self) -> None:
        if self.stop_reason == "quota":
            raise QuotaStop(self.stop_detail or "stopped after a quota response")
        if self.stop_reason == "network":
            raise NetworkStop(self.stop_detail or "stopped after a network failure")
        if self.stop_reason == "http":
            raise FatalHTTPStop(self.stop_detail or "stopped after an HTTP client error")
        if self.stop_reason == "request_cap" or self.used >= self.max_requests:
            self.stop_reason = "request_cap"
            raise BudgetStop()
        now = float(self.clock())
        if self.last_request_at is not None:
            wait = self.min_interval - (now - self.last_request_at)
            if wait > 0:
                self.sleep(wait)
                now = float(self.clock())
        self.last_request_at = now
        self.used += 1

    def add_usage(self, prompt: int, candidates: int, total: int) -> None:
        self.prompt_tokens += int(prompt)
        self.candidates_tokens += int(candidates)
        self.total_tokens += int(total)


def family_of(task: dict) -> str:
    if task.get("family"):
        return str(task["family"])
    if task.get("expect_refusal"):
        return "refusal"
    tools = [call.get("tool") for call in task.get("oracle_calls") or []]
    if len(tools) > 1:
        return "multi"
    if len(tools) == 1 and tools[0]:
        return str(tools[0])
    return "other"


def interleave(tasks: list[dict]) -> list[dict]:
    """Round-robin families so a request cap still covers refusals, not only counts."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        buckets[family_of(task)].append(task)
    rows: list[dict] = []
    while True:
        progressed = False
        for key in sorted(buckets):
            if buckets[key]:
                rows.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            return rows


def load_pool(workspace: Path) -> list[dict]:
    """Dev tasks, corpus scenarios, and programmatic variants. Never the frozen test."""
    seen: set[str] = set()
    rows: list[dict] = []

    def take(task: dict) -> None:
        question = task["question"]
        if question in seen:
            return
        if task.get("subset") not in {"train", "tune"}:
            return
        if str(task.get("id", "")).startswith("test-") or task.get("split") == "test":
            return
        seen.add(question)
        copied = dict(task)
        copied["family"] = family_of(copied)
        rows.append(copied)

    for task in read_jsonl(DEV_PATH):
        take(task)
    for subset in ("train", "tune"):
        for task in materialize(subset, workspace / "scenarios" / subset):
            take(task)
    for task in materialize_variants(workspace / "variants"):
        take(task)
    ids = [task["id"] for task in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("training pool has duplicate task ids")
    return rows


# Tokens that identify a frozen test item even when the wording is below the 0.9 ratio.
FROZEN_FRAGMENTS = ("E_NOZZLE", "G28", "waterjet", "last Tuesday", "filament brand")


def _python_body(code: str) -> str:
    lines = code.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def oracle_key(call: dict) -> str | None:
    """Identity of a tool call, ignoring the task-id comment. Shared file reads are not keys."""
    tool = call.get("tool")
    args = call.get("arguments") or {}
    if tool == "python":
        body = _python_body(args.get("code") or "")
        return f"python:{body}" if body else None
    if tool in {"retrieve", "web_search"}:
        query = args.get("query") or ""
        return f"{tool}:{query}" if query else None
    if tool == "calculator":
        expression = args.get("expression") or ""
        return f"calculator:{expression}" if expression else None
    if tool == "write_file":
        return "write:" + (args.get("path") or "") + ":" + (args.get("content") or "")
    return None


def uses_frozen_oracle(task: dict, test_rows: list[dict]) -> bool:
    """True when every keyed call in the task is also a frozen-test call."""
    keys = {key for key in (oracle_key(call) for call in task.get("oracle_calls") or []) if key}
    if not keys:
        return False
    frozen = set()
    for test in test_rows:
        for call in test.get("oracle_calls") or []:
            key = oracle_key(call)
            if key:
                frozen.add(key)
    return keys <= frozen


def select_tasks(tasks: list[dict], test_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop anything that would put frozen-test text or the same tool call in a teacher prompt."""
    test_ids = {task["id"] for task in test_rows}
    kept: list[dict] = []
    dropped: list[dict] = []
    for task in tasks:
        task_id = task["id"]
        question = task["question"]
        if task_id in test_ids or str(task_id).startswith("test-") or task.get("subset") == "test":
            dropped.append({"id": task_id, "reason": "test_split"})
            continue
        if any(fragment in question for fragment in FROZEN_FRAGMENTS):
            dropped.append({"id": task_id, "reason": "frozen_fragment"})
            continue
        if blocking_test_task(question, test_rows):
            dropped.append({"id": task_id, "reason": "near_duplicate"})
            continue
        if uses_frozen_oracle(task, test_rows):
            dropped.append({"id": task_id, "reason": "frozen_oracle"})
            continue
        kept.append(task)
    return kept, dropped


def trajectory_ok(task: dict, result) -> tuple[bool, str]:
    """Keep a run only when the checker passes and every turn was schema-valid."""
    scored = score_task(task, result)
    if not scored["success"]:
        return False, "checker"
    if int(result.schema_invalid) != 0:
        return False, "schema"
    examples = list(result.examples or [])
    if not examples:
        return False, "empty"
    for example in examples:
        role = example.get("role")
        if role not in {"planner", "worker", "critic"}:
            return False, "role"
        messages = example.get("messages") or []
        if not messages or messages[-1].get("role") != "assistant":
            return False, "schema"
        try:
            payload = json.loads(messages[-1]["content"])
        except json.JSONDecodeError:
            return False, "schema"
        if check_role(role, payload):
            return False, "schema"
    return True, "ok"


def _alternate_answer(task: dict) -> str:
    parts = []
    for claim in task.get("gold_claims") or []:
        if "value" in claim:
            parts.append(f"{claim['name']} equals {claim['value']}.")
        else:
            parts.append(f"{claim['name']} equals {claim['text']}.")
    return " ".join(parts)


def retarget_rows(task: dict, rows: list[dict]) -> list[dict]:
    """Swap a 'name is value' sentence so a frozen answer string is not copied."""
    if task.get("expect_refusal"):
        return rows
    original = task.get("answer") or ""
    alternate = _alternate_answer(task)
    if not original or original == alternate:
        return rows
    for row in rows:
        for message in row["messages"]:
            content = message.get("content") or ""
            if original in content:
                message["content"] = content.replace(original, alternate)
    return rows


def rows_from_result(task: dict, result) -> list[dict]:
    rows = []
    for example in result.examples:
        if example["role"] not in {"planner", "worker", "critic"}:
            continue
        rows.append(
            {
                "id": task["id"],
                "role": example["role"],
                "messages": copy.deepcopy(example["messages"]),
            }
        )
    return retarget_rows(task, rows)


class GuardedBackend:
    def __init__(self, inner, test_rows: list[dict]):
        self.inner = inner
        self.test_rows = test_rows

    def complete(self, *, role: str, messages: list[dict]) -> str:
        guard_messages(messages, self.test_rows)
        return self.inner.complete(role=role, messages=messages)


def fresh_state() -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "requests": 0,
        "prompt_tokens": 0,
        "candidates_tokens": 0,
        "total_tokens": 0,
        "last_request_at": None,
        "stopped_reason": None,
        "stopped_detail": "",
        "attempts": 0,
        "tasks": {},
        "train_rows": [],
        "tune_rows": [],
    }


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        return fresh_state()
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != CHECKPOINT_VERSION:
        raise SystemExit(f"{path} is not checkpoint version {CHECKPOINT_VERSION}; move it aside to start over")
    state.setdefault("tasks", {})
    state.setdefault("train_rows", [])
    state.setdefault("tune_rows", [])
    state.setdefault("attempts", 0)
    state.setdefault("stopped_detail", "")
    return state


def sync_budget(state: dict, budget: RequestBudget | None) -> None:
    if budget is None:
        return
    state["requests"] = budget.used
    state["prompt_tokens"] = budget.prompt_tokens
    state["candidates_tokens"] = budget.candidates_tokens
    state["total_tokens"] = budget.total_tokens
    state["last_request_at"] = budget.last_request_at


def run_distill(
    tasks: list[dict],
    test_rows: list[dict],
    *,
    backend_for,
    attempts: int,
    workspace: Path,
    state: dict | None = None,
    budget: RequestBudget | None = None,
) -> dict:
    """Rejection-sample team runs. A client or quota error stops the run."""
    state = state or fresh_state()
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    workspace.mkdir(parents=True, exist_ok=True)
    for task in interleave(tasks):
        record = state["tasks"].get(task["id"])
        if record and (record.get("accepted") or int(record.get("attempts") or 0) >= attempts):
            continue
        if record is None:
            record = {
                "attempts": 0,
                "accepted": False,
                "subset": task["subset"],
                "family": family_of(task),
                "last_reject": None,
            }
        while record["attempts"] < attempts and not record["accepted"]:
            backend = GuardedBackend(backend_for(task), test_rows)
            try:
                result = run_team(task, backend, workspace / task["id"])
            except FrozenTestLeak as exc:
                sync_budget(state, budget)
                state["stopped_reason"] = "leakage"
                state["tasks"][task["id"]] = record
                raise DistillStop("leakage", exc.detail) from exc
            except (QuotaStop, BudgetStop, NetworkStop, FatalHTTPStop) as exc:
                sync_budget(state, budget)
                state["stopped_reason"] = exc.reason
                state["stopped_detail"] = exc.detail
                state["tasks"][task["id"]] = record
                return state
            record["attempts"] += 1
            state["attempts"] += 1
            ok, why = trajectory_ok(task, result)
            if ok:
                rows = rows_from_result(task, result)
                leak = overlap_error(rows, [], test_rows)
                if leak:
                    ok = False
                    why = "leakage"
                else:
                    bucket = "train_rows" if task["subset"] == "train" else "tune_rows"
                    state[bucket].extend(rows)
                    record["accepted"] = True
            record["last_reject"] = None if ok else why
            sync_budget(state, budget)
        state["tasks"][task["id"]] = record
    sync_budget(state, budget)
    if state.get("stopped_reason") is None:
        state["stopped_reason"] = None
    return state


def build_stats(state: dict, pool: list[dict], *, model: str, max_requests: int, per_minute: float, attempts: int) -> dict:
    families: dict[str, dict] = {}
    for task in pool:
        bucket = families.setdefault(
            family_of(task),
            {"tasks": 0, "attempted": 0, "attempts": 0, "accepted": 0, "pass_rate": None},
        )
        bucket["tasks"] += 1
    for record in state["tasks"].values():
        bucket = families.setdefault(
            record["family"],
            {"tasks": 0, "attempted": 0, "attempts": 0, "accepted": 0, "pass_rate": None},
        )
        bucket["attempted"] += 1
        bucket["attempts"] += int(record["attempts"])
        bucket["accepted"] += 1 if record.get("accepted") else 0
    for bucket in families.values():
        if bucket["attempts"]:
            bucket["pass_rate"] = bucket["accepted"] / bucket["attempts"]
    return {
        "model": model,
        "max_requests": max_requests,
        "requests_per_minute": per_minute,
        "attempts_per_task": attempts,
        "requests": state["requests"],
        "prompt_tokens": state["prompt_tokens"],
        "candidates_tokens": state["candidates_tokens"],
        "total_tokens": state["total_tokens"],
        "stopped_reason": state.get("stopped_reason"),
        "stopped_detail": state.get("stopped_detail") or "",
        "attempts": state["attempts"],
        "n_accepted_tasks": sum(1 for record in state["tasks"].values() if record.get("accepted")),
        "n_train_dialogs": len(state["train_rows"]),
        "n_tune_dialogs": len(state["tune_rows"]),
        "n_train_tasks_pool": sum(1 for task in pool if task["subset"] == "train"),
        "n_tune_tasks_pool": sum(1 for task in pool if task["subset"] == "tune"),
        "per_family": families,
        "test_sent_to_teacher": False,
    }


def write_dataset(output: Path, state: dict, stats: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, key in (("train.jsonl", "train_rows"), ("tune.jsonl", "tune_rows")):
        text = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in state[key])
        (output / name).write_text(text, encoding="utf-8")
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (output / "checkpoint.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "built_from": "teacher trajectories on train and tune tasks; the frozen test was a local blocklist and was not sent",
        "test_sent_to_teacher": False,
        "test_used_as_training_rows": False,
        "n_train_dialogs": len(state["train_rows"]),
        "n_tune_dialogs": len(state["tune_rows"]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def dry_report(kept: list[dict], dropped: list[dict]) -> dict:
    families: dict[str, int] = defaultdict(int)
    for task in kept:
        families[family_of(task)] += 1
    return {
        "dry_run": True,
        "requests": 0,
        "test_sent_to_teacher": False,
        "n_train_tasks": sum(1 for task in kept if task["subset"] == "train"),
        "n_tune_tasks": sum(1 for task in kept if task["subset"] == "tune"),
        "n_refusals": sum(1 for task in kept if task.get("expect_refusal")),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "families": dict(sorted(families.items())),
    }


def _usage(body: dict) -> tuple[int, int, int]:
    usage = body.get("usageMetadata") or {}
    prompt = int(usage.get("promptTokenCount") or 0)
    candidates = int(usage.get("candidatesTokenCount") or 0)
    total = int(usage.get("totalTokenCount") or (prompt + candidates))
    return prompt, candidates, total


def _text_from_body(body: dict) -> str:
    candidates = body.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "\n".join(text for text in texts if text).strip()


def is_quota(status: int, body: dict) -> bool:
    if status == 429:
        return True
    if status not in {400, 403, 503}:
        return False
    text = json.dumps(body).lower()
    return "resource_exhausted" in text or "quota" in text


def http_detail(status: int, body: dict, model: str) -> str:
    error = body.get("error") if isinstance(body, dict) else None
    message = ""
    if isinstance(error, dict):
        message = str(error.get("message") or "")
    elif error:
        message = str(error)
    message = " ".join(message.split())
    if len(message) > 500:
        message = message[:500].rstrip()
    base = f"HTTP {status} from model {model}"
    if message:
        return f"{base}: {message}"
    return base


def _body_from_raw(raw: bytes, status: int) -> dict:
    try:
        body = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        text = raw.decode("utf-8", errors="replace").strip()
        return {"error": {"message": text[:500] or f"non-JSON body, status {status}"}}
    if not isinstance(body, dict):
        return {"error": {"message": f"response status {status} was not an object"}}
    return body


def to_gemini(messages: list[dict]) -> tuple[str, list[dict]]:
    systems: list[str] = []
    contents: list[dict] = []
    for message in messages:
        role = message.get("role")
        text = message.get("content") or ""
        if role == "system":
            systems.append(text)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"][0]["text"] += "\n" + text
        else:
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
    if contents and contents[0]["role"] != "user":
        contents.insert(0, {"role": "user", "parts": [{"text": "Continue."}]})
    return "\n\n".join(systems), contents


def default_transport(url: str, headers: dict, payload: dict, timeout: float) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = int(getattr(response, "status", 200))
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)
    except urllib.error.URLError as exc:
        raise NetworkStop(str(exc.reason)) from exc
    return status, _body_from_raw(raw, status)


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        budget: RequestBudget,
        transport=None,
        temperature: float = 0.4,
        max_output_tokens: int = 512,
        timeout: float = 60.0,
        log=None,
    ):
        if not api_key:
            raise ValueError("missing API key")
        if not all(char.isalnum() or char in "._-" for char in model):
            raise ValueError(f"refusing model name {model!r}")
        self.api_key = api_key
        self.model = model
        self.budget = budget
        self.transport = default_transport if transport is None else transport
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.log = log or (lambda event: print(json.dumps(event), flush=True))

    def generate(self, messages: list[dict]) -> str:
        """Send one prompt. 5xx is retried up to SERVER_ATTEMPTS. Any other error stops."""
        system, contents = to_gemini(messages)
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = GEMINI_URL.format(model=self.model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        last_detail = ""
        for attempt in range(SERVER_ATTEMPTS):
            self.budget.acquire()
            try:
                status, body = self.transport(url, headers, payload, self.timeout)
            except DistillStop:
                self._emit(status=None)
                raise
            prompt, candidates, total = _usage(body)
            self.budget.add_usage(prompt, candidates, total)
            self._emit(status=status)
            if is_quota(status, body):
                detail = http_detail(status, body, self.model)
                self.budget.stop_reason = "quota"
                self.budget.stop_detail = detail
                raise QuotaStop(detail)
            if status >= 500:
                last_detail = (
                    f"{http_detail(status, body, self.model)}. "
                    f"Server error after {attempt + 1} of {SERVER_ATTEMPTS} requests."
                )
                if attempt + 1 < SERVER_ATTEMPTS:
                    continue
                self.budget.stop_reason = "network"
                self.budget.stop_detail = last_detail
                raise NetworkStop(last_detail)
            if status != 200:
                detail = (
                    f"{http_detail(status, body, self.model)}. "
                    "This status is not retried."
                )
                self.budget.stop_reason = "http"
                self.budget.stop_detail = detail
                raise FatalHTTPStop(detail)
            return _text_from_body(body)
        self.budget.stop_reason = "network"
        self.budget.stop_detail = last_detail or "server error"
        raise NetworkStop(self.budget.stop_detail)

    def _emit(self, status: int | None) -> None:
        self.log(
            {
                "event": "request",
                "requests": self.budget.used,
                "max_requests": self.budget.max_requests,
                "prompt_tokens": self.budget.prompt_tokens,
                "candidates_tokens": self.budget.candidates_tokens,
                "total_tokens": self.budget.total_tokens,
                "status": status,
            }
        )


class TeacherBackend:
    def __init__(self, client: GeminiClient):
        self.client = client

    def complete(self, *, role: str, messages: list[dict]) -> str:
        del role
        return self.client.generate(messages)


def preflight(client: GeminiClient) -> None:
    """One request with no task text. A dead model or key fails before the loop."""
    client.generate([{"role": "user", "content": PREFLIGHT_PROMPT}])


def report_stop(reason: str | None, detail: str, requests: int) -> None:
    print(
        json.dumps({"event": "stop", "reason": reason, "detail": detail, "requests": requests}),
        flush=True,
    )
    if reason == "http":
        print(
            f"Stopped: {detail} No further requests will be sent.",
            file=sys.stderr,
        )
    elif reason in {"network", "quota"} and detail:
        print(f"Stopped: {detail}", file=sys.stderr)


def stop_exit_code(reason: str | None) -> int | None:
    if reason == "quota":
        return 2
    if reason in {"http", "network", "leakage"}:
        return 1
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill Orbit trajectories from a Gemini teacher")
    parser.add_argument("--model", default=os.environ.get("ORBIT_TEACHER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--requests-per-minute", type=float, default=DEFAULT_REQUESTS_PER_MINUTE)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key-env", default="ORBIT_TEACHER_API_KEY")
    return parser


def _budget_from_state(state: dict, args, clock=None, sleep=None) -> RequestBudget:
    return RequestBudget(
        args.max_requests,
        args.requests_per_minute,
        clock=clock,
        sleep=sleep,
        used=int(state.get("requests") or 0),
        last_request_at=state.get("last_request_at"),
    )


def main(argv: list[str] | None = None, *, transport=None, clock=None, sleep=None) -> dict:
    args = build_parser().parse_args(argv)
    if args.max_requests < 1:
        raise SystemExit("--max-requests must be at least 1")
    if args.requests_per_minute <= 0:
        raise SystemExit("--requests-per-minute must be positive")
    if args.attempts < 1:
        raise SystemExit("--attempts must be at least 1")
    scratch = Path(tempfile.mkdtemp(prefix="orbit-distill-"))
    try:
        pool = load_pool(scratch / "pool")
        test_rows = _read(TEST)
        kept, dropped = select_tasks(pool, test_rows)
        if args.dry_run:
            report = dry_report(kept, dropped)
            print(json.dumps(report, indent=2))
            return report
        key = os.environ.get(args.api_key_env, "").strip()
        if not key:
            raise SystemExit(
                f"Set {args.api_key_env} to a Gemini API key. "
                "The script never reads a key from disk. --dry-run does not need a key."
            )
        output = args.output
        checkpoint_path = args.checkpoint or (output / "checkpoint.json")
        state = load_checkpoint(checkpoint_path)
        budget = _budget_from_state(state, args, clock=clock, sleep=sleep)
        budget.prompt_tokens = int(state.get("prompt_tokens") or 0)
        budget.candidates_tokens = int(state.get("candidates_tokens") or 0)
        budget.total_tokens = int(state.get("total_tokens") or 0)
        print(
            json.dumps(
                {
                    "event": "start",
                    "model": args.model,
                    "max_requests": args.max_requests,
                    "requests_per_minute": args.requests_per_minute,
                    "requests_already_used": budget.used,
                    "attempts_per_task": args.attempts,
                    "n_tasks": len(kept),
                    "note": (
                        "A preflight request runs first. HTTP 429 and quota errors stop the run "
                        "and are not retried. Any other 4xx stops the run on that response. "
                        "A 5xx response is retried a few times, then the run stops. "
                        "The cap is cumulative across resumes. "
                        "A key on a billed Google project can still incur charges up to this cap; "
                        "confirm billing is off before starting."
                    ),
                }
            ),
            flush=True,
        )
        client = GeminiClient(
            api_key=key,
            model=args.model,
            budget=budget,
            transport=transport,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )

        def backend_for(_task):
            return TeacherBackend(client)

        try:
            preflight(client)
            state = run_distill(
                kept,
                test_rows,
                backend_for=backend_for,
                attempts=args.attempts,
                workspace=scratch / "runs",
                state=state,
                budget=budget,
            )
        except DistillStop as exc:
            state["stopped_reason"] = exc.reason
            state["stopped_detail"] = exc.detail
            sync_budget(state, budget)
            stats = build_stats(
                state,
                kept,
                model=args.model,
                max_requests=args.max_requests,
                per_minute=args.requests_per_minute,
                attempts=args.attempts,
            )
            stats["n_dropped_leakage"] = len(dropped)
            write_dataset(output, state, stats)
            report_stop(exc.reason, exc.detail, budget.used)
            code = stop_exit_code(exc.reason)
            if code is not None:
                raise SystemExit(code) from exc
            return stats
        stats = build_stats(
            state,
            kept,
            model=args.model,
            max_requests=args.max_requests,
            per_minute=args.requests_per_minute,
            attempts=args.attempts,
        )
        stats["n_dropped_leakage"] = len(dropped)
        write_dataset(output, state, stats)
        reason = state.get("stopped_reason")
        report_stop(reason, state.get("stopped_detail") or "", state["requests"])
        code = stop_exit_code(reason)
        if code is not None:
            raise SystemExit(code)
        return stats
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(exc, file=sys.stderr)
        raise
