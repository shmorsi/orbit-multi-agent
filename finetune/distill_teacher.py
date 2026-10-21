"""Distill planner, worker, and critic trajectories from a Claude or Gemini teacher.

The teacher sees train and tune tasks only. The frozen test file is a local
blocklist: its questions are never placed in a request body and never written
as training rows. Tool calls run through Orbit's sandbox. Both providers write
the same JSONL files. Each kept dialog records the teacher model.

The default teacher is Anthropic claude-haiku-5-5. A model name that starts
with claude selects that provider. The key is ORBIT_ANTHROPIC_API_KEY. The
pace is about 20 requests per minute. Anthropic requests omit temperature
unless --temperature is set, because newer models reject that field. Replies
are read from text blocks only. A price table estimates USD from input
and output tokens; an unknown model uses a high fallback price. The run stops
at 5 USD, and on a credit-balance or other billing error. HTTP 429 and server
errors, including 529, wait 20s, then 40s, then 80s.

Gemini stays available with --provider gemini and ORBIT_TEACHER_API_KEY.
That path keeps the free-tier pace of about 4 requests per minute and does
not apply the dollar limit. Gemma model names fold the system text into the
first user turn. A per-minute HTTP 429 with a stated delay of 120s or less
waits once; a per-day quota or a second consecutive 429 stops.

Shared controls:

- hard cap on API requests for this run (800). The checkpoint total does not block the next run unless --max-total-requests is set
- one preflight request, then the task loop
- any 4xx other than 429 stops on that response
- HTTP 503 waits 20s, then 40s, then 80s; each retry counts toward the cap
- another 5xx is retried a few times, then the process stops

This script does not call the API during --dry-run.

    python finetune/distill_teacher.py --dry-run
    ORBIT_ANTHROPIC_API_KEY=... python finetune/distill_teacher.py \\
        --provider anthropic --model claude-haiku-5-5 \\
        --max-requests 800 --requests-per-minute 20 --max-cost-usd 5 \\
        --output finetune/data/distill
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
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
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL
DEFAULT_MAX_REQUESTS = 800
DEFAULT_REQUESTS_PER_MINUTE = 4.0
ANTHROPIC_REQUESTS_PER_MINUTE = 20.0
DEFAULT_MAX_COST_USD = 5.0
GEMINI_TEMPERATURE = 0.4
DEFAULT_ATTEMPTS = 3
SERVER_ATTEMPTS = 3
SERVER_BACKOFF_SECONDS = (20, 40, 80)
HIGH_DEMAND_STATUS = 503
MINUTE_QUOTA_MAX_WAIT_SECONDS = 120
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
LONG_CONTEXT_TOKENS = 100_000
# USD per million tokens. Base input/output from the published Claude API list.
# Haiku 5.5 uses the higher tier once the prompt is past LONG_CONTEXT_TOKENS.
# An unknown model uses FALLBACK_PRICE, which is the retired Opus 4.1 list price,
# above every current model, so a cost cap cannot be under-counted.
ANTHROPIC_PRICES = {
    "claude-haiku-5-5": {
        "input": 0.1,
        "output": 0.5,
        "input_over_long_context": 0.5,
        "output_over_long_context": 2.5,
    },
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0},
    "claude-opus-5": {"input": 5.0, "output": 25.0},
}
FALLBACK_PRICE = {"input": 15.0, "output": 75.0}
PREFLIGHT_PROMPT = "Reply with an empty JSON object."
_RETRY_IN_RE = re.compile(r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s\b", re.IGNORECASE)
_DELAY_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)s$", re.IGNORECASE)
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


class BillingStop(DistillStop):
    def __init__(self, detail: str):
        super().__init__("billing", detail)


class CostStop(DistillStop):
    def __init__(self, detail: str):
        super().__init__("cost", detail)


class RequestBudget:
    """Counts every outbound call and spaces them out.

    max_requests is this process only. used is the checkpoint total.
    max_total_requests, when set, stops the run once that total is reached.
    """

    def __init__(
        self,
        max_requests: int,
        per_minute: float,
        *,
        clock=None,
        sleep=None,
        used: int = 0,
        last_request_at: float | None = None,
        max_total_requests: int | None = None,
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
        if max_total_requests is not None and int(max_total_requests) < 1:
            raise ValueError("max_total_requests must be at least 1")
        self.max_total_requests = None if max_total_requests is None else int(max_total_requests)
        self.last_request_at = last_request_at
        self.prompt_tokens = 0
        self.candidates_tokens = 0
        self.total_tokens = 0
        self.stop_reason: str | None = None
        self.stop_detail = ""
        self.status_counts: dict[int, int] = defaultdict(int)
        self.started_at: float | None = None
        self.session_used = 0
        self.consecutive_429 = 0
        self.estimated_cost_usd = 0.0
        self.price_source = ""
        self.track_cost = False

    def acquire(self) -> None:
        if self.stop_reason == "quota":
            raise QuotaStop(self.stop_detail or "stopped after a quota response")
        if self.stop_reason == "network":
            raise NetworkStop(self.stop_detail or "stopped after a network failure")
        if self.stop_reason == "http":
            raise FatalHTTPStop(self.stop_detail or "stopped after an HTTP client error")
        if self.stop_reason == "billing":
            raise BillingStop(self.stop_detail or "stopped after a billing error")
        if self.stop_reason == "cost":
            raise CostStop(self.stop_detail or "stopped after the cost limit")
        if self.stop_reason == "request_cap" or self.session_used >= self.max_requests:
            self.stop_reason = "request_cap"
            raise BudgetStop()
        if self.max_total_requests is not None and self.used >= self.max_total_requests:
            self.stop_reason = "request_cap"
            raise BudgetStop()
        now = float(self.clock())
        if self.last_request_at is not None:
            wait = self.min_interval - (now - self.last_request_at)
            if wait > 0:
                self.sleep(wait)
                now = float(self.clock())
        self.last_request_at = now
        if self.started_at is None:
            self.started_at = now
        self.used += 1
        self.session_used += 1

    def note_status(self, status: int) -> None:
        self.status_counts[int(status)] = self.status_counts.get(int(status), 0) + 1

    def effective_per_minute(self) -> float | None:
        if self.started_at is None or self.session_used <= 0:
            return None
        elapsed = float(self.clock()) - self.started_at
        if elapsed <= 0:
            return None
        return self.session_used / (elapsed / 60.0)

    def status_count_map(self) -> dict[str, int]:
        return {str(code): self.status_counts[code] for code in sorted(self.status_counts)}

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


def narrow_tasks(tasks: list[dict], *, families: str | None, limit: int | None) -> list[dict]:
    """Keep the named families, round-robin them, then take the first limit tasks."""
    chosen = list(tasks)
    if families:
        names = [part.strip() for part in families.split(",") if part.strip()]
        known = {family_of(task) for task in chosen}
        unknown = [name for name in names if name not in known]
        if unknown:
            raise SystemExit("Unknown family: " + ", ".join(unknown))
        wanted = set(names)
        chosen = [task for task in chosen if family_of(task) in wanted]
    chosen = interleave(chosen)
    if limit is not None:
        if limit < 1:
            raise SystemExit("--limit-tasks must be at least 1")
        chosen = chosen[:limit]
    return chosen


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
    """Keep a run when the checker passes and every saved turn is schema-valid.

    A repaired call increments schema_invalid, but that call is not saved.
    The checker already accepted the extracted final, so the repair is not a reject.
    """
    scored = score_task(task, result)
    if not scored["success"]:
        return False, "checker"
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


def rows_from_result(task: dict, result, *, teacher_model: str = "") -> list[dict]:
    rows = []
    for example in result.examples:
        if example["role"] not in {"planner", "worker", "critic"}:
            continue
        rows.append(
            {
                "id": task["id"],
                "role": example["role"],
                "messages": copy.deepcopy(example["messages"]),
                "metadata": {"teacher_model": teacher_model},
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
    state["status_counts"] = budget.status_count_map()
    state["estimated_cost_usd"] = budget.estimated_cost_usd
    state["price_source"] = budget.price_source


def write_attempt_debug(debug_dir: Path, task: dict, result, *, attempt: int, reason: str) -> None:
    """Save the transcript the tempfile run directory used to discard."""
    folder = debug_dir / str(task["id"])
    folder.mkdir(parents=True, exist_ok=True)
    scored = score_task(task, result)
    payload = {
        "id": task["id"],
        "family": family_of(task),
        "question": task["question"],
        "attempt": attempt,
        "reject_reason": None if reason == "ok" else reason,
        "extracted": {
            "status": result.status,
            "answer": result.answer,
            "claims": result.claims,
        },
        "expected": {
            "expect_refusal": bool(task.get("expect_refusal")),
            "status": task.get("status"),
            "answer": task.get("answer"),
            "gold_claims": task.get("gold_claims") or [],
        },
        "checker": scored,
        "events": result.events,
        "examples": result.examples,
        "tool_calls": result.tool_calls,
    }
    (folder / f"attempt-{attempt}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def run_distill(
    tasks: list[dict],
    test_rows: list[dict],
    *,
    backend_for,
    attempts: int,
    workspace: Path,
    state: dict | None = None,
    budget: RequestBudget | None = None,
    teacher_model: str = "",
    debug_dir: Path | None = None,
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
            except (QuotaStop, BudgetStop, NetworkStop, FatalHTTPStop, BillingStop, CostStop) as exc:
                sync_budget(state, budget)
                state["stopped_reason"] = exc.reason
                state["stopped_detail"] = exc.detail
                state["tasks"][task["id"]] = record
                return state
            record["attempts"] += 1
            state["attempts"] += 1
            ok, why = trajectory_ok(task, result)
            if ok:
                rows = rows_from_result(task, result, teacher_model=teacher_model)
                leak = overlap_error(rows, [], test_rows)
                if leak:
                    ok = False
                    why = "leakage"
                else:
                    bucket = "train_rows" if task["subset"] == "train" else "tune_rows"
                    state[bucket].extend(rows)
                    record["accepted"] = True
            record["last_reject"] = None if ok else why
            if debug_dir is not None:
                write_attempt_debug(debug_dir, task, result, attempt=record["attempts"], reason=why)
            sync_budget(state, budget)
        state["tasks"][task["id"]] = record
    sync_budget(state, budget)
    if state.get("stopped_reason") is None:
        state["stopped_reason"] = None
    return state


def build_stats(
    state: dict,
    pool: list[dict],
    *,
    model: str,
    max_requests: int,
    per_minute: float,
    attempts: int,
    provider: str = "gemini",
) -> dict:
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
        "teacher_model": model,
        "provider": provider,
        "max_requests": max_requests,
        "requests_per_minute": per_minute,
        "attempts_per_task": attempts,
        "requests": state["requests"],
        "prompt_tokens": state["prompt_tokens"],
        "candidates_tokens": state["candidates_tokens"],
        "total_tokens": state["total_tokens"],
        "input_tokens": state["prompt_tokens"],
        "output_tokens": state["candidates_tokens"],
        "estimated_cost_usd": state.get("estimated_cost_usd") or 0,
        "price_source": state.get("price_source") or "",
        "status_counts": state.get("status_counts") or {},
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
    """A quota-shaped body on 400 or 403. 429 and 503 are handled on their own."""
    if status not in {400, 403}:
        return False
    text = json.dumps(body).lower()
    return "resource_exhausted" in text or "quota" in text


def _parse_delay(text: str) -> float | None:
    match = _DELAY_RE.match(text.strip())
    if not match:
        return None
    return float(match.group(1))


def minute_quota_wait(body: dict) -> float | None:
    """Seconds to wait for one per-minute 429. None means stop instead of waiting.

    A per-day quota, a missing delay, and a delay above the pause limit all return None.
    """
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        error = {}
    text = json.dumps(body).lower()
    if "per day" in text or "daily" in text:
        return None
    delay = None
    for item in error.get("details") or []:
        if isinstance(item, dict) and item.get("retryDelay"):
            delay = _parse_delay(str(item["retryDelay"]))
            if delay is not None:
                break
    if delay is None:
        match = _RETRY_IN_RE.search(str(error.get("message") or ""))
        if match:
            delay = float(match.group(1))
    if delay is None or delay <= 0 or delay > MINUTE_QUOTA_MAX_WAIT_SECONDS:
        return None
    if "per minute" not in text and "retry in" not in text and "retrydelay" not in text:
        return None
    return delay


def _price_matches(model: str, key: str) -> bool:
    return model == key or model.startswith(key + "-") or model.startswith(key + "@")


def price_for(model: str, input_tokens: int) -> tuple[dict, str]:
    """Return {input, output} USD per million tokens and 'table' or 'fallback'."""
    chosen = None
    for key in sorted(ANTHROPIC_PRICES, key=len, reverse=True):
        if _price_matches(model, key):
            chosen = ANTHROPIC_PRICES[key]
            break
    if chosen is None:
        return dict(FALLBACK_PRICE), "fallback"
    if input_tokens > LONG_CONTEXT_TOKENS and "input_over_long_context" in chosen:
        return {
            "input": chosen["input_over_long_context"],
            "output": chosen["output_over_long_context"],
        }, "table"
    return {"input": chosen["input"], "output": chosen["output"]}, "table"


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> tuple[float, str]:
    price, source = price_for(model, input_tokens)
    cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    return cost, source


def is_billing(body: dict) -> bool:
    text = json.dumps(body).lower()
    return "credit balance is too low" in text or "billing" in text or "purchase credits" in text


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


def model_folds_system(model: str) -> bool:
    """Gemma rejects systemInstruction on this API. Gemini keeps that field."""
    return model.lower().startswith("gemma")


def _fold_system(payload: dict, system: str) -> None:
    contents = payload.get("contents") or []
    if not contents:
        contents = [{"role": "user", "parts": [{"text": system}]}]
        payload["contents"] = contents
        return
    part = contents[0]["parts"][0]
    part["text"] = system + "\n\n" + part["text"]


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
        temperature: float = GEMINI_TEMPERATURE,
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
        """Send one prompt. 503 backs off. A short per-minute 429 waits once."""
        system, contents = to_gemini(messages)
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        if system and not model_folds_system(self.model):
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        elif system:
            _fold_system(payload, system)
        url = GEMINI_URL.format(model=self.model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        server_tries = 0
        demand_tries = 0
        while True:
            self.budget.acquire()
            try:
                status, body = self.transport(url, headers, payload, self.timeout)
            except DistillStop:
                self._emit(status=None)
                raise
            prompt, candidates, total = _usage(body)
            self.budget.add_usage(prompt, candidates, total)
            self.budget.note_status(status)
            self._emit(status=status)
            if status != 429:
                self.budget.consecutive_429 = 0
            if status == HIGH_DEMAND_STATUS:
                if demand_tries >= len(SERVER_BACKOFF_SECONDS):
                    waited = ", ".join(str(item) for item in SERVER_BACKOFF_SECONDS)
                    self._stop_network(
                        f"{http_detail(status, body, self.model)}. "
                        f"High demand after {demand_tries + 1} requests; waited {waited} seconds.",
                        demand_tries,
                    )
                delay = SERVER_BACKOFF_SECONDS[demand_tries]
                demand_tries += 1
                self._wait(delay, status)
                continue
            if status == 429:
                delay = minute_quota_wait(body)
                if delay is not None and self.budget.consecutive_429 == 0:
                    self.budget.consecutive_429 = 1
                    self._wait(delay, status)
                    continue
                self._stop_quota(self._quota_detail(status, body))
            if is_quota(status, body):
                self._stop_quota(http_detail(status, body, self.model))
            if status >= 500:
                server_tries += 1
                detail = (
                    f"{http_detail(status, body, self.model)}. "
                    f"Server error after {server_tries} of {SERVER_ATTEMPTS} requests."
                )
                if server_tries < SERVER_ATTEMPTS:
                    continue
                self._stop_network(detail, server_tries)
            if status != 200:
                detail = (
                    f"{http_detail(status, body, self.model)}. "
                    "This status is not retried."
                )
                self.budget.stop_reason = "http"
                self.budget.stop_detail = detail
                raise FatalHTTPStop(detail)
            return _text_from_body(body)

    def _quota_detail(self, status: int, body: dict) -> str:
        detail = http_detail(status, body, self.model)
        text = json.dumps(body).lower()
        if self.budget.consecutive_429:
            return detail + " A second consecutive 429 stopped the run."
        if "per day" in text or "daily" in text:
            return detail + " A per-day quota stopped the run."
        return (
            detail
            + " The stated retry delay was missing or longer than the pause limit, so the run stopped."
        )

    def _stop_quota(self, detail: str) -> None:
        self.budget.stop_reason = "quota"
        self.budget.stop_detail = detail
        raise QuotaStop(detail)

    def _stop_network(self, detail: str, tries: int) -> None:
        del tries
        self.budget.stop_reason = "network"
        self.budget.stop_detail = detail
        raise NetworkStop(detail)

    def _wait(self, seconds: float, status: int) -> None:
        self.log(
            {
                "event": "wait",
                "status": status,
                "sleep_sec": seconds,
                "requests": self.budget.used,
                "configured_requests_per_minute": self.budget.per_minute,
                "effective_requests_per_minute": self.budget.effective_per_minute(),
                "status_counts": self.budget.status_count_map(),
            }
        )
        self.budget.sleep(seconds)

    def _emit(self, status: int | None) -> None:
        self.log(
            {
                "event": "request",
                "requests": self.budget.used,
                "max_requests": self.budget.max_requests,
                "prompt_tokens": self.budget.prompt_tokens,
                "candidates_tokens": self.budget.candidates_tokens,
                "total_tokens": self.budget.total_tokens,
                "input_tokens": self.budget.prompt_tokens,
                "output_tokens": self.budget.candidates_tokens,
                "estimated_cost_usd": self.budget.estimated_cost_usd if self.budget.track_cost else None,
                "status": status,
                "configured_requests_per_minute": self.budget.per_minute,
                "effective_requests_per_minute": self.budget.effective_per_minute(),
                "status_counts": self.budget.status_count_map(),
            }
        )


def _as_token_count(value) -> int:
    """Coerce a usage field to a token count. Nested details are not counts."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        return int(value) if value > 0 else 0
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return 0
        return int(parsed) if parsed > 0 else 0
    return 0


def _anthropic_usage(body: dict) -> tuple[int, int, int]:
    """Billed input and output tokens.

    Newer models add output_tokens_details.thinking_tokens, cache counters,
    and other usage fields. output_tokens already includes thinking tokens,
    so those details are not added again. This client does not send
    cache_control, so cache fields are left out of the estimate.
    """
    if not isinstance(body, dict):
        return 0, 0, 0
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt = _as_token_count(usage.get("input_tokens"))
    output = _as_token_count(usage.get("output_tokens"))
    return prompt, output, prompt + output


def _text_from_anthropic(body: dict) -> str:
    """Join type=text blocks. Thinking and any other block type are ignored."""
    if not isinstance(body, dict):
        return ""
    content = body.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts)


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    system, contents = to_gemini(messages)
    converted: list[dict] = []
    for item in contents:
        role = "assistant" if item["role"] == "model" else "user"
        text = item["parts"][0]["text"]
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"] += "\n" + text
        else:
            converted.append({"role": role, "content": text})
    return system, converted


class AnthropicClient:
    """Claude Messages API. 429, 529, and other 5xx back off. Billing stops the run."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        budget: RequestBudget,
        max_cost_usd: float,
        transport=None,
        temperature: float | None = None,
        max_output_tokens: int = 512,
        timeout: float = 60.0,
        log=None,
    ):
        if not api_key:
            raise ValueError("missing API key")
        if not all(char.isalnum() or char in "._-" for char in model):
            raise ValueError(f"refusing model name {model!r}")
        if max_cost_usd < 0:
            raise ValueError("max_cost_usd must be zero or more")
        self.api_key = api_key
        self.model = model
        self.budget = budget
        self.max_cost_usd = float(max_cost_usd)
        self.budget.track_cost = True
        self.transport = default_transport if transport is None else transport
        self.temperature = None if temperature is None else float(temperature)
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.log = log or (lambda event: print(json.dumps(event), flush=True))

    def generate(self, messages: list[dict]) -> str:
        self._raise_if_cost_reached()
        system, contents = to_anthropic(messages)
        # Newer Claude models reject temperature, top_p, and top_k. Send
        # temperature only when the caller set --temperature. max_tokens stays:
        # the Messages API still requires it, and a live call without sampling
        # fields succeeds.
        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": contents,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if system:
            payload["system"] = system
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        backoff_index = 0
        while True:
            self.budget.acquire()
            try:
                status, body = self.transport(ANTHROPIC_URL, headers, payload, self.timeout)
            except DistillStop:
                self._emit(status=None)
                raise
            prompt, output, total = _anthropic_usage(body)
            added, source = estimate_cost_usd(self.model, prompt, output)
            self.budget.add_usage(prompt, output, total)
            self.budget.estimated_cost_usd += added
            if source == "fallback":
                self.budget.price_source = "fallback"
            elif self.budget.price_source != "fallback":
                self.budget.price_source = source
            self.budget.note_status(status)
            self._emit(status=status)
            # A finished reply can mention billing inside a thinking block.
            # Only an error response is a billing stop.
            if status != 200 and is_billing(body):
                detail = (
                    f"{http_detail(status, body, self.model)}. "
                    "A billing error stopped the run. The checkpoint is saved."
                )
                self.budget.stop_reason = "billing"
                self.budget.stop_detail = detail
                raise BillingStop(detail)
            if status == 429 or status >= 500:
                if backoff_index >= len(SERVER_BACKOFF_SECONDS):
                    detail = (
                        f"{http_detail(status, body, self.model)}. "
                        "Backoff is exhausted, so the run stopped."
                    )
                    self.budget.stop_reason = "network"
                    self.budget.stop_detail = detail
                    raise NetworkStop(detail)
                delay = SERVER_BACKOFF_SECONDS[backoff_index]
                backoff_index += 1
                self._wait(delay, status)
                continue
            if status != 200:
                detail = (
                    f"{http_detail(status, body, self.model)}. "
                    "This status is not retried."
                )
                self.budget.stop_reason = "http"
                self.budget.stop_detail = detail
                raise FatalHTTPStop(detail)
            self._raise_if_cost_reached()
            return _text_from_anthropic(body)

    def _raise_if_cost_reached(self) -> None:
        if self.budget.estimated_cost_usd < self.max_cost_usd:
            return
        if self.budget.estimated_cost_usd == 0 and self.max_cost_usd == 0:
            detail = "The cost limit is 0 USD, so no request was sent."
        else:
            detail = (
                f"Estimated cost {self.budget.estimated_cost_usd:.6f} USD reached "
                f"the {self.max_cost_usd:g} USD limit. The run stopped and the checkpoint is saved."
            )
        self.budget.stop_reason = "cost"
        self.budget.stop_detail = detail
        raise CostStop(detail)

    def _wait(self, seconds: float, status: int) -> None:
        self.log(
            {
                "event": "wait",
                "status": status,
                "sleep_sec": seconds,
                "requests": self.budget.used,
                "estimated_cost_usd": self.budget.estimated_cost_usd,
                "input_tokens": self.budget.prompt_tokens,
                "output_tokens": self.budget.candidates_tokens,
            }
        )
        self.budget.sleep(seconds)

    def _emit(self, status: int | None) -> None:
        self.log(
            {
                "event": "request",
                "requests": self.budget.used,
                "max_requests": self.budget.max_requests,
                "prompt_tokens": self.budget.prompt_tokens,
                "candidates_tokens": self.budget.candidates_tokens,
                "total_tokens": self.budget.total_tokens,
                "input_tokens": self.budget.prompt_tokens,
                "output_tokens": self.budget.candidates_tokens,
                "estimated_cost_usd": self.budget.estimated_cost_usd,
                "price_source": self.budget.price_source,
                "status": status,
                "configured_requests_per_minute": self.budget.per_minute,
                "effective_requests_per_minute": self.budget.effective_per_minute(),
                "status_counts": self.budget.status_count_map(),
            }
        )


_CONTRACT_MARK = "Checker contract:"


def teacher_contract(task: dict) -> str:
    """The final-answer and tool-call JSON the checker scores. Values stay hidden."""
    lines = [
        _CONTRACT_MARK + " reply with one JSON object and no prose.",
        'Planner: {"steps": [{"goal": "...", "tool": "python|read_file|write_file|calculator|retrieve|web_search"}]} or {"steps": []} when the corpus has no record.',
        'Tool call: {"action": "tool", "tool": "<name>", "arguments": { ... }}.',
        'calculator arguments are {"expression": "integers with + - * //"}.',
        'read_file arguments are {"path": "<corpus path>"}.',
        'write_file arguments are {"path": "<workspace file>", "content": "<text>"}.',
        'python arguments are {"code": "<code>"}. The python tool keeps only printed text. Read a table with table("jobs") or table("shifts") and a bundled file with text("path"). print the computed value. Do not paste file text into code. Use table, text, print, len, sum, min, max, and int. Do not import.',
        'retrieve and web_search arguments are {"query": "..."}.',
        'Final answer: {"action": "final", "status": "answer", "answer": "<name> is <value>.", "claims": [{"name": "<name>", "value": 0, "unit": "count", "citation": "call_1"}]}.',
        'A text claim uses "text" instead of "value", and the answer is "<name> is <text>.".',
        "citation call_1 is the first tool result and call_2 is the second, in the order the tools ran.",
        "Every number in the answer must appear in the question or in a tool result, or be the integer a successful python call printed, or be the number of lines in the cited text that contain a token from the question.",
        'Refusal: {"action": "final", "status": "refuse", "answer": "The bundled corpus has no record that answers this.", "claims": []}.',
        'Critic: {"verdict": "accept", "reason": "..."} or {"verdict": "revise", "reason": "..."}. One JSON object only, with no markdown and no surrounding prose.',
    ]
    claims = task.get("gold_claims") or []
    if task.get("expect_refusal") or not claims:
        lines.append("This task is a refusal. status is refuse, claims is empty, and the answer has no digit.")
    else:
        parts = []
        for claim in claims:
            kind = "integer value" if "value" in claim else "text"
            parts.append(f"{claim['name']} ({kind}, {claim['citation']})")
        lines.append(
            "The checker requires these claim names and citations, and it does not reveal the values: "
            + "; ".join(parts)
            + "."
        )
    return " ".join(lines)


def attach_teacher_contract(task: dict, role: str, messages: list[dict]) -> None:
    """Add the checker contract once. Retries reuse the same message list."""
    del role
    if not messages:
        return
    contract = teacher_contract(task)
    if any(_CONTRACT_MARK in (message.get("content") or "") for message in messages):
        return
    if messages[0].get("role") == "system":
        messages[0]["content"] = messages[0]["content"] + "\n\n" + contract
        return
    messages.insert(0, {"role": "system", "content": contract})


class TeacherBackend:
    def __init__(self, client: GeminiClient, task: dict):
        self.client = client
        self.task = task

    def complete(self, *, role: str, messages: list[dict]) -> str:
        attach_teacher_contract(self.task, role, messages)
        return self.client.generate(messages)


def preflight(client) -> None:
    """One request with no task text. A dead model or key fails before the loop."""
    client.generate([{"role": "user", "content": PREFLIGHT_PROMPT}])


def _start_note(provider: str) -> str:
    if provider == "anthropic":
        return (
            "A preflight request runs first. HTTP 429 and server errors wait 20s, then 40s, then 80s. "
            "Any other 4xx stops on that response. A billing error or the cost limit stops the run "
            "and saves the checkpoint. This run stops at its own request cap. "
            "The checkpoint total blocks a later run only when --max-total-requests is set. "
            "The cost estimate stays cumulative."
        )
    return (
        "A preflight request runs first. HTTP 503 waits 20s, then 40s, then 80s. "
        "A per-minute 429 with a stated delay of 120s or less waits once. "
        "A per-day quota or a second consecutive 429 stops the run and saves the checkpoint. "
        "Any other 4xx stops on that response. "
        "This run stops at its own request cap. "
        "The checkpoint total blocks a later run only when --max-total-requests is set. "
        "A key on a billed Google project can still incur charges up to this cap; "
        "confirm billing is off before starting."
    )


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
    elif reason in {"network", "quota", "billing", "cost"} and detail:
        print(f"Stopped: {detail}", file=sys.stderr)


def stop_exit_code(reason: str | None) -> int | None:
    if reason in {"quota", "billing", "cost"}:
        return 2
    if reason in {"http", "network", "leakage"}:
        return 1
    return None


def resolve_runtime(args) -> argparse.Namespace:
    """Fill provider, model, pace, and key env. A claude-* model selects Anthropic."""
    provider = args.provider
    model = args.model or os.environ.get("ORBIT_TEACHER_MODEL") or ""
    if not model:
        model = DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_ANTHROPIC_MODEL
    if provider is None:
        provider = "anthropic" if model.lower().startswith("claude") else "gemini"
    if provider not in {"gemini", "anthropic"}:
        raise SystemExit("--provider must be gemini or anthropic")
    per_minute = args.requests_per_minute
    if per_minute is None:
        per_minute = ANTHROPIC_REQUESTS_PER_MINUTE if provider == "anthropic" else DEFAULT_REQUESTS_PER_MINUTE
    key_env = args.api_key_env
    if not key_env:
        key_env = "ORBIT_ANTHROPIC_API_KEY" if provider == "anthropic" else "ORBIT_TEACHER_API_KEY"
    resolved = argparse.Namespace(**vars(args))
    resolved.provider = provider
    resolved.model = model
    resolved.requests_per_minute = per_minute
    resolved.api_key_env = key_env
    resolved.max_cost_usd = float(args.max_cost_usd)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill Orbit trajectories from a Gemini or Claude teacher")
    parser.add_argument("--provider", choices=("gemini", "anthropic"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS, help="Cap for this run only")
    parser.add_argument(
        "--max-total-requests",
        type=int,
        default=None,
        help="Optional cumulative cap across the checkpoint",
    )
    parser.add_argument("--requests-per-minute", type=float, default=None)
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--families", default=None, help="Comma-separated family names, such as count_jobs,log,refusal")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Write every attempt transcript here")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature. Anthropic omits this field unless it is set. "
            f"Gemini uses {GEMINI_TEMPERATURE:g} when it is omitted."
        ),
    )
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key-env", default=None)
    return parser


def _budget_from_state(state: dict, args, clock=None, sleep=None) -> RequestBudget:
    return RequestBudget(
        args.max_requests,
        args.requests_per_minute,
        clock=clock,
        sleep=sleep,
        used=int(state.get("requests") or 0),
        last_request_at=state.get("last_request_at"),
        max_total_requests=args.max_total_requests,
    )


def main(argv: list[str] | None = None, *, transport=None, clock=None, sleep=None) -> dict:
    args = resolve_runtime(build_parser().parse_args(argv))
    if args.max_requests < 1:
        raise SystemExit("--max-requests must be at least 1")
    if args.requests_per_minute <= 0:
        raise SystemExit("--requests-per-minute must be positive")
    if args.max_cost_usd < 0:
        raise SystemExit("--max-cost-usd must be zero or more")
    if args.attempts < 1:
        raise SystemExit("--attempts must be at least 1")
    if args.max_total_requests is not None and args.max_total_requests < 1:
        raise SystemExit("--max-total-requests must be at least 1")
    scratch = Path(tempfile.mkdtemp(prefix="orbit-distill-"))
    try:
        pool = load_pool(scratch / "pool")
        test_rows = _read(TEST)
        kept, dropped = select_tasks(pool, test_rows)
        kept = narrow_tasks(kept, families=args.families, limit=args.limit_tasks)
        if args.dry_run:
            report = dry_report(kept, dropped)
            report["provider"] = args.provider
            report["model"] = args.model
            report["requests_per_minute"] = args.requests_per_minute
            report["max_cost_usd"] = args.max_cost_usd
            print(json.dumps(report, indent=2))
            return report
        key = os.environ.get(args.api_key_env, "").strip()
        if not key:
            raise SystemExit(
                f"Set {args.api_key_env} to the teacher API key. "
                "The script never reads a key from disk. --dry-run does not need a key."
            )
        output = args.output
        checkpoint_path = args.checkpoint or (output / "checkpoint.json")
        state = load_checkpoint(checkpoint_path)
        budget = _budget_from_state(state, args, clock=clock, sleep=sleep)
        budget.prompt_tokens = int(state.get("prompt_tokens") or 0)
        budget.candidates_tokens = int(state.get("candidates_tokens") or 0)
        budget.total_tokens = int(state.get("total_tokens") or 0)
        budget.estimated_cost_usd = float(state.get("estimated_cost_usd") or 0)
        budget.price_source = str(state.get("price_source") or "")
        for code, count in (state.get("status_counts") or {}).items():
            budget.status_counts[int(code)] = int(count)
        state["stopped_reason"] = None
        state["stopped_detail"] = ""
        print(
            json.dumps(
                {
                    "event": "start",
                    "provider": args.provider,
                    "model": args.model,
                    "max_requests": args.max_requests,
                    "requests_per_minute": args.requests_per_minute,
                    "max_cost_usd": args.max_cost_usd,
                    "estimated_cost_usd": budget.estimated_cost_usd,
                    "max_total_requests": args.max_total_requests,
                    "requests_already_used": budget.used,
                    "attempts_per_task": args.attempts,
                    "n_tasks": len(kept),
                    "note": _start_note(args.provider),
                }
            ),
            flush=True,
        )
        if args.provider == "anthropic":
            client = AnthropicClient(
                api_key=key,
                model=args.model,
                budget=budget,
                max_cost_usd=args.max_cost_usd,
                transport=transport,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )
        else:
            client = GeminiClient(
                api_key=key,
                model=args.model,
                budget=budget,
                transport=transport,
                temperature=GEMINI_TEMPERATURE if args.temperature is None else args.temperature,
                max_output_tokens=args.max_output_tokens,
            )

        def backend_for(task):
            return TeacherBackend(client, task)

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
                teacher_model=args.model,
                debug_dir=args.debug_dir,
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
                provider=args.provider,
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
            provider=args.provider,
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
