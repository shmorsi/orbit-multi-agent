"""Fictional Northline boardroom. Not a scored eval.

The auditor checks every claim against the files. A figure the CSV does not
contain is refused and left out of the decision memo.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from orbit_agents.engine import RunResult
from orbit_agents.paths import ROOT

SCENARIO = ROOT / "data" / "scenarios" / "boardroom"
CORPUS = SCENARIO / "corpus"
TASKS = SCENARIO / "tasks.jsonl"
# Planted CFO miss. The auditor compares the claim with the CSV, not with this delta.
REVENUE_DELTA = 440000

FLIGHT_RULES_BOARD = {
    "team": "boardroom",
    "roles": ["ceo", "cfo", "cto", "coo", "auditor"],
    "ceo": "Frames the decision. Does not invent a figure.",
    "cfo": "Cites financials.csv. A figure the file does not contain is refused.",
    "cto": "Cites product_metrics.csv.",
    "coo": "Cites the launch memo.",
    "auditor": "Code. Keeps a claim only when the cited file states it.",
    "max_turns": 8,
    "max_critic_retries": 2,
    "max_schema_tries": 2,
    "replay_clock_step_seconds": 4,
}


def boardroom_tasks() -> list[dict]:
    return [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_by_id(task_id: str) -> dict:
    for task in boardroom_tasks():
        if task["id"] == task_id:
            return task
    raise KeyError(task_id)


def _read(relative: str) -> str:
    return (CORPUS / relative).read_text(encoding="utf-8")


def financials() -> dict[str, dict[str, int]]:
    """Quarter rows from financials.csv."""
    rows = {}
    for row in csv.DictReader(_read("financials.csv").splitlines()):
        rows[row["quarter"]] = {
            "revenue_usd": int(row["revenue_usd"]),
            "operating_cost_usd": int(row["operating_cost_usd"]),
        }
    return rows


def q4_revenue() -> int:
    return financials()["2025Q4"]["revenue_usd"]


def q4_operating_cost() -> int:
    return financials()["2025Q4"]["operating_cost_usd"]


def product_metrics() -> dict[str, int]:
    rows = {}
    for row in csv.DictReader(_read("product_metrics.csv").splitlines()):
        rows[row["metric"]] = int(row["value"])
    return rows


def activation_rate() -> int:
    return product_metrics()["activation_rate_pct"]


def launch_crew() -> int:
    match = re.search(r"launch crew is (\d+)", _read("memos/launch.md"))
    if not match:
        raise RuntimeError("launch memo has no crew count")
    return int(match.group(1))


def freeze_date() -> str:
    match = re.search(r"freeze date is (\S+)", _read("memos/launch.md"))
    if not match:
        raise RuntimeError("launch memo has no freeze date")
    return match.group(1).rstrip(".")


def check_claim(claim: dict) -> dict:
    """One claim against the cited file. ok is false when the file disagrees."""
    source = claim.get("source")
    name = claim.get("name")
    claimed = claim.get("value")
    actual = None
    ok = False
    if source == "financials.csv":
        quarter = financials().get(claim.get("quarter"))
        if quarter and claim.get("column") in quarter:
            actual = quarter[claim["column"]]
            ok = actual == claimed
    elif source == "product_metrics.csv":
        metrics = product_metrics()
        key = claim.get("metric")
        if key in metrics:
            actual = metrics[key]
            ok = actual == claimed
    elif source == "memos/launch.md":
        if name == "launch_crew":
            actual = launch_crew()
            ok = actual == claimed
        elif name == "freeze_date":
            actual = freeze_date()
            ok = actual == claimed
    return {"claim": name, "ok": ok, "claimed": claimed, "actual": actual, "source": source}


def decision_memo(findings: list[dict]) -> str:
    """Supported findings only. A refused claim is named and not given a number."""
    lines = [
        "Northline launch memo.",
        "Hold the launch until the freeze date in the launch memo.",
    ]
    for finding in findings:
        if not finding["ok"]:
            continue
        lines.append(f"{finding['claim']} is {finding['actual']} in {finding['source']}.")
    refused = [finding["claim"] for finding in findings if not finding["ok"]]
    if refused:
        names = ", ".join(name for name in refused if name)
        lines.append(f"The auditor refused {names}. The cited file does not support that figure.")
    return " ".join(lines)


def corpus_manifest() -> dict:
    """Hashes of the fictional corpus. The demo is not a score."""
    files = {}
    for path in sorted(CORPUS.rglob("*")):
        if not path.is_file():
            continue
        body = path.read_bytes()
        rel = path.relative_to(CORPUS).as_posix()
        files[rel] = {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    return {"corpus": "corpus", "fictional": True, "files": files}


class BoardRun:
    """CEO, CFO, CTO, COO, then the auditor. The auditor does not call the policy."""

    def __init__(self, task: dict, policy, on_event=None):
        self.task = task
        self.policy = policy
        self.on_event = on_event
        self.events: list[dict] = []
        self.n_steps = 0
        self.schema_valid = 0
        self.schema_invalid = 0

    def _event(self, agent: str, kind: str, **payload) -> None:
        event = {"index": len(self.events), "agent": agent, "kind": kind, **payload}
        event["t_sec"] = event["index"] * FLIGHT_RULES_BOARD["replay_clock_step_seconds"]
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def _seat(self, role: str) -> dict:
        self.n_steps += 1
        raw = self.policy.complete_text(
            [
                {"role": "system", "content": f"You are the {role}. Reply with one JSON object."},
                {"role": "user", "content": self.task["question"]},
            ],
            {},
            role=role,
        )
        self._event(role, "message", step=self.n_steps, content=raw)
        from orbit_agents.engine import extract_json

        payload = extract_json(raw)
        self.schema_valid += 1
        self._event(role, "parsed", step=self.n_steps, payload=payload)
        return payload

    def execute(self) -> RunResult:
        self._seat("ceo")
        seats = {role: self._seat(role) for role in ("cfo", "cto", "coo")}
        findings = []
        for role in ("cfo", "cto", "coo"):
            for claim in seats[role].get("claims", []):
                finding = check_claim(claim)
                finding["by"] = role
                findings.append(finding)
        refused = [finding for finding in findings if not finding["ok"]]
        verdict = "refuse" if refused else "pass"
        reason = (
            "The auditor refused a claim the cited file does not support."
            if refused
            else "Every claim matches the cited file."
        )
        self._event("auditor", "audit", verdict=verdict, reason=reason, findings=findings)
        if refused:
            self._event(
                "auditor",
                "gate",
                code_verdict="refuse",
                reason=reason,
                claim={"name": refused[0]["claim"]},
            )
        answer = decision_memo(findings)
        kept = [
            {
                "name": finding["claim"],
                "value": finding["actual"],
                "source": finding["source"],
                "citation": finding["source"],
            }
            for finding in findings
            if finding["ok"]
        ]
        self._event("auditor", "final", status="answer", answer=answer)
        return RunResult(
            task_id=self.task["id"],
            mode="boardroom",
            status="answer",
            answer=answer,
            claims=kept,
            events=self.events,
            examples=[],
            n_steps=self.n_steps,
            schema_valid=self.schema_valid,
            schema_invalid=self.schema_invalid,
            tool_calls=[],
            gate_overrode=bool(refused),
            messages=[],
        )


def run_boardroom(task: dict, policy, workspace: Path | None = None, on_event=None) -> RunResult:
    """Run the scripted or model board. workspace is unused; the corpus is read-only."""
    del workspace
    return BoardRun(task, policy, on_event=on_event).execute()


def dump_board_trace(task: dict, result: RunResult, *, harness: str | None) -> str:
    header = {
        "kind": "mission",
        "source": "orbit_agents.boardroom",
        "fabricated": False,
        "task_id": result.task_id,
        "mode": result.mode,
        "question": task["question"],
        "status": result.status,
        "answer": result.answer,
        "claims": result.claims,
        "n_steps": result.n_steps,
        "schema_valid": result.schema_valid,
        "schema_invalid": result.schema_invalid,
        "n_tool_calls": 0,
        "n_retries": 0,
        "gate_overrode": result.gate_overrode,
        "flight_rules": FLIGHT_RULES_BOARD,
        "team": "boardroom",
        "scenario": "boardroom",
    }
    if harness:
        header["harness"] = harness
        header["scenario_note"] = "Scripted demo on a fictional company corpus. Not a model score."
    rows = [header, *result.events]
    return "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
