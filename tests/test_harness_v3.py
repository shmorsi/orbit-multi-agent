"""Harness v3: the two smoke failures, using train variants only."""

import json
from pathlib import Path

import pytest
from distill_teacher import teacher_contract, trajectory_ok
from variants import iter_specs

from orbit_agents.engine import (
    UNGROUNDED,
    claims_are_grounded,
    contains_number,
    extract_json,
    run_team,
)
from orbit_agents.eval.build_tasks import expand
from orbit_agents.eval.checks import evaluate
from orbit_agents.scripted import ScriptedBackend
from orbit_agents.tasks import task_by_id
from orbit_agents.tools import call_tool, run_python, run_retrieve

COUNT_QUESTION = "From the jobs table, count rows whose machine cell is printer-a."
LOG_QUESTION = "In logs/printer-a.log, how many lines contain the token J02?"

COUNT_CODE = (
    "rows = table('jobs')\n"
    "n = 0\n"
    "for row in rows:\n"
    "    if row['machine'] == 'printer-a':\n"
    "        n += 1\n"
    "print(n)\n"
)

LOG_CODE = (
    "n = 0\n"
    "for line in text('logs/printer-a.log').splitlines():\n"
    "    if 'J02' in line:\n"
    "        n += 1\n"
    "print(n)\n"
)


def _variant(question: str, workspace: Path) -> dict:
    spec = next(row for row in iter_specs() if row["question"] == question)
    spec = dict(spec)
    spec["id"] = "harness-v3"
    spec["subset"] = "train"
    return expand(spec, workspace)


def _final(task: dict) -> str:
    return json.dumps(
        {
            "action": "final",
            "status": "answer",
            "answer": task["answer"],
            "claims": task["gold_claims"],
        }
    )


class _CountWithPython:
    """One python call reads the jobs table. The gold citation is that first call."""

    def __init__(self, task):
        self.task = task
        self.ran = False

    def complete(self, *, role, messages):
        del messages
        if role == "planner":
            return json.dumps({"steps": [{"goal": "Count the rows", "tool": "python"}]})
        if role == "critic":
            return json.dumps({"verdict": "accept", "reason": "The printed count matches the claim."})
        if not self.ran:
            self.ran = True
            return json.dumps({"action": "tool", "tool": "python", "arguments": {"code": COUNT_CODE}})
        return _final(self.task)


class _CountWithText:
    def __init__(self, task):
        self.task = task
        self.ran = False

    def complete(self, *, role, messages):
        del messages
        if role == "planner":
            return json.dumps({"steps": [{"goal": "Count matching lines", "tool": "python"}]})
        if role == "critic":
            return (
                "The count is grounded.\n```json\n"
                '{"verdict": "accept", "reason": "Python printed the line count."}\n'
                "```\n"
            )
        if not self.ran:
            self.ran = True
            return json.dumps({"action": "tool", "tool": "python", "arguments": {"code": LOG_CODE}})
        return _final(self.task)


class _CiteTheLog:
    """The smoke path: read the log, cite it, and keep going when the critic is not JSON."""

    def __init__(self, task):
        self.task = task
        self.read = False
        self.critic_tries = 0

    def complete(self, *, role, messages):
        del messages
        if role == "planner":
            return json.dumps(
                {
                    "steps": [
                        {"goal": "Read the log", "tool": "read_file"},
                        {"goal": "Count the token", "tool": "python"},
                    ]
                }
            )
        if role == "critic":
            self.critic_tries += 1
            if self.critic_tries == 1:
                return ""
            if self.critic_tries == 2:
                return "The count looks right."
            return (
                "Accepting the worker.\n```json\n"
                '{"verdict": "accept", "reason": "The cited file has that many matching lines."}\n'
                "```\n"
            )
        if not self.read:
            self.read = True
            return json.dumps(
                {"action": "tool", "tool": "read_file", "arguments": {"path": "logs/printer-a.log"}}
            )
        return _final(self.task)


def test_search_returns_the_jobs_table_and_still_returns_notes():
    found = run_retrieve("jobs table machine column printer-a")
    assert found["matches"][0]["path"] == "jobs.csv"
    assert "table('jobs')" in found["matches"][0]["text"]
    notes = {
        "who covers the laser": ("docs/shifts.md", "Jules covers the laser."),
        "laser second person in the room": ("docs/laser.md", "second person"),
        "PLA job grams cap": ("docs/printers.md", "50 grams"),
        "failed jobs still count grams": ("docs/printers.md", "Failed jobs still count"),
        "printer-a single job 90 minutes": ("docs/printers.md", "90 minutes"),
    }
    for query, (path, text) in notes.items():
        match = run_retrieve(query)["matches"][0]
        assert match["path"] == path, query
        assert text in match["text"], query
    log = run_retrieve("logs/printer-a.log")["matches"][0]
    assert log["path"] == "logs/printer-a.log"
    assert "text('logs/printer-a.log')" in log["text"]


def test_python_reads_bundled_files_by_path():
    jobs = run_python(COUNT_CODE)
    assert jobs["ok"] is True, jobs
    assert int(jobs["stdout"].strip()) == evaluate(
        {"op": "count_rows", "table": "jobs", "where": {"machine": "printer-a"}}
    )
    lines = run_python(LOG_CODE)
    assert lines["ok"] is True, lines
    assert int(lines["stdout"].strip()) == evaluate(
        {"op": "count_text", "file": "logs/printer-a.log", "needle": "J02"}
    )
    blocked = run_python("import os\nprint(1)")
    assert blocked["ok"] is False


def test_extract_json_reads_an_object_wrapped_in_prose():
    parsed = extract_json(
        'Sure.\n```json\n{"verdict": "accept", "reason": "grounded"}\n```\nDone.'
    )
    assert parsed["verdict"] == "accept"
    parsed = extract_json('note {not json} then {"verdict": "revise", "reason": "fix"}')
    assert parsed["verdict"] == "revise"
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("   ")
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("no object here")


def test_a_printed_count_and_a_matching_line_count_are_grounded(tmp_path):
    log_task = _variant(LOG_QUESTION, tmp_path / "log")
    cited = call_tool("read_file", {"path": "logs/printer-a.log"}, tmp_path)
    value = log_task["gold_claims"][0]["value"]
    assert not contains_number(cited["text"], value)
    assert claims_are_grounded(
        log_task["question"],
        log_task["gold_claims"],
        [{"result": cited}],
        log_task["answer"],
    )
    printed = run_python(LOG_CODE)
    assert claims_are_grounded(
        log_task["question"],
        log_task["gold_claims"],
        [{"result": printed}],
        log_task["answer"],
    )
    assert not claims_are_grounded(
        "How many lines contain the token J02?",
        [{"name": "line_count", "value": 0, "citation": "call_1"}],
        [{"result": {"ok": True, "text": "no such token here\n"}}],
        "line_count is 0.",
    )


def test_worker_prompt_lists_files_and_a_refusal_still_stands(tmp_path):
    task = task_by_id("dev-015")
    result = run_team(task, ScriptedBackend(task), tmp_path)
    blob = "\n".join(
        message["content"] for example in result.examples for message in example["messages"]
    )
    assert "jobs.csv" in blob
    assert "logs/printer-a.log" in blob
    assert "table(name)" in blob
    assert "text(path)" in blob
    assert "table('jobs')" in blob
    contract = teacher_contract(task)
    assert "text(" in contract
    assert "Do not paste file text" in contract
    assert "One JSON object only" in contract
    assert result.status == "refuse"
    assert result.claims == []
    assert result.answer == task["answer"]


def test_printer_count_is_kept_when_python_reads_the_table(tmp_path):
    task = _variant(COUNT_QUESTION, tmp_path / "task")
    result = run_team(task, _CountWithPython(task), tmp_path / "run")
    assert result.gate_overrode is False
    assert result.answer == task["answer"]
    assert result.answer != UNGROUNDED
    assert result.claims == task["gold_claims"]
    kept, reason = trajectory_ok(task, result)
    assert kept, reason


def test_log_count_is_kept_when_the_cited_file_or_python_supports_it(tmp_path):
    task = _variant(LOG_QUESTION, tmp_path / "task")
    cited = _CiteTheLog(task)
    result = run_team(task, cited, tmp_path / "cited")
    assert cited.critic_tries == 3
    assert result.schema_invalid >= 2
    assert result.gate_overrode is False
    assert result.answer == task["answer"]
    assert result.answer != UNGROUNDED
    assert result.claims == task["gold_claims"]
    kept, reason = trajectory_ok(task, result)
    assert kept, reason

    printed = run_team(task, _CountWithText(task), tmp_path / "printed")
    assert printed.gate_overrode is False
    assert printed.answer == task["answer"]
    assert printed.claims == task["gold_claims"]
    kept, reason = trajectory_ok(task, printed)
    assert kept, reason
