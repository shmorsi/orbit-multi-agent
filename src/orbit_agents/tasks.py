"""Load the frozen task files. Live questions that are not in those files stay unmarked."""

from __future__ import annotations

import json
from pathlib import Path

from orbit_agents.paths import ROOT

DEV_PATH = ROOT / "eval" / "dev.jsonl"
TEST_PATH = ROOT / "eval" / "test.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into dict rows."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def all_tasks() -> list[dict]:
    """Dev tasks followed by test tasks."""
    return read_jsonl(DEV_PATH) + read_jsonl(TEST_PATH)


def task_by_id(task_id: str) -> dict | None:
    """Return the task with this id, or None."""
    for task in all_tasks():
        if task["id"] == task_id:
            return task
    return None


def ad_hoc_task(question: str) -> dict:
    """A live question with no gold and no oracle. It cannot be scored."""
    return {
        "id": "live",
        "split": "live",
        "subset": "live",
        "question": question.strip(),
        "expect_refusal": False,
        "plan": [],
        "oracle_calls": [],
        "gold_claims": [],
        "answer": "",
        "status": "answer",
    }
