"""Fail if the frozen test split appears in the training dialogues."""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from orbit_agents.eval.catalog import REFUSAL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "finetune" / "data" / "train.jsonl"
TUNE = ROOT / "finetune" / "data" / "tune.jsonl"
TEST = ROOT / "eval" / "test.jsonl"
MANIFEST = ROOT / "finetune" / "data" / "manifest.json"
BUILT_FROM = "eval/dev.jsonl and corpus tool scenarios; eval/test.jsonl was not read"
NEAR_DUPLICATE_RATIO = 0.9


class FrozenTestLeak(RuntimeError):
    """A prompt or row would have carried frozen-test material. Nothing was sent."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.reason = "leakage"
        self.detail = detail


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _folded(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _task_lines(row: dict) -> list[str]:
    found = []
    for message in row.get("messages", []):
        content = message.get("content") or ""
        for line in content.splitlines():
            if line.startswith("Task: "):
                found.append(line[len("Task: ") :].strip())
    return found


def _near_duplicate(left: str, right: str) -> bool:
    folded_left = _folded(left)
    folded_right = _folded(right)
    if len(folded_left) < 24 or len(folded_right) < 24:
        return False
    if folded_left == folded_right or folded_left in folded_right or folded_right in folded_left:
        return True
    return SequenceMatcher(None, folded_left, folded_right).ratio() >= NEAR_DUPLICATE_RATIO


def blocking_test_task(question: str, test_rows: list[dict]) -> dict | None:
    """Return the frozen task whose question matches, or None.

    Used to keep a candidate out of the teacher prompt. The test question is not copied.
    """
    for task in test_rows:
        if question == task["question"] or _near_duplicate(question, task["question"]):
            return task
    return None


def guard_messages(messages: list[dict], test_rows: list[dict]) -> None:
    """Raise FrozenTestLeak before a prompt that contains a frozen id or question is sent."""
    blob = "\n".join((message.get("content") or "") for message in messages)
    for task in test_rows:
        task_id = task["id"]
        question = task["question"]
        if task_id and task_id in blob:
            raise FrozenTestLeak(f"test id {task_id}")
        if question and question in blob:
            raise FrozenTestLeak(f"test question {task_id}")
        for line in blob.splitlines():
            if not line.startswith("Task: "):
                continue
            asked = line[len("Task: ") :].strip()
            if asked == question or _near_duplicate(question, asked):
                raise FrozenTestLeak(f"near-duplicate of {task_id}")


def overlap_error(train_rows: list[dict], tune_rows: list[dict], test_rows: list[dict]) -> str | None:
    train_ids = {row["id"] for row in train_rows}
    tune_ids = {row["id"] for row in tune_rows}
    overlap = train_ids & tune_ids
    if overlap:
        return f"tune ids also appear in train: {sorted(overlap)}"
    train_blob = _norm(json.dumps(train_rows, ensure_ascii=True))
    tune_blob = _norm(json.dumps(tune_rows, ensure_ascii=True))
    folded_train = _folded(train_blob)
    folded_tune = _folded(tune_blob)
    for task in test_rows:
        task_id = task["id"]
        if task_id in train_blob or task_id in tune_blob:
            return f"test id leaked: {task_id}"
        question = task["question"]
        if question in train_blob or question in tune_blob:
            return f"test question leaked: {task_id}"
        for row in train_rows + tune_rows:
            for line in _task_lines(row):
                if _near_duplicate(question, line):
                    return f"near-duplicate of test question: {task_id} in {row['id']}"
        answer = _norm(task.get("answer") or "")
        # Every trap, including the dev traps, uses this one harness sentence.
        if answer and answer != REFUSAL:
            if len(answer) >= 8 and (answer in train_blob or answer in tune_blob):
                return f"test answer leaked: {task_id}"
            folded_answer = _folded(answer)
            if len(folded_answer) >= 8 and (folded_answer in folded_train or folded_answer in folded_tune):
                return f"test answer leaked: {task_id}"
        for call in task.get("oracle_calls") or []:
            code = call["arguments"].get("code") or call["arguments"].get("query") or ""
            folded = _norm(code)
            if len(folded) >= 12 and (folded in train_blob or folded in tune_blob):
                return f"test tool text leaked: {task_id}"
    return None


def check() -> dict:
    test_rows = _read(TEST)
    train_rows = _read(TRAIN)
    tune_rows = _read(TUNE)
    error = overlap_error(train_rows, tune_rows, test_rows)
    if error:
        raise SystemExit(error)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("test_file_read") is not False or manifest.get("built_from") != BUILT_FROM:
        raise SystemExit("finetune manifest does not record a corpus build that left the test file unread")
    return {
        "ok": True,
        "n_test_questions_checked": len(test_rows),
        "n_train_dialogs": len(train_rows),
        "n_tune_dialogs": len(tune_rows),
        "near_duplicate_ratio": NEAR_DUPLICATE_RATIO,
        "checks": ["ids", "questions", "near_duplicate_text", "answer_strings", "tool_snippets"],
    }


def main() -> None:
    print(json.dumps(check(), indent=2))


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(exc, file=sys.stderr)
        raise
