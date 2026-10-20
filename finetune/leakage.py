"""Fail if the frozen test split appears in the training dialogues."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "finetune" / "data" / "train.jsonl"
TUNE = ROOT / "finetune" / "data" / "tune.jsonl"
TEST = ROOT / "eval" / "test.jsonl"
MANIFEST = ROOT / "finetune" / "data" / "manifest.json"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def check() -> dict:
    test_rows = _read(TEST)
    train_blob = _norm(TRAIN.read_text(encoding="utf-8"))
    tune_blob = _norm(TUNE.read_text(encoding="utf-8"))
    train_rows = _read(TRAIN)
    tune_rows = _read(TUNE)
    train_ids = {row["id"] for row in train_rows}
    tune_ids = {row["id"] for row in tune_rows}
    overlap = train_ids & tune_ids
    if overlap:
        raise SystemExit(f"tune ids also appear in train: {sorted(overlap)}")
    for task in test_rows:
        if task["id"] in train_blob or task["id"] in tune_blob:
            raise SystemExit(f"test id leaked: {task['id']}")
        if task["question"] in train_blob or task["question"] in tune_blob:
            raise SystemExit(f"test question leaked: {task['id']}")
        for call in task["oracle_calls"]:
            code = call["arguments"].get("code") or call["arguments"].get("query") or ""
            folded = _norm(code)
            if len(folded) >= 12 and (folded in train_blob or folded in tune_blob):
                raise SystemExit(f"test tool text leaked: {task['id']}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("test_file_read") is not False or manifest.get("built_from") != "eval/dev.jsonl only":
        raise SystemExit("finetune manifest does not record a dev-only build")
    return {
        "ok": True,
        "n_test_questions_checked": len(test_rows),
        "n_train_dialogs": len(train_rows),
        "n_tune_dialogs": len(tune_rows),
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
