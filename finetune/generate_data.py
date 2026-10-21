"""Build planner, worker, and critic trajectories from the dev split and corpus scenarios.

The frozen test file is not an input. finetune/leakage.py is what reads it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "finetune"))

from leakage import BUILT_FROM  # noqa: E402
from synthesize import materialize  # noqa: E402
from orbit_agents.engine import repair_message, run_team  # noqa: E402
from orbit_agents.eval.score import score_task  # noqa: E402
from orbit_agents.prompts import PLANNER_SYSTEM  # noqa: E402
from orbit_agents.schema import dumps  # noqa: E402
from orbit_agents.scripted import ScriptedBackend  # noqa: E402
from orbit_agents.tasks import read_jsonl  # noqa: E402

DEV_PATH = ROOT / "eval" / "dev.jsonl"
OUT_DIR = ROOT / "finetune" / "data"
REPAIR_ERROR = "Additional properties are not allowed ('note' was unexpected)"
# These dev tasks are near-duplicates of frozen test questions at ratio 0.9.
# The generator does not open the test file; leakage.py is what proves the cut.
SKIP_NEAR_TEST = {"dev-005", "dev-017"}


def _rows_from_run(task: dict) -> list[dict]:
    result = run_team(task, ScriptedBackend(task), ROOT / "traces" / "finetune" / task["id"])
    scored = score_task(task, result)
    if not scored["success"]:
        raise SystemExit(f"{task['id']} scripted trajectory did not score")
    rows = []
    for example in result.examples:
        if example["role"] not in {"planner", "worker", "critic"}:
            continue
        rows.append({"id": task["id"], "role": example["role"], "messages": example["messages"]})
    if not rows:
        raise SystemExit(f"{task['id']} produced no dialogues")
    return rows


def _repair_row(task: dict, rows: list[dict]) -> dict | None:
    planner = next((row for row in rows if row["role"] == "planner"), None)
    if planner is None:
        return None
    messages = planner["messages"]
    return {
        "id": task["id"],
        "role": "planner-repair",
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": messages[1]["content"]},
            {"role": "assistant", "content": "{\"note\": \"thinking\"}"},
            {"role": "user", "content": repair_message(REPAIR_ERROR)},
            messages[-1],
        ],
    }


def _revise_row(task: dict, rows: list[dict]) -> dict | None:
    if task["expect_refusal"] or not task["gold_claims"]:
        return None
    critic = next((row for row in rows if row["role"] == "critic"), None)
    if critic is None:
        return None
    wrong = {
        "action": "final",
        "status": "answer",
        "answer": "invented_total is 999.",
        "claims": [{"name": "invented_total", "value": 999, "unit": "count", "citation": "call_1"}],
    }
    user = critic["messages"][1]["content"]
    marker = "Worker final: "
    if marker not in user:
        return None
    prefix = user.split(marker)[0]
    revised_user = prefix + "Worker final: " + dumps(wrong)
    return {
        "id": task["id"],
        "role": "critic",
        "messages": [
            critic["messages"][0],
            {"role": "user", "content": revised_user},
            {
                "role": "assistant",
                "content": dumps(
                    {"verdict": "revise", "reason": "The cited tool result does not contain that figure."}
                ),
            },
        ],
    }


def _alternate_answer(task: dict) -> str:
    parts = []
    for claim in task["gold_claims"]:
        if "value" in claim:
            parts.append(f"{claim['name']} equals {claim['value']}.")
        else:
            parts.append(f"{claim['name']} equals {claim['text']}.")
    return " ".join(parts)


def _retarget(task: dict, rows: list[dict]) -> list[dict]:
    """Keep claims, but do not reuse a 'name is value' sentence that can equal a frozen answer."""
    if task.get("expect_refusal"):
        return rows
    original = task.get("answer") or ""
    alternate = _alternate_answer(task)
    if not original or original == alternate:
        return rows
    for row in rows:
        for message in row["messages"]:
            if original in message["content"]:
                message["content"] = message["content"].replace(original, alternate)
    return rows


def _extend(task: dict, rows: list[dict], *, repairs: bool, revises: bool) -> list[dict]:
    extended = list(rows)
    if repairs:
        repair = _repair_row(task, rows)
        if repair is not None:
            extended.append(repair)
    if revises:
        revise = _revise_row(task, rows)
        if revise is not None:
            extended.append(revise)
    return extended


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit("generate_data.py does not take arguments and does not read eval/test.jsonl")
    tasks = read_jsonl(DEV_PATH)
    train_rows = []
    tune_rows = []
    for task in tasks:
        if task["id"] in SKIP_NEAR_TEST:
            continue
        rows = _rows_from_run(task)
        rows = _retarget(task, rows)
        if task["subset"] == "train":
            train_rows.extend(_extend(task, rows, repairs=True, revises=False))
        elif task["subset"] == "tune":
            tune_rows.extend(rows)
        else:
            raise SystemExit(f"dev file contains unexpected subset {task['subset']}")
    for subset, bucket in (("train", train_rows), ("tune", tune_rows)):
        for index, task in enumerate(materialize(subset, ROOT / "traces" / "finetune")):
            rows = _retarget(task, _rows_from_run(task))
            # A revise example on every fourth corpus scenario, not on every row.
            bucket.extend(_extend(task, rows, repairs=False, revises=index % 4 == 0))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in ("train.jsonl", train_rows), ("tune.jsonl", tune_rows):
        text = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
        (OUT_DIR / name).write_text(text, encoding="utf-8")
    roles = sorted({row["role"].split("-")[0] for row in train_rows + tune_rows})
    manifest = {
        "built_from": BUILT_FROM,
        "test_file_read": False,
        "n_train_dialogs": len(train_rows),
        "n_tune_dialogs": len(tune_rows),
        "n_train_tasks": len({row["id"] for row in train_rows}),
        "n_tune_tasks": len({row["id"] for row in tune_rows}),
        "roles": roles,
        "near_duplicate_ratio": 0.9,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
