"""Build planner and worker trajectories from the dev split only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit_agents.engine import run_team  # noqa: E402
from orbit_agents.eval.score import score_task  # noqa: E402
from orbit_agents.prompts import PLANNER_SYSTEM  # noqa: E402
from orbit_agents.scripted import ScriptedBackend  # noqa: E402
from orbit_agents.tasks import read_jsonl  # noqa: E402

DEV_PATH = ROOT / "eval" / "dev.jsonl"
OUT_DIR = ROOT / "finetune" / "data"
REPAIR = "The previous draft was not a JSON object. Reply with one JSON object."


def dialogue_rows(task: dict) -> list[dict]:
    result = run_team(task, ScriptedBackend(task), ROOT / "traces" / "finetune" / task["id"])
    scored = score_task(task, result)
    if not scored["success"]:
        raise SystemExit(f"{task['id']} scripted trajectory did not score")
    rows = []
    for example in result.examples:
        if example["role"] not in {"planner", "worker"}:
            continue
        rows.append({"id": task["id"], "role": example["role"], "messages": example["messages"]})
    if task["subset"] == "train" and rows:
        first = rows[0]["messages"]
        repair_user = first[1]["content"] + "\n" + REPAIR
        rows.append(
            {
                "id": task["id"],
                "role": "planner-repair",
                "messages": [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": repair_user},
                    first[-1],
                ],
            }
        )
    return rows


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit("generate_data.py reads eval/dev.jsonl only")
    tasks = read_jsonl(DEV_PATH)
    train_rows = []
    tune_rows = []
    for task in tasks:
        if task["subset"] == "train":
            train_rows.extend(dialogue_rows(task))
        elif task["subset"] == "tune":
            tune_rows.extend(dialogue_rows(task))
        else:
            raise SystemExit(f"dev file contains unexpected subset {task['subset']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in ("train.jsonl", train_rows), ("tune.jsonl", tune_rows):
        text = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
        (OUT_DIR / name).write_text(text, encoding="utf-8")
    manifest = {
        "built_from": "eval/dev.jsonl only",
        "test_file_read": False,
        "n_train_dialogs": len(train_rows),
        "n_tune_dialogs": len(tune_rows),
        "n_train_tasks": sum(1 for task in tasks if task["subset"] == "train"),
        "n_tune_tasks": sum(1 for task in tasks if task["subset"] == "tune"),
        "roles": ["planner", "worker"],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
