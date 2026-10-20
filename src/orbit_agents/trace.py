"""Write a JSONL trace from a real engine run. The replay page reads these files and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

from orbit_agents.engine import FLIGHT_RULES, run_single, run_team
from orbit_agents.paths import ROOT
from orbit_agents.scripted import ReviseOnce, ScriptedBackend
from orbit_agents.tasks import all_tasks

TRACE_DIR = ROOT / "mission" / "traces"


def mission_header(task: dict, result, *, harness: str | None = None) -> dict:
    header = {
        "kind": "mission",
        "source": "orbit_agents.engine",
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
        "n_tool_calls": len(result.tool_calls),
        "n_retries": sum(1 for event in result.events if event["kind"] == "revise"),
        "gate_overrode": result.gate_overrode,
        "flight_rules": FLIGHT_RULES,
    }
    if harness:
        header["harness"] = harness
    return header


def dump_trace(task: dict, result, *, harness: str | None = None) -> str:
    rows = [mission_header(task, result, harness=harness), *result.events]
    return "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)


def write_trace(path: Path, task: dict, backend, *, single: bool = False, harness: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    workspace = path.parent / ".runs" / path.stem
    runner = run_single if single else run_team
    result = runner(task, backend, workspace)
    path.write_text(dump_trace(task, result, harness=harness), encoding="utf-8")
    return path


def bundled_traces(directory: Path | None = None) -> list[Path]:
    directory = directory or TRACE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {task["id"]: task for task in all_tasks()}
    written = [
        write_trace(directory / "quotient.jsonl", by_id["dev-012"], ScriptedBackend(by_id["dev-012"]), harness="scripted_harness"),
        write_trace(directory / "abort.jsonl", by_id["dev-015"], ScriptedBackend(by_id["dev-015"]), harness="scripted_harness"),
        write_trace(directory / "revise.jsonl", by_id["dev-001"], ReviseOnce(by_id["dev-001"]), harness="scripted_harness"),
    ]
    index = []
    for path in written:
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        index.append(
            {
                "file": path.name,
                "task_id": header["task_id"],
                "status": header["status"],
                "question": header["question"],
            }
        )
    (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return written


def main() -> None:
    for path in bundled_traces():
        print(path)


if __name__ == "__main__":
    main()
