"""Write a JSONL trace from a real engine run. The replay page reads these files and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

from orbit_agents.engine import FLIGHT_RULES, run_single, run_team
from orbit_agents.paths import ROOT
from orbit_agents.scripted import ReviseOnce, ScriptedBackend
from orbit_agents.tasks import all_tasks

TRACE_DIR = ROOT / "mission" / "traces"


def mission_header(task: dict, result, *, harness: str | None = None, team: str = "v1") -> dict:
    """The first trace row. A v1 header is byte-for-byte what it was before team v2 existed."""
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
    if team == "v2":
        from orbit_agents.team_v2 import FLIGHT_RULES_V2

        header["team"] = "v2"
        header["flight_rules"] = FLIGHT_RULES_V2
        header["n_audit_refusals"] = 1 if result.gate_overrode else 0
    elif team == "v3":
        from orbit_agents.team_v3 import FLIGHT_RULES_V3

        header["team"] = "v3"
        header["flight_rules"] = FLIGHT_RULES_V3
        header["n_audit_refusals"] = 1 if result.gate_overrode else 0
    if harness:
        header["harness"] = harness
    return header


def dump_trace(task: dict, result, *, harness: str | None = None, team: str = "v1") -> str:
    """Serialize one run as JSONL, header first."""
    rows = [mission_header(task, result, harness=harness, team=team), *result.events]
    return "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)


def write_trace(path: Path, task: dict, backend, *, single: bool = False, harness: str | None = None) -> Path:
    """Run the three-role team or the single model and write its JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    workspace = path.parent / ".runs" / path.stem
    runner = run_single if single else run_team
    result = runner(task, backend, workspace)
    path.write_text(dump_trace(task, result, harness=harness), encoding="utf-8")
    return path


def bundled_traces(directory: Path | None = None) -> list[Path]:
    """Rewrite the committed v1 Mission Control traces from the engine."""
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


TRACE_DIR_V2 = TRACE_DIR / "v2"
# One accept, one revise, one refuse. The refusal is the auditor's: the critic accepted
# a draft whose answer added a figure that no tool produced.
V2_EXAMPLES = (
    ("accept.jsonl", "dev-012", "accept"),
    ("revise.jsonl", "dev-008", "revise"),
    ("refuse.jsonl", "dev-009", "auditor"),
)
AUDITOR_EXTRA_NUMBER = 40


def write_trace_v2(path: Path, task: dict, policy, *, harness: str | None = None) -> Path:
    """Run the six-role scripted policy and write its JSONL."""
    from orbit_agents.team_v2 import run_team_v2

    path.parent.mkdir(parents=True, exist_ok=True)
    workspace = path.parent / ".runs" / path.stem
    result = run_team_v2(task, policy, workspace)
    path.write_text(dump_trace(task, result, harness=harness, team="v2"), encoding="utf-8")
    return path


def _v2_policy(kind: str, task: dict):
    from orbit_agents.scripted_v2 import AuditorCatch, ReviseOnceV2, ScriptedPolicyV2

    if kind == "revise":
        return ReviseOnceV2(task)
    if kind == "auditor":
        return AuditorCatch(task, AUDITOR_EXTRA_NUMBER)
    return ScriptedPolicyV2(task)


def bundled_traces_v2(directory: Path | None = None) -> list[Path]:
    """Real six-role engine runs under the scripted harness, in the replay's JSONL schema."""
    directory = directory or TRACE_DIR_V2
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {task["id"]: task for task in all_tasks()}
    written = []
    index = []
    for name, task_id, kind in V2_EXAMPLES:
        task = by_id[task_id]
        path = write_trace_v2(directory / name, task, _v2_policy(kind, task), harness="scripted_harness")
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        written.append(path)
        index.append(
            {
                "file": path.name,
                "team": "v2",
                "task_id": header["task_id"],
                "status": header["status"],
                "outcome": kind,
                "question": header["question"],
            }
        )
    (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return written


TRACE_DIR_V3 = TRACE_DIR / "v3"
V3_EXAMPLES = (
    ("accept.jsonl", "dev-006", "accept"),
    ("revise.jsonl", "dev-008", "revise"),
    ("refuse.jsonl", "dev-009", "auditor"),
)


def write_trace_v3(path: Path, task: dict, policy, *, harness: str | None = None) -> Path:
    """Run the round table and write its JSONL."""
    from orbit_agents.team_v3 import run_team_v3

    path.parent.mkdir(parents=True, exist_ok=True)
    workspace = path.parent / ".runs" / path.stem
    result = run_team_v3(task, policy, workspace)
    path.write_text(dump_trace(task, result, harness=harness, team="v3"), encoding="utf-8")
    return path


def _v3_policy(kind: str, task: dict):
    from orbit_agents.scripted_v3 import AuditorCatchV3, ReviseOnceV3, ScriptedPolicyV3

    if kind == "revise":
        return ReviseOnceV3(task)
    if kind == "auditor":
        return AuditorCatchV3(task, AUDITOR_EXTRA_NUMBER)
    return ScriptedPolicyV3(task)


def bundled_traces_v3(directory: Path | None = None) -> list[Path]:
    """Real round-table engine runs under the scripted harness."""
    directory = directory or TRACE_DIR_V3
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {task["id"]: task for task in all_tasks()}
    written = []
    index = []
    for name, task_id, kind in V3_EXAMPLES:
        task = by_id[task_id]
        path = write_trace_v3(directory / name, task, _v3_policy(kind, task), harness="scripted_harness")
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        written.append(path)
        index.append(
            {
                "file": path.name,
                "team": "v3",
                "task_id": header["task_id"],
                "status": header["status"],
                "outcome": kind,
                "question": header["question"],
            }
        )
    (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> None:
    """Rewrite the bundled traces for v1, v2, or the round table."""
    import argparse

    parser = argparse.ArgumentParser(description="Rewrite the bundled scripted-harness traces")
    parser.add_argument("--team", choices=("v1", "v2", "v3"), default="v1")
    args = parser.parse_args(argv)
    if args.team == "v3":
        paths = bundled_traces_v3()
    elif args.team == "v2":
        paths = bundled_traces_v2()
    else:
        paths = bundled_traces()
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
