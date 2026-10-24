"""Run the frozen split. A scripted run is a harness check. A model run is the only measured model score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbit_agents.engine import run_single, run_team
from orbit_agents.eval.metrics import summarize
from orbit_agents.eval.score import score_task
from orbit_agents.eval.score_team import verify_frozen_test
from orbit_agents.paths import ROOT
from orbit_agents.scripted import ScriptedBackend
from orbit_agents.tasks import read_jsonl

# Committed files the generators treat as archived scores, harness checks, or
# pending placeholders. A new model score does not land in any of these.
LOCKED_RESULT_FILES = {
    "results/single_base.json",
    "results/team_base.json",
    "results/team_lora.json",
    "results/team_distill.json",
    "results/comparison.json",
    "results/finetune_eval.json",
    "results/harness_single.json",
    "results/harness_team.json",
}
LOCKED_RESULT_PREFIXES = ("results/v1_", "results/v2_")
MODEL_NOTE = "One score on the frozen test."
HARNESS_NOTE = "A system_check replays the scripted harness. It is not a language-model score."


def result_relative(path: Path, root: Path = ROOT) -> str | None:
    """Path relative to the repo, or None when it is outside the checkout."""
    candidate = path if path.is_absolute() else root / path
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def locked_result_reason(path: Path, root: Path = ROOT) -> str | None:
    """Refuse writes onto archived scores, harness checks, and pending placeholders."""
    relative = result_relative(path, root)
    if relative is None:
        return None
    if relative in LOCKED_RESULT_FILES or relative.startswith(LOCKED_RESULT_PREFIXES):
        return (
            f"{relative} is an archived result, a harness check, or a pending placeholder. "
            "A new model score goes under results/v3_qwen1.5b/ or results/v3_api/<model>/."
        )
    return None


def measured_model_eval(path: Path, root: Path = ROOT) -> bool:
    """True when the file exists and is a measured model eval."""
    candidate = path if path.is_absolute() else root / path
    if not candidate.is_file():
        return False
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "measured" and payload.get("kind") == "model_eval"

SPLITS = {
    "dev": ROOT / "eval" / "dev.jsonl",
    "tune": ROOT / "eval" / "dev.jsonl",
    "test": ROOT / "eval" / "test.jsonl",
}


def load_tasks(split: str) -> list[dict]:
    """Load one split. The tune split keeps only rows marked tune."""
    rows = read_jsonl(SPLITS[split])
    if split == "tune":
        return [task for task in rows if task.get("subset") == "tune"]
    return rows


def load_backend(name: str, task: dict, model: str, adapter: str | None, max_new_tokens: int):
    """Build the named backend. A scripted backend is tied to one task."""
    if name == "scripted":
        return ScriptedBackend(task)
    if name == "transformers":
        from orbit_agents.backends import TransformersBackend

        return TransformersBackend(model, adapter=adapter, max_new_tokens=max_new_tokens)
    if name == "ollama":
        from orbit_agents.backends import OllamaBackend

        return OllamaBackend(model)
    if name == "llamacpp":
        from orbit_agents.backends import LlamaCppBackend

        return LlamaCppBackend(model)
    raise SystemExit(f"unknown backend {name}")


def execute_v2(
    tasks: list[dict], *, backend_name: str, model: str, adapter: str | None, max_new_tokens: int, workspace: Path
) -> list[dict]:
    """The six-role team. Same tools, gate, and scorer as v1."""
    from orbit_agents.policy import LegacyBackendPolicy
    from orbit_agents.scripted_v2 import ScriptedPolicyV2
    from orbit_agents.team_v2 import run_team_v2

    shared = None
    if backend_name != "scripted":
        shared = LegacyBackendPolicy(
            load_backend(backend_name, tasks[0], model, adapter, max_new_tokens), provider=backend_name, model=model
        )
    rows = []
    for task in tasks:
        policy = shared or ScriptedPolicyV2(task)
        rows.append(score_task(task, run_team_v2(task, policy, workspace / task["id"])))
    return rows


def execute_v3(
    tasks: list[dict], *, backend_name: str, model: str, adapter: str | None, max_new_tokens: int, workspace: Path
) -> list[dict]:
    """The round table. Same tools, gate, and scorer as v2. A system check, not an API score."""
    from orbit_agents.policy import LegacyBackendPolicy
    from orbit_agents.scripted_v3 import ScriptedPolicyV3
    from orbit_agents.team_v3 import run_team_v3

    shared = None
    if backend_name != "scripted":
        shared = LegacyBackendPolicy(
            load_backend(backend_name, tasks[0], model, adapter, max_new_tokens), provider=backend_name, model=model
        )
    rows = []
    for task in tasks:
        policy = shared or ScriptedPolicyV3(task)
        rows.append(score_task(task, run_team_v3(task, policy, workspace / task["id"])))
    return rows


def execute(
    tasks: list[dict],
    *,
    mode: str,
    backend_name: str,
    model: str,
    adapter: str | None,
    max_new_tokens: int,
    workspace: Path,
) -> list[dict]:
    """Score every task with the three-role team or the single model."""
    shared = None
    if backend_name != "scripted":
        shared = load_backend(backend_name, tasks[0], model, adapter, max_new_tokens)
    rows = []
    runner = run_team if mode == "team" else run_single
    for task in tasks:
        backend = shared or load_backend("scripted", task, model, adapter, max_new_tokens)
        result = runner(task, backend, workspace / task["id"])
        rows.append(score_task(task, result))
    return rows


def main(argv: list[str] | None = None) -> None:
    """Score a split and write one JSON file, unless the output path is locked."""
    parser = argparse.ArgumentParser(description="Score Orbit on a frozen split")
    parser.add_argument("--split", choices=("dev", "tune", "test"), default="test")
    parser.add_argument("--mode", choices=["team", "single"], default="team")
    parser.add_argument(
        "--team",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="v1: planner, worker, critic. v2: six roles. v3: round table",
    )
    parser.add_argument("--backend", choices=["scripted", "transformers", "ollama", "llamacpp"], default="scripted")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.split == "test":
        verify_frozen_test(ROOT)
    locked = locked_result_reason(args.out)
    if locked:
        raise SystemExit(locked)
    if args.split == "test" and measured_model_eval(args.out):
        raise SystemExit(f"{args.out} is already measured. The frozen test is scored once per config.")
    tasks = load_tasks(args.split)
    kind = "system_check" if args.backend == "scripted" else "model_eval"
    if args.team == "v2":
        if args.mode != "team":
            raise SystemExit("--team v2 is a team; --mode single is the v1 baseline")
        rows = execute_v2(
            tasks,
            backend_name=args.backend,
            model=args.model,
            adapter=args.adapter,
            max_new_tokens=args.max_new_tokens,
            workspace=ROOT / "traces" / "eval" / "team_v2" / args.backend,
        )
    elif args.team == "v3":
        if args.mode != "team":
            raise SystemExit("--team v3 is a team; --mode single is the v1 baseline")
        rows = execute_v3(
            tasks,
            backend_name=args.backend,
            model=args.model,
            adapter=args.adapter,
            max_new_tokens=args.max_new_tokens,
            workspace=ROOT / "traces" / "eval" / "team_v3" / args.backend,
        )
    else:
        rows = execute(
            tasks,
            mode=args.mode,
            backend_name=args.backend,
            model=args.model,
            adapter=args.adapter,
            max_new_tokens=args.max_new_tokens,
            workspace=ROOT / "traces" / "eval" / args.mode / args.backend,
        )
    summary = summarize(rows)
    report = {
        "status": "measured",
        "kind": kind,
        "split": args.split,
        "mode": args.mode if args.team == "v1" else f"team_{args.team}",
        "backend": args.backend,
        "model": None if args.backend == "scripted" else args.model,
        "adapter": args.adapter,
        **summary,
        "per_task": rows,
        "note": MODEL_NOTE if kind == "model_eval" else HARNESS_NOTE,
    }
    if args.team != "v1":
        report["team"] = args.team
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "kind": kind, "n_success": summary["n_success"], "n_tasks": summary["n_tasks"]}))


if __name__ == "__main__":
    main()
