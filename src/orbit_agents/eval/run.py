"""Run the frozen split. A scripted run is a harness check. A model run is the only measured model score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbit_agents.eval.metrics import summarize
from orbit_agents.eval.score import score_task
from orbit_agents.engine import run_single, run_team
from orbit_agents.paths import ROOT
from orbit_agents.scripted import ScriptedBackend
from orbit_agents.tasks import read_jsonl

SPLITS = {
    "dev": ROOT / "eval" / "dev.jsonl",
    "test": ROOT / "eval" / "test.jsonl",
}


def load_backend(name: str, task: dict, model: str, adapter: str | None, max_new_tokens: int):
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


def execute(tasks: list[dict], *, mode: str, backend_name: str, model: str, adapter: str | None, max_new_tokens: int, workspace: Path):
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
    parser = argparse.ArgumentParser(description="Score Orbit on a frozen split")
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--mode", choices=["team", "single"], default="team")
    parser.add_argument("--backend", choices=["scripted", "transformers", "ollama", "llamacpp"], default="scripted")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    tasks = read_jsonl(SPLITS[args.split])
    kind = "system_check" if args.backend == "scripted" else "model_eval"
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
        "mode": args.mode,
        "backend": args.backend,
        "model": None if args.backend == "scripted" else args.model,
        "adapter": args.adapter,
        **summary,
        "per_task": rows,
        "note": "A system_check replays the scripted harness. It is not a language-model score.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "kind": kind, "n_success": summary["n_success"], "n_tasks": summary["n_tasks"]}))


if __name__ == "__main__":
    main()
