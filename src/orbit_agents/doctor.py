"""Check the local machine. The live model path is optional and never fakes a result."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from orbit_agents.engine import run_team
from orbit_agents.scripted import ScriptedBackend
from orbit_agents.tasks import task_by_id


def environment() -> dict:
    """Python, platform, and torch device, without loading a model."""
    device = "cpu"
    torch_version = None
    try:
        import torch

        from orbit_agents.backends import pick_device

        device = pick_device(torch)
        torch_version = torch.__version__
    except ImportError:
        device = "cpu"
    return {
        "python": platform.python_version(),
        "python_ok": sys.version_info >= (3, 11),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "torch": torch_version,
    }


def scripted_task() -> dict:
    """Time one scripted harness task and return the doctor record."""
    task = task_by_id("dev-001")
    if task is None:
        raise SystemExit("eval/dev.jsonl is missing dev-001")
    started = time.perf_counter()
    result = run_team(task, ScriptedBackend(task), Path("traces") / "doctor" / "dev-001")
    elapsed = time.perf_counter() - started
    return {
        "task_id": task["id"],
        "status": result.status,
        "ok": result.status == "answer" and result.schema_invalid == 0,
        "seconds": elapsed,
        "kind": "scripted_harness",
    }


def live_task(model: str) -> dict:
    """Time one real model task and return the doctor record."""
    from orbit_agents.backends import TransformersBackend

    task = task_by_id("dev-001")
    started = time.perf_counter()
    backend = TransformersBackend(model, max_new_tokens=128)
    result = run_team(task, backend, Path("traces") / "doctor" / "live")
    return {
        "task_id": task["id"],
        "status": result.status,
        "device": backend.device,
        "seconds": time.perf_counter() - started,
        "kind": "model_run",
        "model": model,
    }


def main(argv: list[str] | None = None) -> None:
    """Print the environment and one scripted task. --live adds a model call."""
    parser = argparse.ArgumentParser(description="Check Python, the compute device, and one tiny Orbit task")
    parser.add_argument("--live", action="store_true", help="Download the small model and run one real task")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args(argv)
    report = {"environment": environment(), "scripted": scripted_task(), "live": None}
    if not report["environment"]["python_ok"] or not report["scripted"]["ok"]:
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    if args.live:
        if report["environment"]["torch"] is None:
            print(json.dumps(report, indent=2))
            raise SystemExit("torch is not installed. Run make setup, then doctor --live. QLoRA is not required.")
        report["live"] = live_task(args.model)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
