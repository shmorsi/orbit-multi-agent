"""Score Orbit Code once on the frozen code split.

    python -m orbit_agents.eval.score_code --config team --model claude-haiku-5-5 --dry-run
    ORBIT_ANTHROPIC_API_KEY=... python -m orbit_agents.eval.score_code --config team --model claude-haiku-5-5

Dry-run prints the request bound and the cost cap. It does not read a key and
it does not write a file. A measured file is never overwritten.

Results for code-eval-v2 go to results/code_v2/<model>/<config>.json. The
superseded code-eval-v1 record stays in results/code/<model>/<config>.json and
is not overwritten. The test runner and the reviewer are code, so they do not
spend a model request. Transport retries, including the read-timeout retry,
are the distillation client's.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from orbit_agents.code_sandbox import run_pytest
from orbit_agents.code_team import run_code_single, run_code_team
from orbit_agents.codebench import PROTOCOL, PROTOCOL_V2, load_tasks, repos_path, split_spec, verify_benchmark
from orbit_agents.engine import MAX_CRITIC_RETRIES, MAX_SCHEMA_TRIES
from orbit_agents.eval.score_team import DEFAULT_MAX_COST_USD, RETRIES_PER_REQUEST, is_measured
from orbit_agents.paths import ROOT
from orbit_agents.policy import make_policy

CONFIGS = {"single": "Single agent", "team": "Team with reviewer"}


def request_bound(n_tasks: int, config: str) -> dict:
    """Model requests this command can send. Pytest and the reviewer are not requests.

    Each model turn can be repeated for schema retries. Each of those requests
    can be retried by the transport, including a read timeout, up to the teacher cap.
    """
    schema_attempts = MAX_SCHEMA_TRIES + 1
    if config == "single":
        model_turns = schema_attempts
    else:
        cycles = 1 + MAX_CRITIC_RETRIES
        model_turns = (1 + cycles * 2) * schema_attempts
    turns = n_tasks * model_turns
    return {
        "preflight": 1,
        "model_turns_per_task": model_turns,
        "max_turn_requests": turns,
        "upper_bound_without_retries": 1 + turns,
        "hard_cap": (1 + turns) * (1 + RETRIES_PER_REQUEST),
        "schema_attempts_per_turn": schema_attempts,
        "transport_retries_per_request": RETRIES_PER_REQUEST,
        "note": "Test runner and reviewer do not spend a request. A read timeout counts as a transport retry.",
    }


def model_dir_name(model: str) -> str:
    """Folder name for the model id. Slashes become a double hyphen."""
    if not model or ".." in model or "\\" in model or "\x00" in model:
        raise SystemExit(f"refusing model name {model!r}")
    slug = model.replace("/", "--")
    if not all(char.isalnum() or char in "._-" for char in slug):
        raise SystemExit(f"refusing model name {model!r}")
    return slug


def output_path(root: Path, config: str, model: str, protocol: str = PROTOCOL_V2) -> Path:
    return root / split_spec(protocol)["results"] / model_dir_name(model) / f"{config}.json"


def _price(model: str) -> tuple[dict, str]:
    from orbit_agents.policy import _distill_teacher

    price, source = _distill_teacher().price_for(model, 0)
    return {"input": price["input"], "output": price["output"]}, source


def _hidden_rate(task: dict, workspace: Path, root: Path) -> dict:
    """Copy hidden tests into a scratch dir and run them. The agent workspace is unchanged."""
    scratch = workspace.parent / f"{workspace.name}-hidden"
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(workspace, scratch)
    hidden_src = repos_path(root, task.get("protocol") or PROTOCOL) / task["repo"] / "hidden"
    dest = scratch / "hidden_tests"
    shutil.copytree(hidden_src, dest)
    result = run_pytest(scratch, select="hidden_tests")
    total = result["n_passed"] + result["n_failed"] + result["n_errors"]
    rate = 0.0 if total == 0 else result["n_passed"] / total
    shutil.rmtree(scratch, ignore_errors=True)
    return {"rate": rate, "n_passed": result["n_passed"], "n_total": total, "ok": result["ok"]}


def score_one(task: dict, policy, workspace: Path, *, config: str, root: Path) -> dict:
    """One task. pass@1 needs a reviewer accept and hidden tests that pass."""
    runner = run_code_single if config == "single" else run_code_team
    result = runner(task, policy, workspace, root=root)
    hidden = _hidden_rate(task, workspace / "repo", root)
    rejected = result.status != "answer"
    passed = result.status == "answer" and hidden["ok"]
    return {
        "id": task["id"],
        "pass": passed,
        "hidden_passed_rate": hidden["rate"],
        "n_hidden_passed": hidden["n_passed"],
        "n_hidden": hidden["n_total"],
        "reviewer_rejected": rejected,
        "status": result.status,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        raise ValueError("no rows")
    n_pass = sum(1 for row in rows if row["pass"])
    n_rejected = sum(1 for row in rows if row["reviewer_rejected"])
    return {
        "n_tasks": n,
        "n_pass": n_pass,
        "pass_at_1": n_pass / n,
        "tests_passed_rate": sum(row["hidden_passed_rate"] for row in rows) / n,
        "n_reviewer_rejections": n_rejected,
        "reviewer_rejection_rate": n_rejected / n,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Orbit Code once on the frozen code split")
    parser.add_argument("--config", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--provider", choices=("anthropic", "transformers", "ollama", "llamacpp"), default="anthropic")
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument("--requests-per-minute", type=float, default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--split", choices=(PROTOCOL_V2, PROTOCOL), default=PROTOCOL_V2)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Print the request bound and cost cap. No key and no file.")
    return parser


def main(argv: list[str] | None = None, *, policy=None) -> dict:
    """Score one config once, unless that file is already measured."""
    args = build_parser().parse_args(argv)
    frozen = verify_benchmark(args.root, args.split)
    tasks = load_tasks(args.root, args.split)
    if any(task.get("split") != "test" for task in tasks):
        raise SystemExit("code benchmark tasks must use the test split")
    bound = request_bound(len(tasks), args.config)
    max_requests = args.max_requests or bound["hard_cap"]
    if max_requests < 1 or args.max_cost_usd < 0:
        raise SystemExit("--max-requests must be at least 1 and --max-cost-usd zero or more")
    out = output_path(args.root, args.config, args.model, args.split)
    price = None
    price_source = None
    if args.provider == "anthropic":
        price, price_source = _price(args.model)
    plan = {
        "event": "plan",
        "config": args.config,
        "label": CONFIGS[args.config],
        "provider": args.provider,
        "model": args.model,
        "frozen_test": frozen,
        "n_tasks": len(tasks),
        "requests": bound,
        "max_requests": max_requests,
        "max_cost_usd": args.max_cost_usd if args.provider == "anthropic" else None,
        "price_usd_per_million": price,
        "price_source": price_source,
        "out": str(out.relative_to(args.root)),
        "already_measured": is_measured(out),
    }
    print(json.dumps(plan), flush=True)
    if plan["already_measured"]:
        raise SystemExit(f"{plan['out']} is already measured. The frozen code split is scored once per config.")
    if args.dry_run:
        return plan
    kwargs: dict = {}
    if args.provider == "anthropic":
        kwargs = {
            "max_requests": max_requests,
            "max_cost_usd": args.max_cost_usd,
            "requests_per_minute": args.requests_per_minute,
        }
    elif args.provider == "transformers":
        kwargs = {"adapter": args.adapter, "max_new_tokens": args.max_new_tokens}
    shared = policy
    if shared is None:
        shared = make_policy(args.provider, args.model, **kwargs)
        if hasattr(shared, "preflight"):
            try:
                shared.preflight()
            except shared.stops as exc:
                return _write_stop(out, args, frozen, max_requests, kwargs, shared, [], exc, None)
    stops = getattr(shared, "stops", ())
    rows = []
    stop = None
    for task in tasks:
        try:
            rows.append(
                score_one(
                    task,
                    shared,
                    args.root / "traces" / "code_eval" / args.config / task["id"],
                    config=args.config,
                    root=args.root,
                )
            )
        except stops as exc:
            stop = exc
            break
    if stop is not None:
        return _write_stop(out, args, frozen, max_requests, kwargs, shared, rows, stop, rows[-1]["id"] if rows else None)
    report = _base(args, frozen, max_requests, kwargs, shared)
    report.update({"status": "measured", **summarize(rows), "per_task": rows})
    report["note"] = (
        "One score on the frozen code split. The policy saw the issue and the source files, not the hidden tests."
    )
    _write(out, report)
    print(json.dumps({"event": "done", "status": "measured", "out": plan["out"], "pass_at_1": report["pass_at_1"]}), flush=True)
    return report


def _base(args, frozen: dict, max_requests: int, kwargs: dict, policy) -> dict:
    return {
        "kind": "model_eval",
        "split": "test",
        "protocol": frozen["protocol"],
        "split_tag": frozen["split_tag"],
        "bundle_sha256": frozen["bundle_sha256"],
        "config": args.config,
        "config_label": CONFIGS[args.config],
        "provider": args.provider,
        "model": args.model,
        "adapter": args.adapter,
        "max_requests": max_requests,
        "max_cost_usd": kwargs.get("max_cost_usd"),
        "usage": policy.usage() if policy is not None else {},
    }


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _write_stop(path, args, frozen, max_requests, kwargs, policy, rows, exc, after_task) -> dict:
    report = _base(args, frozen, max_requests, kwargs, policy)
    report.update(
        {
            "status": "stopped",
            "stop": {"reason": exc.reason, "detail": exc.detail, "after_task": after_task},
            "n_tasks_finished": len(rows),
            "per_task": None,
            "note": "Stopped before every task finished, so there is no score. Nothing here is a measurement.",
        }
    )
    _write(path, report)
    print(json.dumps({"event": "done", "status": "stopped", "out": str(path)}), flush=True)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
