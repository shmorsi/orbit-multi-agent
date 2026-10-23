"""Score one team with one policy on the frozen test, once.

    python -m orbit_agents.eval.score_team --team v2 --provider anthropic --model claude-haiku-5-5 --dry-run
    ORBIT_ANTHROPIC_API_KEY=... python -m orbit_agents.eval.score_team --team v2 --provider anthropic --model claude-haiku-5-5

Before any request this command checks the frozen test: its sha256 must match
eval/manifest.json and, when the git tag is present, the file at tag
team-eval-v1. It refuses to run when the output file is already measured. An
API run goes through AnthropicPolicy, which reuses the distillation request cap
and cost stop. A run that stops early writes `status: stopped` and no metrics.

Results go under results/v3_api/<model>/team_v1.json or team_v2.json. The
folder name says harness v3, the same tools and gate as results/v3_qwen1.5b/.
When both teams are measured for one model, comparison.json holds the paired
bootstrap of v2 minus v1. The README generator reads only measured files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from orbit_agents.engine import MAX_TURNS, run_team
from orbit_agents.eval.metrics import bootstrap_mean_difference, summarize
from orbit_agents.eval.score import score_task
from orbit_agents.paths import ROOT
from orbit_agents.policy import PER_TASK_PROVIDERS, PolicyBackend, make_policy
from orbit_agents.tasks import read_jsonl

API_RESULTS = Path("results") / "v3_api"
TEAMS = {"v1": "v1 team (3 roles)", "v2": "v2 team (6 roles)"}
DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_MAX_COST_USD = 5.0
# The teacher's backoff schedule: up to three retries per request, each counted.
RETRIES_PER_REQUEST = 3
PROTOCOL = "team-eval-v1"


def request_bound(n_tasks: int) -> dict:
    """Requests this command can send. Every team turn, schema retry included, is one request."""
    turns = n_tasks * MAX_TURNS
    return {
        "preflight": 1,
        "max_turn_requests": turns,
        "upper_bound_without_retries": 1 + turns,
        "hard_cap": (1 + turns) * (1 + RETRIES_PER_REQUEST),
    }


def verify_frozen_test(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "eval" / "manifest.json").read_text(encoding="utf-8"))
    body = (root / "eval" / "test.jsonl").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if manifest.get("protocol") != PROTOCOL or digest != manifest.get("test_sha256"):
        raise SystemExit("eval/test.jsonl does not match eval/manifest.json. Do not score a changed frozen test.")
    probe = subprocess.run(
        ["git", "-C", str(root), "show", f"{PROTOCOL}:eval/test.jsonl"],
        capture_output=True,
        check=False,
    )
    tag = "missing"
    if probe.returncode == 0:
        if hashlib.sha256(probe.stdout).hexdigest() != digest:
            raise SystemExit(f"eval/test.jsonl differs from tag {PROTOCOL}. Do not score a changed frozen test.")
        tag = "matches"
    return {"protocol": PROTOCOL, "test_sha256": digest, "tag": tag}


def output_path(root: Path, team: str, model: str) -> Path:
    if not model or not all(char.isalnum() or char in "._-" for char in model):
        raise SystemExit(f"refusing model name {model!r}")
    return root / API_RESULTS / model / f"team_{team}.json"


def is_measured(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "measured" and payload.get("kind") == "model_eval"


def _runner(team: str):
    if team == "v2":
        from orbit_agents.team_v2 import run_team_v2

        return lambda task, policy, workspace: run_team_v2(task, policy, workspace)
    return lambda task, policy, workspace: run_team(task, PolicyBackend(policy), workspace)


def score_tasks(tasks: list[dict], *, team: str, provider: str, model: str | None, workspace: Path, policy_kwargs: dict, policy=None):
    """Run every task once. Returns (rows, policy, stop) where stop is None or {reason, detail}."""
    runner = _runner(team)
    shared = policy
    if shared is None and provider not in PER_TASK_PROVIDERS:
        shared = make_policy(provider, model, team=team, **policy_kwargs)
        if hasattr(shared, "preflight"):
            try:
                shared.preflight()
            except shared.stops as exc:
                return [], shared, {"reason": exc.reason, "detail": exc.detail, "after_task": None}
    stops = getattr(shared, "stops", ())
    rows = []
    for task in tasks:
        current = shared or make_policy(provider, model, task=task, team=team)
        try:
            result = runner(task, current, workspace / task["id"])
        except stops as exc:
            return rows, shared, {"reason": exc.reason, "detail": exc.detail, "after_task": rows[-1]["id"] if rows else None}
        rows.append(score_task(task, result))
    return rows, shared, None


def write_comparison(folder: Path) -> Path | None:
    """v2 minus v1 on the same model, when both are measured. Seeded like eval.compare."""
    left, right = folder / "team_v2.json", folder / "team_v1.json"
    if not (is_measured(left) and is_measured(right)):
        return None
    v2 = json.loads(left.read_text(encoding="utf-8"))["per_task"]
    v1 = json.loads(right.read_text(encoding="utf-8"))["per_task"]
    if [row["id"] for row in v2] != [row["id"] for row in v1]:
        raise SystemExit("team_v1 and team_v2 task order differ")
    block = {}
    for label, field in (("task_success_rate", "success"), ("mean_steps", "n_steps"), ("schema_valid_rate", "schema_valid_rate")):
        treatment = [float(row[field]) for row in v2]
        control = [float(row[field]) for row in v1]
        block[label] = bootstrap_mean_difference(treatment, control)
    report = {"status": "measured", "kind": "comparison", "n_resamples": 10000, "seed": 0, "pairs": {"team_v2_minus_team_v1": block}}
    path = folder / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score one team once on the frozen test")
    parser.add_argument("--team", choices=tuple(TEAMS), required=True)
    parser.add_argument("--provider", choices=("anthropic", "transformers", "ollama", "llamacpp"), default="anthropic")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-requests", type=int, default=None, help="Hard request cap; defaults to the computed bound")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument("--requests-per-minute", type=float, default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Check the protocol and print the request bound; no key, no request")
    return parser


def main(argv: list[str] | None = None, *, policy=None) -> dict:
    args = build_parser().parse_args(argv)
    frozen = verify_frozen_test(args.root)
    tasks = read_jsonl(args.root / "eval" / "test.jsonl")
    bound = request_bound(len(tasks))
    max_requests = args.max_requests or bound["hard_cap"]
    if max_requests < 1 or args.max_cost_usd < 0:
        raise SystemExit("--max-requests must be at least 1 and --max-cost-usd zero or more")
    out = output_path(args.root, args.team, args.model)
    plan = {
        "event": "plan",
        "team": args.team,
        "label": TEAMS[args.team],
        "provider": args.provider,
        "model": args.model,
        "frozen_test": frozen,
        "n_tasks": len(tasks),
        "requests": bound,
        "max_requests": max_requests,
        "max_cost_usd": args.max_cost_usd if args.provider == "anthropic" else None,
        "out": str(out.relative_to(args.root)),
        "already_measured": is_measured(out),
    }
    print(json.dumps(plan), flush=True)
    if plan["already_measured"]:
        raise SystemExit(f"{plan['out']} is already measured. The frozen test is scored once per config.")
    if args.dry_run:
        return plan
    previous = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    kwargs: dict = {}
    if args.provider == "anthropic":
        kwargs = {"max_requests": max_requests, "max_cost_usd": args.max_cost_usd, "requests_per_minute": args.requests_per_minute}
    elif args.provider == "transformers":
        kwargs = {"adapter": args.adapter, "max_new_tokens": args.max_new_tokens}
    workspace = args.root / "traces" / "eval" / f"team_{args.team}" / args.provider
    rows, used, stop = score_tasks(
        tasks, team=args.team, provider=args.provider, model=args.model, workspace=workspace, policy_kwargs=kwargs, policy=policy
    )
    report = {
        "kind": "model_eval",
        "split": "test",
        "protocol": PROTOCOL,
        "test_sha256": frozen["test_sha256"],
        "harness": "v3",
        "n_roles": 6 if args.team == "v2" else 3,
        "mode": f"team_{args.team}",
        "team": args.team,
        "team_label": TEAMS[args.team],
        "provider": args.provider,
        "backend": args.provider,
        "model": args.model,
        "adapter": args.adapter,
        "max_requests": max_requests,
        "max_cost_usd": kwargs.get("max_cost_usd"),
        "usage": used.usage() if used is not None else {},
        "previous_stops": int(previous.get("previous_stops") or 0) + (1 if previous.get("status") == "stopped" else 0),
    }
    if stop is not None:
        report.update({"status": "stopped", "stop": stop, "n_tasks_finished": len(rows), "per_task": None})
        report["note"] = "Stopped before every task finished, so there is no score. Nothing here is a measurement."
    else:
        report.update({"status": "measured", **summarize(rows), "per_task": rows})
        report["note"] = "One score on the frozen test. The policy saw only the role prompts and the task."
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    comparison = write_comparison(out.parent) if stop is None else None
    print(
        json.dumps(
            {
                "event": "done",
                "status": report["status"],
                "out": plan["out"],
                "n_success": report.get("n_success"),
                "n_tasks": report.get("n_tasks"),
                "comparison": None if comparison is None else str(comparison.relative_to(args.root)),
                "usage": report["usage"],
            }
        ),
        flush=True,
    )
    if stop is not None:
        raise SystemExit(2)
    return report


if __name__ == "__main__":
    main()
