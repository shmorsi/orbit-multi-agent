"""Compare measured model runs. Pending files do not grow confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbit_agents.eval.metrics import bootstrap_mean_difference
from orbit_agents.eval.run import locked_result_reason

KEYS = {
    "task_success_rate": "success",
    "mean_steps": "n_steps",
    "schema_valid_rate": "schema_valid_rate",
}


def _rows(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "measured" or payload.get("kind") != "model_eval":
        raise SystemExit(f"{path} is not a measured model eval")
    return payload, payload["per_task"]


BASE_PAIRS = (
    ("team_base", "single_base"),
    ("team_lora", "team_base"),
    ("team_lora", "single_base"),
)
DISTILL_PAIRS = (
    ("team_distill", "team_base"),
    ("team_distill", "team_lora"),
    ("team_distill", "single_base"),
)


def compare(paths: dict[str, Path], *, n_resamples: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap of measured runs. Pending files are not given an interval."""
    for name in ("single_base", "team_base", "team_lora"):
        if name not in paths:
            raise SystemExit(f"missing {name}")
    loaded = {name: _rows(path) for name, path in paths.items()}
    order = [row["id"] for row in loaded["team_base"][1]]
    for name, (_payload, rows) in loaded.items():
        if [row["id"] for row in rows] != order:
            raise SystemExit(f"{name} task order does not match the team base run")
    report = {"status": "measured", "kind": "comparison", "n_resamples": n_resamples, "seed": seed, "pairs": {}}
    pairs = list(BASE_PAIRS)
    if "team_distill" in loaded:
        pairs.extend(DISTILL_PAIRS)
    for left, right in pairs:
        left_rows = {row["id"]: row for row in loaded[left][1]}
        right_rows = {row["id"]: row for row in loaded[right][1]}
        block = {}
        for label, field in KEYS.items():
            treatment = [float(left_rows[task_id][field]) for task_id in order]
            control = [float(right_rows[task_id][field]) for task_id in order]
            block[label] = bootstrap_mean_difference(treatment, control, n_resamples=n_resamples, seed=seed)
        report["pairs"][f"{left}_minus_{right}"] = block
    return report


def main(argv: list[str] | None = None) -> None:
    """Write a comparison JSON for measured runs, unless the output path is locked."""
    parser = argparse.ArgumentParser(description="Bootstrap measured Orbit runs")
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--team", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--distill", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {"single_base": args.single, "team_base": args.team, "team_lora": args.lora}
    if args.distill is not None:
        paths["team_distill"] = args.distill
    locked = locked_result_reason(args.out)
    if locked:
        raise SystemExit(locked)
    try:
        report = compare(paths)
    except SystemExit as exc:
        report = {
            "status": "pending",
            "kind": "comparison",
            "metrics": None,
            "reason": str(exc),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
