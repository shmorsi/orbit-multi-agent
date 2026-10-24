"""Write mission/board.js from the measured v2 files.

The page is static. GitHub Pages and a local open of mission/ both load this
script. Team distill is a slot: it is not scored until that run exists.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from orbit_agents.eval.metrics import measurements_match

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "v2_qwen1.5b"
OUT = ROOT / "mission" / "board.js"

RUNS = (
    ("single_base", "Single base"),
    ("team_base", "Team base"),
    ("team_lora", "Team LoRA"),
)


def _read(name: str) -> dict:
    return json.loads((RESULT / name).read_text(encoding="utf-8"))


def bootstrap_mean(values: list[float], *, n_resamples: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    point = sum(values) / len(values)
    rng = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _index in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    low_index = int((alpha / 2) * n_resamples)
    high_index = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    return {
        "point": point,
        "low": means[low_index],
        "high": means[high_index],
        "n": count,
        "n_resamples": n_resamples,
        "seed": seed,
        "method": "percentile_bootstrap",
    }


def _series(rows: list[dict], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row[field]
        if value is None:
            continue
        values.append(float(value))
    return values


def build_board() -> dict:
    comparison = _read("comparison.json")
    loss_rows = [
        json.loads(line)
        for line in (RESULT / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runs = []
    per_task = {}
    for key, label in RUNS:
        payload = _read(f"{key}.json")
        per_task[key] = payload["per_task"]
        success = _series(payload["per_task"], "success")
        schema = _series(payload["per_task"], "schema_valid_rate")
        runs.append(
            {
                "id": key,
                "label": label,
                "pending": False,
                "n_success": payload["n_success"],
                "n_tasks": payload["n_tasks"],
                "task_success_rate": payload["task_success_rate"],
                "schema_valid_rate": payload["schema_valid_rate"],
                "n_traps": payload["n_traps"],
                "n_correct_refusals": payload["n_correct_refusals"],
                "success_interval": bootstrap_mean(success, seed=0),
                "schema_interval": bootstrap_mean(schema, seed=0),
            }
        )
    runs.append(
        {
            "id": "team_distill",
            "label": "Team distill",
            "pending": True,
            "n_success": None,
            "n_tasks": 12,
            "task_success_rate": None,
            "schema_valid_rate": None,
            "n_traps": 2,
            "n_correct_refusals": None,
            "success_interval": None,
            "schema_interval": None,
        }
    )
    trap_ids = [row["id"] for row in per_task["team_base"] if row["expect_refusal"]]
    traps = []
    for task_id in trap_ids:
        outcomes = {}
        for key, _label in RUNS:
            match = next(row for row in per_task[key] if row["id"] == task_id)
            outcomes[key] = bool(match["correct_refusal"])
        outcomes["team_distill"] = None
        traps.append({"id": task_id, "outcomes": outcomes})
    pair = comparison["pairs"]["team_lora_minus_team_base"]
    return {
        "protocol": "team-eval-v1",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "honesty": "Task-success intervals include zero. The gain is not statistically significant.",
        "runs": runs,
        "paired": {
            "label": "Team LoRA minus team base",
            "task_success_rate": pair["task_success_rate"],
            "schema_valid_rate": pair["schema_valid_rate"],
            "distill": None,
        },
        "loss": [{"step": row["step"], "loss": row["loss"]} for row in loss_rows],
        "traps": traps,
    }


def render(payload: dict) -> str:
    return "window.ORBIT_BOARD = " + json.dumps(payload, indent=2) + ";\n"


def _committed_matches(payload: dict, text: str) -> bool:
    if not OUT.is_file():
        return False
    current = OUT.read_text(encoding="utf-8")
    if current == text:
        return True
    prefix = "window.ORBIT_BOARD = "
    suffix = ";\n"
    if not current.startswith(prefix) or not current.endswith(suffix):
        return False
    try:
        committed = json.loads(current[len(prefix) : -len(suffix)])
    except json.JSONDecodeError:
        return False
    return measurements_match(committed, payload)


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    payload = build_board()
    text = render(payload)
    if _committed_matches(payload, text):
        if "--check" not in args:
            print(json.dumps({"out": str(OUT), "unchanged": True}))
        return
    if "--check" in args:
        raise SystemExit("mission/board.js is stale; run python mission/build_board.py")
    OUT.write_text(text, encoding="utf-8")
    print(json.dumps({"out": str(OUT), "runs": [run["id"] for run in payload["runs"]]}))


if __name__ == "__main__":
    main()
