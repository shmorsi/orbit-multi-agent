"""Write the README results block from the committed results files.

The block sits between the RESULTS_START and RESULTS_END markers. The v1 and v2
runs are archived and render the same text every time. Harness v3 rows appear
only for files under results/v3_qwen1.5b/ that are measured model evals; a
missing or pending file renders as "not scored". Paired intervals appear only
when that directory's comparison.json is measured.

API-policy rows (results/v3_api/<model>/team_v1.json, team_v2.json, and
team_v3.json, written once each by `python -m orbit_agents.eval.score_team`)
appear only for measured files. With none measured, the "v2 team (6 roles)"
row does not exist. The round-table team stays "not scored" until team_v3.json
is a measured model eval.

    python -m orbit_agents.eval.readme          # rewrite the block
    python -m orbit_agents.eval.readme --check  # fail if the block is stale
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from orbit_agents.paths import ROOT

START = "<!-- RESULTS_START -->"
END = "<!-- RESULTS_END -->"
BLOCK_RE = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)

RUNS = (
    ("single_base", "Single base"),
    ("team_base", "Team base"),
    ("team_lora", "Team plus adapter"),
    ("team_distill", "Team distill"),
)
V3_PAIRS = (
    ("team_distill_minus_team_base", "team distill minus team base"),
    ("team_distill_minus_team_lora", "team distill minus team plus adapter"),
    ("team_lora_minus_team_base", "team plus adapter minus team base"),
)
PENDING_API = (
    ("team_v1", "Claude policy, three-role team, harness v3", "results/v3_api/<model>/team_v1.json"),
    ("team_v2", "Claude policy, six-role team, harness v3", "results/v3_api/<model>/team_v2.json"),
    ("team_v3", "Claude policy, round table, harness v3", "results/v3_api/<model>/team_v3.json"),
)


def display_number(value: int | float) -> str:
    """Show an int as digits and a float to three decimal places.

    This is display rounding of a committed value. The JSON file keeps the full number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"not a measured number: {type(value).__name__}")
    if isinstance(value, int):
        return str(value)
    quantized = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return format(quantized, "f")


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _measured(payload: dict | None) -> bool:
    return bool(payload) and payload.get("status") == "measured" and payload.get("kind") == "model_eval"


def _row(label: str, payload: dict) -> str:
    return (
        f"| {label} | {payload['n_success']} of {payload['n_tasks']} | {display_number(payload['task_success_rate'])} "
        f"| {display_number(payload['schema_valid_rate'])} | {payload['n_correct_refusals']} of {payload['n_traps']} |"
    )


def _interval(interval: dict) -> str:
    return (
        f"{display_number(interval['low'])} to {display_number(interval['high'])} "
        f"(point {display_number(interval['point'])})"
    )


def _zero(interval: dict) -> str:
    if interval["low"] > 0:
        return "That interval is above zero."
    if interval["high"] < 0:
        return "That interval is below zero."
    return "That interval includes zero."


def _table_head() -> list[str]:
    return [
        "| Run | Successes | Task-success rate | Schema-valid rate | Trap refusals |",
        "| --- | --- | --- | --- | --- |",
    ]


def _v2(root: Path, distill_v3: bool) -> list[str]:
    folder = root / "results" / "v2_qwen1.5b"
    run = _read(folder / "RUN.json")
    loss = _read(folder / "loss_summary.json")
    comparison = _read(folder / "comparison.json")
    runs = {key: _read(folder / f"{key}.json") for key, _label in RUNS[:3]}
    pair = comparison["pairs"]["team_lora_minus_team_base"]
    success = pair["task_success_rate"]
    schema = pair["schema_valid_rate"]
    lines = [
        f"Measured once on the frozen test `team-eval-v1`, Qwen2.5-1.5B-Instruct, QLoRA, {run['n_train_dialogs']} "
        f"train dialogues and {run['n_tune_dialogs']} tune dialogues. Files: `results/v2_qwen1.5b/`. Logged loss went "
        f"from {display_number(loss['first_logged_loss'])} at step {loss['first_logged_step']} to "
        f"{display_number(loss['last_logged_loss'])} at step {loss['last_logged_step']}. Task success did not clear zero.",
        "",
        *_table_head(),
    ]
    for key, label in RUNS[:3]:
        lines.append(_row(label, runs[key]))
    lines.append("| Team distill | not scored | | | |")
    lines.extend(
        [
            "",
            "The adapter's two answerable successes sit next to a missed trap: correct refusals went from "
            f"{runs['team_base']['n_correct_refusals']} to {runs['team_lora']['n_correct_refusals']}. The paired "
            f"bootstrap for task success, adapter minus team base, is {_interval(success)}. {_zero(success)} "
            f"The adapter is not separated from the base team on task success. The same pair's schema-valid interval "
            f"is {_interval(schema)}. {_zero(schema)} Schema-valid rate moved. Task success did not, on this "
            f"{runs['team_base']['n_tasks']}-task set.",
            "",
            "![Logged training loss for the measured Qwen2.5-1.5B QLoRA run. The curve is the committed file.]"
            "(results/v2_qwen1.5b/loss_curve.png)",
            "",
        ]
    )
    lines.append(_v1(root))
    lines.append("")
    tail = (
        "Team distill is a fourth config, scored only under harness v3 below."
        if distill_v3
        else "Team distill is a fourth config. It has not been scored."
    )
    lines.append(
        "The placeholder files at the top of `results/` stay pending. A harness file in `results/` is a system check "
        "of the scripted tools, not a model score. " + tail
    )
    return lines


def _v1(root: Path) -> str:
    folder = root / "results" / "v1_qwen0.5b"
    runs = [_read(folder / f"{key}.json") for key, _label in RUNS[:3]]
    loss = _read(folder / "loss_summary.json")
    successes = ", ".join(f"{run['n_success']} of {run['n_tasks']}" for run in runs[:2])
    schema = ", ".join(display_number(run["schema_valid_rate"]) for run in runs[:2])
    return (
        f"The earlier 0.5B run is in `results/v1_qwen0.5b/`. Successes were {successes}, then "
        f"{runs[2]['n_success']} of {runs[2]['n_tasks']}, with schema-valid rates {schema}, and "
        f"{display_number(runs[2]['schema_valid_rate'])}. Logged loss went from "
        f"{display_number(loss['first_logged_loss'])} to {display_number(loss['last_logged_loss'])}. "
        "Its task-success interval also failed to clear zero. Fine-tuning taught the format."
    )


def _v3(root: Path) -> list[str]:
    folder = root / "results" / "v3_qwen1.5b"
    runs = {key: _read(folder / f"{key}.json") for key, _label in RUNS}
    intro = (
        "The harness changed in v3. Search now lists tables and logs, python can read a bundled file by path, a count "
        "printed by python or equal to the matching lines in the cited text stays grounded, and the critic must return "
        "one JSON object. The v2 numbers above stay archived in `results/v2_qwen1.5b/`. single_base, team_base, "
        "team_lora, and team_distill must be scored again on the frozen test under v3 before any comparison."
    )
    if not any(_measured(payload) for payload in runs.values()):
        return [intro + " Those v3 scores are not in yet."]
    lines = [
        intro,
        "",
        "Harness v3, frozen test `team-eval-v1`, each run scored once. Files: `results/v3_qwen1.5b/`.",
        "",
        *_table_head(),
    ]
    for key, label in RUNS:
        payload = runs[key]
        lines.append(_row(label, payload) if _measured(payload) else f"| {label} | not scored | | | |")
    comparison = _read(folder / "comparison.json")
    lines.append("")
    if not comparison or comparison.get("status") != "measured":
        lines.append(
            "The v3 paired comparison needs single_base, team_base, team_lora, and team_distill measured under v3. "
            "It is not in yet, so no v3 difference is claimed."
        )
        return lines
    sentences = []
    for key, label in V3_PAIRS:
        pair = comparison["pairs"].get(key)
        if pair is None:
            continue
        success = pair["task_success_rate"]
        schema = pair["schema_valid_rate"]
        sentences.append(
            f"The paired bootstrap for task success, {label}, is {_interval(success)}. {_zero(success)} "
            f"Its schema-valid interval is {_interval(schema)}."
        )
    lines.append(" ".join(sentences))
    return lines


API_TEAMS = (
    ("team_v1", "v1 team (3 roles)"),
    ("team_v2", "v2 team (6 roles)"),
    ("team_v3", "Round-table team"),
)


def _round_table(root: Path) -> list[str]:
    """One sentence when no measured round-table file exists. No figure is invented."""
    folder = root / "results" / "v3_api"
    if folder.is_dir():
        for model_dir in folder.iterdir():
            if model_dir.is_dir() and _measured(_read(model_dir / "team_v3.json")):
                return []
    return ["", "The round-table team is not scored."]


def _api(root: Path) -> list[str]:
    folder = root / "results" / "v3_api"
    if not folder.is_dir():
        return []
    lines: list[str] = []
    for model_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
        runs = {key: _read(model_dir / f"{key}.json") for key, _label in API_TEAMS}
        if not any(_measured(payload) for payload in runs.values()):
            continue
        model = next(payload["model"] for payload in runs.values() if _measured(payload))
        lines.extend(
            [
                "",
                (
                    f"API policy `{model}`, harness v3, frozen test `team-eval-v1`, each team scored once with the same "
                    f"model and role prompts only. Files: `results/v3_api/{model_dir.name}/`."
                ),
                "",
                *_table_head(),
            ]
        )
        for key, label in API_TEAMS:
            payload = runs[key]
            lines.append(_row(label, payload) if _measured(payload) else f"| {label} | not scored | | | |")
        comparison = _read(model_dir / "comparison.json")
        lines.append("")
        pair = None
        if comparison and comparison.get("status") == "measured":
            pair = comparison.get("pairs", {}).get("team_v2_minus_team_v1")
        if pair is None:
            lines.append(
                "The paired comparison needs both teams measured on this model. It is not in yet, so no difference is claimed."
            )
        else:
            success = pair["task_success_rate"]
            schema = pair["schema_valid_rate"]
            lines.append(
                f"The paired bootstrap for task success, v2 team minus v1 team, is {_interval(success)}. "
                f"{_zero(success)} Its schema-valid interval is {_interval(schema)}. {_zero(schema)}"
            )
    return lines


def _pending(root: Path) -> list[str]:
    """Slots with no measured file. The row is a placeholder, not a result."""
    slots: list[tuple[str, str]] = []
    folder = root / "results" / "v3_qwen1.5b"
    for key, label in RUNS:
        if _measured(_read(folder / f"{key}.json")):
            continue
        name = f"{label}, harness v3"
        if key == "team_distill":
            name = "Team distill, QLoRA on a T4, harness v3"
        slots.append((name, f"results/v3_qwen1.5b/{key}.json"))
    api = root / "results" / "v3_api"
    seen = {key: False for key, _label, _path in PENDING_API}
    if api.is_dir():
        for model_dir in api.iterdir():
            if not model_dir.is_dir():
                continue
            for key, _label, _path in PENDING_API:
                if _measured(_read(model_dir / f"{key}.json")):
                    seen[key] = True
    for key, label, path in PENDING_API:
        if not seen[key]:
            slots.append((label, path))
    if not slots:
        return ["", "No pending model slot. Every configured score file is measured."]
    if seen["team_v1"] or seen["team_v2"]:
        intro = (
            "Pending slots. A row here is not a result. The round table and the QLoRA distill run on a T4 stay here "
            "until a measured file exists. The v3 rescores of the archived runs stay here for the same reason."
        )
    else:
        intro = (
            "Pending slots. A row here is not a result. Score-v1 and score-v2 on Claude, the round table, and the "
            "QLoRA distill run on a T4, stay here until a measured file exists. The v3 rescores of the archived runs "
            "stay here for the same reason."
        )
    lines = [
        "",
        intro,
        "",
        "| Slot | File | Status |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {label} | `{path}` | pending |" for label, path in slots)
    return lines


CODE_CONFIGS = (
    ("single", "Single agent"),
    ("team", "Team with reviewer"),
)

_V1_SUPERSEDED = (
    "The earlier split `code-eval-v1` was scored and is superseded. "
    "The team pipeline refused a task when the planner reply failed the schema, "
    "and a critic accept stopped a red build before the coder could revise. "
    "Those files stay at `results/code/claude-haiku-5-5/` as the record of that run. "
    "They are not the current score."
)


def _measured_code(payload: dict | None) -> bool:
    return _measured(payload) and payload.get("protocol") == "code-eval-v2"


def _code_cells(payload: dict) -> tuple[str, str]:
    usage = payload.get("usage") or {}
    tokens = ""
    if isinstance(usage.get("input_tokens"), int) and isinstance(usage.get("output_tokens"), int):
        tokens = f"{display_number(usage['input_tokens'])} in, {display_number(usage['output_tokens'])} out"
    cost = usage.get("estimated_cost_usd")
    cost_cell = ""
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        cost_cell = display_number(cost)
    return tokens, cost_cell


def _code(root: Path) -> list[str]:
    """Orbit Code rows for code-eval-v2. Pending slots have no number.

    The code-eval-v1 files under results/code/ are a superseded record. This block
    does not copy their pass rates.
    """
    lines = [
        "Orbit Code, frozen split `code-eval-v2`. A row is filled only from a measured file under `results/code_v2/`. "
        "Anything else is pending, not scored.",
        "",
        "| Config | Model | pass@1 | Tests-passed rate | Reviewer rejections | Tokens | Cost USD |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    folder = root / "results" / "code_v2"
    measured: list[dict] = []
    if folder.is_dir():
        for model_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
            for key, _label in CODE_CONFIGS:
                payload = _read(model_dir / f"{key}.json")
                if _measured_code(payload):
                    measured.append(payload)
    if not measured:
        lines.extend(
            [
                "| Single agent | pending | pending, not scored | | | | |",
                "| Team with reviewer | pending | pending, not scored | | | | |",
                "| Single agent, local Qwen | pending | pending, not scored | | | | |",
            ]
        )
        lines.extend(["", _V1_SUPERSEDED])
        return lines
    for payload in measured:
        tokens, cost = _code_cells(payload)
        lines.append(
            f"| {payload.get('config_label', payload['config'])} | `{payload['model']}` | "
            f"{display_number(payload['pass_at_1'])} | {display_number(payload['tests_passed_rate'])} | "
            f"{display_number(payload['n_reviewer_rejections'])} of {display_number(payload['n_tasks'])} | "
            f"{tokens} | {cost} |"
        )
    have_single = any(item.get("config") == "single" and item.get("provider") != "transformers" for item in measured)
    have_team = any(item.get("config") == "team" for item in measured)
    have_qwen = any(
        item.get("provider") == "transformers" or "qwen" in str(item.get("model", "")).lower() for item in measured
    )
    if not have_single:
        lines.append("| Single agent | pending | pending, not scored | | | | |")
    if not have_team:
        lines.append("| Team with reviewer | pending | pending, not scored | | | | |")
    if not have_qwen:
        lines.append("| Single agent, local Qwen | pending | pending, not scored | | | | |")
    lines.extend(["", _V1_SUPERSEDED])
    return lines


def render(root: Path = ROOT) -> str:
    """Render the Key results block from the committed result files."""
    distill = _measured(_read(root / "results" / "v3_qwen1.5b" / "team_distill.json"))
    lines = [
        START,
        (
            "Rates, losses, and intervals below are the committed values, shown to three decimal places. "
            "The JSON files hold the full values."
        ),
        "",
        *_code(root),
        "",
        *_v2(root, distill),
        "",
        *_v3(root),
        *_round_table(root),
        *_api(root),
        *_pending(root),
        END,
    ]
    return "\n".join(lines)


def updated_readme(root: Path = ROOT) -> str:
    """Return README.md with the Key results block replaced by render()."""
    text = (root / "README.md").read_text(encoding="utf-8")
    if not BLOCK_RE.search(text):
        raise SystemExit("README.md has no RESULTS_START/RESULTS_END block")
    block = render(root)
    return BLOCK_RE.sub(lambda _match: block, text, count=1)


def main(argv: list[str] | None = None) -> None:
    """Rewrite or check the Key results block."""
    parser = argparse.ArgumentParser(description="Write the README results block from results/")
    parser.add_argument("--check", action="store_true", help="Fail if README.md does not match the results files")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    path = args.root / "README.md"
    current = path.read_text(encoding="utf-8")
    text = updated_readme(args.root)
    if args.check:
        if text != current:
            raise SystemExit("README results block is stale; run python -m orbit_agents.eval.readme")
        print(json.dumps({"readme": str(path), "status": "current"}))
        return
    path.write_text(text, encoding="utf-8")
    print(json.dumps({"readme": str(path), "status": "written" if text != current else "unchanged"}))


if __name__ == "__main__":
    main()
