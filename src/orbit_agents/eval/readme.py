"""Write the two generated README blocks from the committed results files.

RESULTS_START/RESULTS_END holds Orbit Code on code-eval-v2. FINETUNE_START/FINETUNE_END
holds the harness-v3 Qwen table, the API-policy teams, the archived v1 and v2 runs,
and the pending slots. Paired bootstrap intervals render as compact tables, one row
per pair, each cell saying whether the interval includes zero.

Harness v3 rows appear only for files under results/v3_qwen1.5b/ that are measured
model evals; a missing or pending file renders as "not scored". Paired intervals
appear only when that directory's comparison.json is measured.

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
FINE_START = "<!-- FINETUNE_START -->"
FINE_END = "<!-- FINETUNE_END -->"
BLOCK_RE = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
FINE_RE = re.compile(re.escape(FINE_START) + r".*?" + re.escape(FINE_END), re.S)

RUNS = (
    ("single_base", "Single base"),
    ("team_base", "Team base"),
    ("team_lora", "Team plus adapter"),
    ("team_distill", "Team distill"),
)
V3_PAIRS = (
    ("team_base_minus_single_base", "team base minus single base"),
    ("team_lora_minus_single_base", "team plus adapter minus single base"),
    ("team_lora_minus_team_base", "team plus adapter minus team base"),
    ("team_distill_minus_single_base", "team distill minus single base"),
    ("team_distill_minus_team_base", "team distill minus team base"),
    ("team_distill_minus_team_lora", "team distill minus team plus adapter"),
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


def _zero_word(interval: dict) -> str:
    """Where a paired interval sits relative to zero. An interval on zero is not a difference."""
    if interval["low"] > 0:
        return "above zero"
    if interval["high"] < 0:
        return "below zero"
    return "includes zero"


def _significance(interval: dict) -> str:
    """Say whether a paired interval excludes zero. An interval on zero is not a difference."""
    if interval["low"] > 0:
        return "That interval excludes zero and is above zero."
    if interval["high"] < 0:
        return "That interval excludes zero and is below zero."
    return "That interval includes zero, so the difference is not significant."


def _table_head() -> list[str]:
    return [
        "| Run | Successes | Task-success rate | Schema-valid rate | Trap refusals |",
        "| --- | --- | --- | --- | --- |",
    ]


def _pair_cell(interval: dict) -> str:
    return f"{_interval(interval)}, {_zero_word(interval)}"


def _pair_table(rows: list[tuple[str, dict]]) -> list[str]:
    """One row per paired difference: task success and schema-valid rate, each with its interval."""
    lines = [
        "| Paired difference | Task success | Schema-valid rate |",
        "| --- | --- | --- |",
    ]
    for label, pair in rows:
        lines.append(f"| {label} | {_pair_cell(pair['task_success_rate'])} | {_pair_cell(pair['schema_valid_rate'])} |")
    return lines


def _method(root: Path) -> str:
    """How the intervals were computed, read from the first measured comparison file."""
    for folder in ("v3_qwen1.5b", "v2_qwen1.5b"):
        comparison = _read(root / "results" / folder / "comparison.json")
        if not comparison or comparison.get("status") != "measured":
            continue
        for pair in comparison["pairs"].values():
            interval = pair["task_success_rate"]
            return (
                f"Intervals are a paired percentile bootstrap (alpha {interval['alpha']:g}, "
                f"{display_number(interval['n_resamples'])} resamples). "
            )
    return ""


def _v2(root: Path, distill_v3: bool) -> list[str]:
    folder = root / "results" / "v2_qwen1.5b"
    run = _read(folder / "RUN.json")
    loss = _read(folder / "loss_summary.json")
    comparison = _read(folder / "comparison.json")
    runs = {key: _read(folder / f"{key}.json") for key, _label in RUNS[:3]}
    pair = comparison["pairs"]["team_lora_minus_team_base"]
    lines = [
        "#### Archived: harness v2",
        "",
        f"Qwen2.5-1.5B-Instruct, QLoRA, {run['n_train_dialogs']} train and {run['n_tune_dialogs']} tune dialogues. "
        f"Logged loss went from {display_number(loss['first_logged_loss'])} at step {loss['first_logged_step']} to "
        f"{display_number(loss['last_logged_loss'])} at step {loss['last_logged_step']}. Files: `results/v2_qwen1.5b/`. "
        "Harness v3 changed the tools and the gate, so this table is a record, not a baseline for the tables above.",
        "",
        *_table_head(),
    ]
    for key, label in RUNS[:3]:
        lines.append(_row(label, runs[key]))
    lines.append("| Team distill | not scored | | | |")
    lines.extend(["", *_pair_table([("Team plus adapter minus team base", pair)]), ""])
    tail = (
        "Team distill was not scored under v2. Its only score is in the harness v3 table."
        if distill_v3
        else "Team distill is a fourth config. It has not been scored."
    )
    lines.append(
        "Fine-tuning moved the schema-valid rate, not task success, and the adapter missed a trap: correct refusals "
        f"went from {runs['team_base']['n_correct_refusals']} to {runs['team_lora']['n_correct_refusals']}. " + tail
    )
    lines.extend(
        [
            "",
            "![Logged training loss for the measured Qwen2.5-1.5B QLoRA run. The curve is the committed file.]"
            "(results/v2_qwen1.5b/loss_curve.png)",
            "",
            _v1(root),
        ]
    )
    return lines


def _v1(root: Path) -> str:
    folder = root / "results" / "v1_qwen0.5b"
    runs = [_read(folder / f"{key}.json") for key, _label in RUNS[:3]]
    loss = _read(folder / "loss_summary.json")
    successes = ", ".join(f"{run['n_success']} of {run['n_tasks']}" for run in runs)
    schema = ", ".join(display_number(run["schema_valid_rate"]) for run in runs)
    return (
        f"The earlier 0.5B run (`results/v1_qwen0.5b/`): single base, team base, and team plus adapter scored "
        f"{successes}, with schema-valid rates {schema}. Logged loss went from "
        f"{display_number(loss['first_logged_loss'])} to {display_number(loss['last_logged_loss'])}. "
        "Its task-success interval also includes zero."
    )


def _v3(root: Path) -> list[str]:
    folder = root / "results" / "v3_qwen1.5b"
    runs = {key: _read(folder / f"{key}.json") for key, _label in RUNS}
    harness = (
        "Harness v3 changed the tools and the gate: search lists tables and logs, python can read a bundled file by "
        "path, a count printed by python or equal to the matching lines in the cited text stays grounded, and the "
        "critic must return one JSON object."
    )
    if not any(_measured(payload) for payload in runs.values()):
        return [
            "#### Qwen2.5-1.5B, harness v3",
            "",
            harness + " single_base, team_base, team_lora, and team_distill must be scored again on the frozen test "
            "under v3 before any comparison. Those v3 scores are not in yet.",
        ]
    lines = [
        "#### Qwen2.5-1.5B, harness v3 (current)",
        "",
        harness + " The four configs were scored again under v3, once each. Files: `results/v3_qwen1.5b/`.",
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
    rows = [
        (label[:1].upper() + label[1:], comparison["pairs"][key])
        for key, label in V3_PAIRS
        if key in comparison["pairs"]
    ]
    lines.extend(_pair_table(rows))
    lines.extend(
        [
            "",
            "No task-success interval excludes zero."
            if all(_zero_word(pair["task_success_rate"]) == "includes zero" for _label, pair in rows)
            else "At least one task-success interval excludes zero.",
        ]
    )
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
                f"#### API policy `{model}`, three roles against six",
                "",
                (
                    "Harness v3, same frozen test, same model and role prompts for every team, each team scored once. "
                    f"Files: `results/v3_api/{model_dir.name}/`."
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
            lines.extend(_pair_table([("v2 team minus v1 team", pair)]))
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
    lines = [
        "",
        "Pending slots. A row here has no number until its measured file exists.",
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
    "The earlier split `code-eval-v1` is superseded: its team pipeline refused a task when the planner reply failed "
    "the schema, and a critic accept could end a red build before the coder revised. That record stays in "
    "`results/code/claude-haiku-5-5/`. It is not the current score."
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
        "Orbit Code on the frozen split `code-eval-v2`: each config scored once, graded by hidden tests the agents "
        "never see. A row is filled only from a measured file under `results/code_v2/`; anything else is pending.",
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
    pair_lines: list[str] = []
    seen_dirs: list[Path] = []
    for payload in measured:
        tokens, cost = _code_cells(payload)
        lines.append(
            f"| {payload.get('config_label', payload['config'])} | `{payload['model']}` | "
            f"{display_number(payload['n_pass'])} of {display_number(payload['n_tasks'])} | "
            f"{display_number(payload['tests_passed_rate'])} | "
            f"{display_number(payload['n_reviewer_rejections'])} of {display_number(payload['n_tasks'])} | "
            f"{tokens} | {cost} |"
        )
    if folder.is_dir():
        seen_dirs = sorted(path for path in folder.iterdir() if path.is_dir())
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
    for model_dir in seen_dirs:
        single = _read(model_dir / "single.json")
        team = _read(model_dir / "team.json")
        if _measured_code(single) and _measured_code(team) and single.get("provider") != "transformers":
            pair_lines.extend(_code_pair_lines(model_dir))
    if pair_lines:
        lines.extend(["", *pair_lines])
    lines.extend(["", _V1_SUPERSEDED])
    return lines


def _code_pair_lines(model_dir: Path) -> list[str]:
    """Paired pass@1 interval for one model directory, or an explicit non-claim."""
    comparison = _read(model_dir / "comparison.json")
    pair = None
    if comparison and comparison.get("status") == "measured":
        pair = comparison.get("pairs", {}).get("team_minus_single", {}).get("pass_at_1")
    if not pair:
        return [
            "The paired comparison of team with reviewer minus single agent is not in this directory, "
            "so no pass@1 difference is claimed."
        ]
    return [
        "The paired bootstrap for pass@1, team with reviewer minus single agent, is "
        f"{_interval(pair)}. {_significance(pair)} n is {display_number(pair['n'])}."
    ]


def render_results(root: Path = ROOT) -> str:
    """The headline block: Orbit Code on code-eval-v2."""
    lines = [START, *_code(root), END]
    return "\n".join(lines)


def render_finetune(root: Path = ROOT) -> str:
    """Harness-v3 Qwen table, the API teams, then the archived harness-v2 runs and the pending slots."""
    distill = _measured(_read(root / "results" / "v3_qwen1.5b" / "team_distill.json"))
    base = _read(root / "results" / "v2_qwen1.5b" / "team_base.json")
    lines = [
        FINE_START,
        (
            "Committed values from `results/`, shown to three decimal places; the JSON files keep full precision. "
            f"Every config was scored once on the frozen test `team-eval-v1`: {base['n_tasks']} tasks, "
            f"{base['n_traps']} of them traps. {_method(root)}An interval that includes zero is not a difference."
        ),
        "",
        *_v3(root),
        *_api(root),
        "",
        *_v2(root, distill),
        *_round_table(root),
        *_pending(root),
        FINE_END,
    ]
    return "\n".join(lines)


def render(root: Path = ROOT) -> str:
    """Render both generated blocks from the committed result files."""
    return render_results(root) + "\n" + render_finetune(root)


def updated_readme(root: Path = ROOT) -> str:
    """Return README.md with both generated blocks replaced."""
    text = (root / "README.md").read_text(encoding="utf-8")
    if not BLOCK_RE.search(text):
        raise SystemExit("README.md has no RESULTS_START/RESULTS_END block")
    if not FINE_RE.search(text):
        raise SystemExit("README.md has no FINETUNE_START/FINETUNE_END block")
    text = BLOCK_RE.sub(lambda _match: render_results(root), text, count=1)
    return FINE_RE.sub(lambda _match: render_finetune(root), text, count=1)


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
