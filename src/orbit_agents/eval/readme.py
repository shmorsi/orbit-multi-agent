"""Write the README results block from the committed results files.

The block sits between the RESULTS_START and RESULTS_END markers. The v1 and v2
runs are archived and render the same text every time. Harness v3 rows appear
only for files under results/v3_qwen1.5b/ that are measured model evals; a
missing or pending file renders as "not scored". Paired intervals appear only
when that directory's comparison.json is measured.

    python -m orbit_agents.eval.readme          # rewrite the block
    python -m orbit_agents.eval.readme --check  # fail if the block is stale
"""

from __future__ import annotations

import argparse
import json
import re
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


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _measured(payload: dict | None) -> bool:
    return bool(payload) and payload.get("status") == "measured" and payload.get("kind") == "model_eval"


def _row(label: str, payload: dict) -> str:
    return (
        f"| {label} | {payload['n_success']} of {payload['n_tasks']} | {payload['task_success_rate']} "
        f"| {payload['schema_valid_rate']} | {payload['n_correct_refusals']} of {payload['n_traps']} |"
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
        f"from {loss['first_logged_loss']} at step {loss['first_logged_step']} to {loss['last_logged_loss']} at step "
        f"{loss['last_logged_step']}. Task success did not clear zero.",
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
            f"bootstrap for task success, adapter minus team base, is {success['low']} to {success['high']} (point "
            f"{success['point']}). That interval includes zero, so the task-success gain is not statistically "
            f"significant. The same pair's schema-valid interval is {schema['low']} to {schema['high']} (point "
            f"{schema['point']}). Schema-valid rate moved. Task success did not, on this 12-task set.",
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
    schema = ", ".join(str(run["schema_valid_rate"]) for run in runs[:2])
    return (
        f"The earlier 0.5B run is in `results/v1_qwen0.5b/`. Successes were {successes}, then "
        f"{runs[2]['n_success']} of {runs[2]['n_tasks']}, with schema-valid rates {schema}, and "
        f"{runs[2]['schema_valid_rate']}. Logged loss went from {loss['first_logged_loss']} to "
        f"{loss['last_logged_loss']}. Its task-success interval also failed to clear zero. Fine-tuning taught the format."
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
            f"The paired bootstrap for task success, {label}, is {success['low']} to {success['high']} (point "
            f"{success['point']}). {_zero(success)} Its schema-valid interval is {schema['low']} to {schema['high']} "
            f"(point {schema['point']})."
        )
    lines.append(" ".join(sentences))
    return lines


def render(root: Path = ROOT) -> str:
    distill = _measured(_read(root / "results" / "v3_qwen1.5b" / "team_distill.json"))
    lines = [START, *_v2(root, distill), "", *_v3(root), END]
    return "\n".join(lines)


def updated_readme(root: Path = ROOT) -> str:
    text = (root / "README.md").read_text(encoding="utf-8")
    if not BLOCK_RE.search(text):
        raise SystemExit("README.md has no RESULTS_START/RESULTS_END block")
    block = render(root)
    return BLOCK_RE.sub(lambda _match: block, text, count=1)


def main(argv: list[str] | None = None) -> None:
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
