import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from orbit_agents.eval.compare import compare
from orbit_agents.tasks import task_by_id


def test_printer_a_count_matches_the_csv():
    with Path("data/corpus/jobs.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counted = sum(1 for row in rows if row["machine"] == "printer-a")
    task = task_by_id("dev-001")
    assert counted == 5
    assert task["gold_claims"][0]["value"] == counted


def test_harness_files_match_a_fresh_run(tmp_path):
    for mode, name in ("team", "harness_team.json"), ("single", "harness_single.json"):
        out = tmp_path / name
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "orbit_agents.eval.run",
                "--split",
                "test",
                "--mode",
                mode,
                "--backend",
                "scripted",
                "--out",
                str(out),
            ]
        )
        fresh = json.loads(out.read_text(encoding="utf-8"))
        committed = json.loads(Path("results", name).read_text(encoding="utf-8"))
        assert fresh == committed
        assert committed["kind"] == "system_check"
        assert committed["status"] == "measured"


def test_compare_refuses_a_harness_file():
    with pytest.raises(SystemExit):
        compare(
            {
                "single_base": Path("results/harness_single.json"),
                "team_base": Path("results/harness_team.json"),
                "team_lora": Path("results/harness_team.json"),
            }
        )


def test_tune_split_scores_only_the_dev_tune_tasks(tmp_path):
    out = tmp_path / "tune.json"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "orbit_agents.eval.run",
            "--split",
            "tune",
            "--mode",
            "team",
            "--backend",
            "scripted",
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["split"] == "tune"
    assert report["kind"] == "system_check"
    assert [row["id"] for row in report["per_task"]] == [
        "dev-017",
        "dev-018",
        "dev-019",
        "dev-020",
        "dev-021",
        "dev-022",
        "dev-023",
        "dev-024",
    ]
    assert all(row["success"] for row in report["per_task"])


def test_three_way_comparison_matches_the_measured_v2_file():
    report = compare(
        {
            "single_base": Path("results/v2_qwen1.5b/single_base.json"),
            "team_base": Path("results/v2_qwen1.5b/team_base.json"),
            "team_lora": Path("results/v2_qwen1.5b/team_lora.json"),
        }
    )
    committed = json.loads(Path("results/v2_qwen1.5b/comparison.json").read_text(encoding="utf-8"))
    assert report == committed


def test_team_distill_pairs_extend_the_original_comparison(tmp_path):
    paths = {
        "single_base": Path("results/v2_qwen1.5b/single_base.json"),
        "team_base": Path("results/v2_qwen1.5b/team_base.json"),
        "team_lora": Path("results/v2_qwen1.5b/team_lora.json"),
    }
    base = compare(paths)
    distill = tmp_path / "team_distill.json"
    distill.write_text(Path("results/v2_qwen1.5b/team_lora.json").read_text(encoding="utf-8"), encoding="utf-8")
    extra = compare({**paths, "team_distill": distill})
    for key, value in base["pairs"].items():
        assert extra["pairs"][key] == value
    assert set(extra["pairs"]) == set(base["pairs"]) | {
        "team_distill_minus_team_base",
        "team_distill_minus_team_lora",
        "team_distill_minus_single_base",
    }


def test_scorer_refuses_locked_and_already_measured_outputs(tmp_path):
    from orbit_agents.eval.compare import main as compare_main
    from orbit_agents.eval.run import main

    before_placeholder = Path("results/single_base.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="pending placeholder"):
        main(["--split", "tune", "--backend", "scripted", "--out", "results/single_base.json"])
    assert Path("results/single_base.json").read_text(encoding="utf-8") == before_placeholder
    with pytest.raises(SystemExit, match="archived result"):
        main(["--split", "tune", "--backend", "scripted", "--out", "results/v2_qwen1.5b/team_base.json"])
    with pytest.raises(SystemExit, match="pending placeholder"):
        compare_main(
            [
                "--single",
                "results/v2_qwen1.5b/single_base.json",
                "--team",
                "results/v2_qwen1.5b/team_base.json",
                "--lora",
                "results/v2_qwen1.5b/team_lora.json",
                "--out",
                "results/comparison.json",
            ]
        )
    measured = tmp_path / "team_base.json"
    measured.write_text(Path("results/v2_qwen1.5b/team_base.json").read_text(encoding="utf-8"), encoding="utf-8")
    before = measured.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="already measured"):
        main(["--split", "test", "--backend", "scripted", "--out", str(measured)])
    assert measured.read_text(encoding="utf-8") == before


def test_model_files_are_pending():
    for name in ("single_base.json", "team_base.json", "team_lora.json", "comparison.json", "finetune_eval.json"):
        payload = json.loads(Path("results", name).read_text(encoding="utf-8"))
        assert payload["status"] == "pending"
        assert payload["metrics"] is None
