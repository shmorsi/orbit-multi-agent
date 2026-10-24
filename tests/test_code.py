"""Orbit Code sandbox, frozen manifest, scorer, and scripted trace."""

import json
import shutil
from pathlib import Path

import pytest

from orbit_agents.code_sandbox import review_patch, run_pytest, touches_tests
from orbit_agents.code_team import run_code_single
from orbit_agents.codebench import load_tasks, materialize, read_variant, task_by_id, task_prompt, verify_benchmark
from orbit_agents.eval.score_code import main, request_bound
from orbit_agents.policy import LLMClient
from orbit_agents.scripted_code import ScriptedCodePolicy
from orbit_agents.trace import bundled_traces_code


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_sandbox_times_out(tmp_path):
    _write(tmp_path / "tests" / "test_sleep.py", "import time\n\ndef test_sleep():\n    time.sleep(30)\n")
    _write(tmp_path / "noop.py", "x = 1\n")
    result = run_pytest(tmp_path, timeout=1)
    assert result["timed_out"] is True
    assert result["ok"] is False


def test_sandbox_counts_a_pass_inside_the_checkout():
    """A parent pyproject that adds -q must not hide the pass count."""
    task = task_by_id("sum-inclusive")
    dest = Path(".sandbox-check")
    if dest.exists():
        shutil.rmtree(dest)
    try:
        materialize(task, dest)
        (dest / task["entry"]).write_text(read_variant(task, "oracle"), encoding="utf-8")
        result = run_pytest(dest)
        assert result["ok"] is True
        assert result["n_passed"] >= 1
        assert result["returncode"] == 0
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def test_sandbox_blocks_network(tmp_path):
    _write(
        tmp_path / "tests" / "test_net.py",
        "import socket\n\ndef test_net():\n    socket.create_connection(('192.0.2.1', 9), timeout=1)\n",
    )
    _write(tmp_path / "noop.py", "x = 1\n")
    result = run_pytest(tmp_path, timeout=8)
    blob = result["stdout"] + result["stderr"]
    assert result["ok"] is False
    assert result["timed_out"] is False
    assert "network disabled" in blob


def test_reviewer_rejects_a_test_edit_and_a_failure():
    diff = "--- a/tests/test_visible.py\n+++ b/tests/test_visible.py\n@@\n-assert False\n+assert True\n"
    assert touches_tests(diff)
    rejected = review_patch(diff=diff, pytest_result={"ok": True, "timed_out": False, "n_passed": 1})
    assert rejected["verdict"] == "reject"
    assert "test" in rejected["reason"].lower()
    failed = review_patch(diff="--- a/mod.py\n+++ b/mod.py\n", pytest_result={"ok": False, "timed_out": False})
    assert failed["verdict"] == "reject"
    assert "failed" in failed["reason"].lower()
    accepted = review_patch(diff="--- a/mod.py\n+++ b/mod.py\n", pytest_result={"ok": True, "timed_out": False, "n_passed": 1})
    assert accepted["verdict"] == "accept"


def test_manifest_hash_matches_and_a_change_is_refused(tmp_path):
    frozen = verify_benchmark()
    manifest = json.loads(Path("data/scenarios/code/manifest.json").read_text(encoding="utf-8"))
    assert frozen["bundle_sha256"] == manifest["bundle_sha256"]
    assert frozen["split_tag"] == "code-eval-v1"
    assert manifest["n_tasks"] == len(load_tasks())
    root = tmp_path / "repo"
    shutil.copytree("data/scenarios/code", root / "data/scenarios/code")
    tasks = root / "data/scenarios/code/tasks.jsonl"
    tasks.write_text(tasks.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match"):
        verify_benchmark(root)


def test_prompt_hides_hidden_tests_and_the_oracle():
    for task in load_tasks():
        text = task_prompt(task)
        assert "# orbit-hidden" not in text
        assert "def test_" not in text
        assert read_variant(task, "oracle") not in text


def test_hidden_tests_fail_on_the_bug_and_pass_on_the_oracle(tmp_path):
    for task in load_tasks():
        dest = tmp_path / task["id"]
        materialize(task, dest)
        visible = run_pytest(dest)
        shutil.copytree(Path("data/scenarios/code/repos") / task["repo"] / "hidden", dest / "hidden_tests")
        hidden = run_pytest(dest, select="hidden_tests")
        assert visible["ok"] is False, task["id"]
        assert hidden["ok"] is False, task["id"]
        (dest / task["entry"]).write_text(read_variant(task, "oracle"), encoding="utf-8")
        assert run_pytest(dest)["ok"] is True, task["id"]
        assert run_pytest(dest, select="hidden_tests")["ok"] is True, task["id"]


class _Oracle(LLMClient):
    provider = "fake"
    stops = ()

    def __init__(self, task):
        self.task = task

    def complete_text(self, messages, schema, *, role):
        del messages, schema
        if role == "coder":
            return json.dumps({"edits": [{"path": self.task["entry"], "content": read_variant(self.task, "oracle")}]})
        if role == "critic":
            return json.dumps({"verdict": "accept", "reason": "The tests passed."})
        return json.dumps({"steps": [{"goal": "Fix the source."}]})


def test_single_oracle_passes_and_tamper_is_not_a_pass(tmp_path):
    from orbit_agents.eval.score_code import score_one

    task = task_by_id("sum-inclusive")
    passed = score_one(task, _Oracle(task), tmp_path / "ok", config="single", root=Path("."))
    assert passed["pass"] is True
    assert passed["reviewer_rejected"] is False
    assert passed["hidden_passed_rate"] == 1
    refused = score_one(task, ScriptedCodePolicy(task, mode="tamper"), tmp_path / "bad", config="single", root=Path("."))
    assert refused["pass"] is False
    assert refused["reviewer_rejected"] is True
    assert run_code_single(task, ScriptedCodePolicy(task, mode="repair"), tmp_path / "single").status == "refuse"


def test_score_dry_run_prints_the_bound_and_writes_nothing():
    before = sorted(path.relative_to("results") for path in Path("results").rglob("*") if path.is_file())
    plan = main(["--config", "team", "--model", "claude-haiku-5-5", "--dry-run"])
    after = sorted(path.relative_to("results") for path in Path("results").rglob("*") if path.is_file())
    assert after == before
    assert plan["max_cost_usd"] == 5.0
    assert plan["requests"] == request_bound(len(load_tasks()), "team")
    assert plan["requests"]["hard_cap"] > request_bound(len(load_tasks()), "single")["hard_cap"]
    assert plan["price_usd_per_million"]["input"] > 0
    assert plan["price_source"] == "table"
    assert plan["already_measured"] is False
    assert "read timeout" in plan["requests"]["note"]


def test_scorer_refuses_to_overwrite_a_measured_file(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree("data/scenarios/code", root / "data/scenarios/code")
    out = root / "results" / "code" / "claude-haiku-5-5" / "team.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({"status": "measured", "kind": "model_eval"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="already measured"):
        main(["--config", "team", "--model", "claude-haiku-5-5", "--root", str(root), "--dry-run"])


def test_scripted_traces_show_the_repair_and_the_test_edit_rejection(tmp_path):
    written = bundled_traces_code(tmp_path)
    by_name = {path.name: path for path in written}
    repair_rows = [json.loads(line) for line in by_name["repair.jsonl"].read_text(encoding="utf-8").splitlines()]
    header = repair_rows[0]
    assert header["fabricated"] is False
    assert header["harness"] == "scripted_harness"
    assert header["scenario"] == "code"
    assert "not a model score" in header["scenario_note"].lower()
    assert header["status"] == "answer"
    events = repair_rows[1:]
    tests = [event["ok"] for event in events if event["kind"] == "test"]
    reviews = [event["verdict"] for event in events if event["kind"] == "review"]
    assert tests == [False, True]
    assert reviews[0] == "reject"
    assert reviews[-1] == "accept"
    assert any(event["kind"] == "revise" for event in events)
    assert any(event["agent"] == "planner" for event in events)
    reject_rows = [json.loads(line) for line in by_name["reject.jsonl"].read_text(encoding="utf-8").splitlines()]
    assert reject_rows[0]["status"] == "refuse"
    edit = next(event for event in reject_rows[1:] if event["kind"] == "edit")
    assert "tests/test_visible.py" in edit["diff"]
    review = next(event for event in reject_rows[1:] if event["kind"] == "review")
    assert review["verdict"] == "reject"
    assert "test" in review["reason"].lower()
    for path in written:
        committed = Path("mission/traces/code") / path.name
        assert path.read_bytes() == committed.read_bytes()
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index == json.loads(Path("mission/traces/code/index.json").read_text(encoding="utf-8"))


def test_mission_control_lists_the_code_scenario():
    html = Path("mission/index.html").read_text(encoding="utf-8")
    script = Path("mission/replay.js").read_text(encoding="utf-8")
    assert 'id="scenario-code"' in html
    assert 'id="scenario-boardroom"' in html
    assert 'id="scenario-incident"' in html
    assert 'id="diffview"' in html
    assert 'id="testlamp"' in html
    assert 'id="lanes"' in html
    assert "pending, not scored" in html
    assert "traces/code/index.json" in script
    assert "Orbit Code" in script
    assert "Orbit Boardroom" in script
