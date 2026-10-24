"""Orbit Code sandbox, frozen manifest, scorer, and scripted trace."""

import difflib
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from orbit_agents.code_sandbox import review_patch, run_pytest, touches_tests
from orbit_agents.code_team import run_code_single, run_code_team
from orbit_agents.codebench import (
    PROTOCOL_V2,
    load_tasks,
    materialize,
    read_variant,
    task_by_id,
    task_prompt,
    verify_benchmark,
)
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
    assert frozen["bundle_sha256"] == "18665421458c6661299b60c1aa982394c9d3b773660a16bb3d7066f090a7028f"
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


def test_score_dry_run_prints_the_bound_and_writes_nothing(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree("data/scenarios/code_v2", root / "data/scenarios/code_v2")
    before = sorted(path.relative_to("results") for path in Path("results").rglob("*") if path.is_file())
    plan = main(["--config", "team", "--model", "claude-haiku-5-5", "--root", str(root), "--dry-run"])
    after = sorted(path.relative_to("results") for path in Path("results").rglob("*") if path.is_file())
    assert after == before
    assert not (root / "results").exists()
    n_tasks = len(load_tasks(protocol=PROTOCOL_V2))
    assert plan["max_cost_usd"] == 5.0
    assert plan["frozen_test"]["protocol"] == PROTOCOL_V2
    assert plan["out"] == "results/code_v2/claude-haiku-5-5/team.json"
    assert plan["requests"] == request_bound(n_tasks, "team")
    assert plan["requests"]["hard_cap"] > request_bound(n_tasks, "single")["hard_cap"]
    assert plan["price_usd_per_million"]["input"] > 0
    assert plan["price_source"] == "table"
    assert plan["already_measured"] is False
    assert "read timeout" in plan["requests"]["note"]
    single = main(["--config", "single", "--model", "claude-haiku-5-5", "--root", str(root), "--dry-run"])
    assert single["requests"] == request_bound(n_tasks, "single")
    assert single["max_cost_usd"] == 5.0
    assert single["out"] == "results/code_v2/claude-haiku-5-5/single.json"
    assert single["already_measured"] is False


def test_scorer_refuses_to_overwrite_a_measured_file(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree("data/scenarios/code", root / "data/scenarios/code")
    out = root / "results" / "code" / "claude-haiku-5-5" / "team.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({"status": "measured", "kind": "model_eval"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="already measured"):
        main(
            [
                "--config",
                "team",
                "--model",
                "claude-haiku-5-5",
                "--split",
                "code-eval-v1",
                "--root",
                str(root),
                "--dry-run",
            ]
        )


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
    assert "code-eval-v2" in html


class _Script(LLMClient):
    """One JSON object per role. The replies are fixed, so the loop can be tested offline."""

    provider = "fake"
    stops = ()

    def __init__(self, replies):
        self.replies = replies
        self.roles = []
        self.users = []

    def complete_text(self, messages, schema, *, role):
        del schema
        self.roles.append(role)
        self.users.append(messages[-1]["content"])
        reply = self.replies[role]
        if callable(reply):
            return reply(self, messages)
        return reply


def test_planner_extra_fields_do_not_skip_the_coder(tmp_path):
    task = task_by_id("sum-inclusive")
    oracle = read_variant(task, "oracle")
    policy = _Script(
        {
            "planner": json.dumps({"steps": [{"goal": "Fix the source. " + ("x" * 240), "file": "x.py"}], "note": "extra"}),
            "coder": json.dumps({"edits": [{"path": task["entry"], "content": oracle, "why": "fix"}]}),
            "critic": json.dumps({"verdict": "accept", "reason": "Ship it."}),
        }
    )
    result = run_code_team(task, policy, tmp_path)
    assert "coder" in policy.roles
    assert result.status == "answer"
    assert policy.roles.count("planner") == 1


def test_a_dead_planner_still_reaches_the_coder(tmp_path):
    task = task_by_id("sum-inclusive")
    oracle = read_variant(task, "oracle")
    policy = _Script(
        {
            "planner": json.dumps({"steps": "not-a-list"}),
            "coder": json.dumps({"edits": [{"path": task["entry"], "content": oracle}]}),
            "critic": json.dumps({"verdict": "accept", "reason": "Ship it."}),
        }
    )
    result = run_code_team(task, policy, tmp_path)
    assert policy.roles.count("coder") == 1
    assert result.status == "answer"


def test_schema_repair_names_the_failed_check(tmp_path):
    task = task_by_id("sum-inclusive")
    oracle = read_variant(task, "oracle")
    calls = {"planner": 0}

    def planner(_policy, _messages):
        calls["planner"] += 1
        if calls["planner"] == 1:
            return json.dumps({"steps": "not-a-list"})
        return json.dumps({"steps": [{"goal": "Fix the source."}]})

    policy = _Script(
        {
            "planner": planner,
            "coder": json.dumps({"edits": [{"path": task["entry"], "content": oracle}]}),
            "critic": json.dumps({"verdict": "accept", "reason": "Ship it."}),
        }
    )
    result = run_code_team(task, policy, tmp_path)
    planner_users = [text for role, text in zip(policy.roles, policy.users, strict=True) if role == "planner"]
    assert "Validator says" in planner_users[1]
    assert result.status == "answer"


def test_critic_accept_on_a_red_build_still_revises(tmp_path):
    task = task_by_id("sum-inclusive")
    wrong = read_variant(task, "wrong")
    oracle = read_variant(task, "oracle")
    calls = {"coder": 0}

    def coder(policy, _messages):
        calls["coder"] += 1
        content = oracle if calls["coder"] > 1 and "Tests failed." in policy.users[-1] else wrong
        return json.dumps({"edits": [{"path": task["entry"], "content": content}]})

    policy = _Script(
        {
            "planner": json.dumps({"steps": [{"goal": "Fix the source."}]}),
            "coder": coder,
            "critic": json.dumps({"verdict": "accept", "reason": "Ship it."}),
        }
    )
    result = run_code_team(task, policy, tmp_path)
    assert calls["coder"] == 2
    assert result.status == "answer"
    assert any(event["kind"] == "revise" for event in result.events)


def test_a_test_edit_is_not_revised_into_a_pass(tmp_path):
    task = task_by_id("sum-inclusive")
    policy = ScriptedCodePolicy(task, mode="tamper")
    result = run_code_team(task, policy, tmp_path)
    assert result.status == "refuse"
    assert policy.coder_calls == 1
    assert any(event["kind"] == "review" and event["verdict"] == "reject" for event in result.events)


def test_unified_diff_and_a_nested_path_patch_the_tree(tmp_path):
    task = task_by_id("sum-inclusive")
    original = Path("data/scenarios/code/repos/sum_inclusive/sum_inclusive.py").read_text(encoding="utf-8")
    oracle = read_variant(task, "oracle")
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            oracle.splitlines(keepends=True),
            fromfile="a/nested/sum_inclusive.py",
            tofile="b/nested/sum_inclusive.py",
        )
    )
    policy = _Script(
        {
            "coder": json.dumps({"edits": [{"path": "nested/sum_inclusive.py", "content": diff}]}),
            "critic": json.dumps({"verdict": "accept", "reason": "Ship it."}),
        }
    )
    result = run_code_single(task, policy, tmp_path)
    assert result.status == "answer"
    assert (tmp_path / "repo" / "sum_inclusive.py").read_text(encoding="utf-8") == oracle


def test_v1_score_files_are_not_overwritten_or_rewritten():
    expected = {
        "single.json": "372bbe9e38b3b58357463e765e11dcf742d5e91ecc513478c230e31690ab4c90",
        "team.json": "f67038e781dff027bb3de704e0df193c65d52faa04ec7ef6dffb0dc1a916efb8",
    }
    for name, digest in expected.items():
        body = (Path("results/code/claude-haiku-5-5") / name).read_bytes()
        assert hashlib.sha256(body).hexdigest() == digest
    before = {name: (Path("results/code/claude-haiku-5-5") / name).read_bytes() for name in expected}
    for config in ("single", "team"):
        with pytest.raises(SystemExit, match="already measured"):
            main(["--config", config, "--model", "claude-haiku-5-5", "--split", "code-eval-v1", "--dry-run"])
    after = {name: (Path("results/code/claude-haiku-5-5") / name).read_bytes() for name in expected}
    assert after == before
