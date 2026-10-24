"""Team v2: six roles, explicit permissions, an auditor, and a provider-agnostic policy layer."""

import json
import shutil
from pathlib import Path

import pytest

from orbit_agents.engine import MAX_CRITIC_RETRIES, MAX_TURNS, UNGROUNDED
from orbit_agents.eval.score import score_task
from orbit_agents.policy import (
    PROVIDERS,
    AnthropicPolicy,
    LLMClient,
    OpenAICompatiblePolicy,
    PolicyBackend,
    make_policy,
)
from orbit_agents.prompts import prompt_blob, prompt_blob_v2
from orbit_agents.schema import ROLE_TOOLS, check_role_v2, role_schema_v2
from orbit_agents.scripted_v2 import AuditorCatch, ReviseOnceV2, ScriptedPolicyV2
from orbit_agents.tasks import all_tasks, read_jsonl, task_by_id
from orbit_agents.team_v2 import FLIGHT_RULES_V2, TraceView, audit, run_team_v2


def _tool(tool, **arguments):
    return {"action": "tool", "tool": tool, "arguments": arguments}


# ---------------------------------------------------------------- permissions


def test_role_permissions_are_minimal_and_explicit():
    assert ROLE_TOOLS == {
        "commander": (),
        "scout": ("retrieve", "read_file", "web_search"),
        "engineer": ("python", "write_file"),
        "analyst": ("calculator",),
        "critic": (),
        "auditor": (),
    }
    assert FLIGHT_RULES_V2["permissions"]["scout"] == ["retrieve", "read_file", "web_search"]


def test_scout_cannot_run_code_and_engineer_cannot_answer():
    assert check_role_v2("scout", _tool("retrieve", query="laser")) is None
    assert "scout may use only" in check_role_v2("scout", _tool("python", code="print(1)"))
    assert check_role_v2("engineer", _tool("python", code="print(1)")) is None
    final = {"action": "final", "status": "refuse", "answer": "No record.", "claims": []}
    assert "engineer cannot give the final answer" in check_role_v2("engineer", final, may_answer=True)
    assert "scout cannot give the final answer" in check_role_v2("scout", final, may_answer=True)
    assert "drafting turn" in check_role_v2("analyst", final, may_answer=False)
    assert check_role_v2("analyst", final, may_answer=True) is None
    assert "analyst may use only calculator" in check_role_v2("analyst", _tool("python", code="print(1)"), may_answer=True)


def test_role_schemas_list_only_that_roles_tools():
    assert role_schema_v2("scout")["properties"]["tool"]["enum"] == ["retrieve", "read_file", "web_search"]
    assert role_schema_v2("engineer")["properties"]["action"]["enum"] == ["tool"]
    assert role_schema_v2("analyst", may_answer=True)["properties"]["action"]["enum"] == ["tool", "final"]


def test_commander_cannot_assign_a_tool_outside_the_role():
    bad = {"steps": [{"goal": "Count jobs", "role": "scout", "tool": "python"}]}
    assert "scout may use only" in check_role_v2("commander", bad)
    unknown = {"steps": [{"goal": "Judge", "role": "auditor", "tool": "python"}]}
    assert check_role_v2("commander", unknown) is not None
    good = {"steps": [{"goal": "Count jobs", "role": "engineer", "tool": "python"}]}
    assert check_role_v2("commander", good) is None


class _ScoutTriesPython(ScriptedPolicyV2):
    def complete_text(self, messages, schema, *, role):
        if role == "scout":
            return json.dumps(_tool("python", code="print(1)"))
        return super().complete_text(messages, schema, role=role)


def test_an_out_of_role_tool_call_never_runs(tmp_path):
    task = task_by_id("dev-006")
    result = run_team_v2(task, _ScoutTriesPython(task), tmp_path)
    assert result.tool_calls == []
    assert result.schema_invalid >= 1
    assert any(event["kind"] == "invalid" and "scout may use only" in event["error"] for event in result.events)
    assert result.status == "refuse"


def test_auditor_reads_a_frozen_copy_of_the_trace():
    results = [{"by": "engineer", "tool": "python", "arguments": {}, "result": {"ok": True, "stdout": "5\n"}}]
    view = TraceView.from_results("How many?", results)
    results[0]["result"]["stdout"] = "999\n"
    assert view.tool_results()[0]["result"]["stdout"] == "5\n"
    with pytest.raises(TypeError):
        view.results[0]["result"] = {}
    copy = view.tool_results()
    copy[0]["result"]["stdout"] = "7\n"
    assert view.tool_results()[0]["result"]["stdout"] == "5\n"


# ---------------------------------------------------------------- auditor


def test_auditor_refuses_an_unsupported_number():
    results = [{"by": "engineer", "tool": "python", "arguments": {}, "result": {"ok": True, "stdout": "5\n"}}]
    view = TraceView.from_results("How many jobs used printer-a?", results)
    claims = [{"name": "job_count", "value": 5, "citation": "call_1"}]
    ok = audit(view, "answer", "job_count is 5.", claims)
    assert ok["verdict"] == "pass"
    extra = audit(view, "answer", "job_count is 5, about 40 percent.", claims)
    assert extra["verdict"] == "refuse"
    assert {"number": "40", "ok": False, "reason": "appears in no tool result and not in the question"} in extra["findings"]
    wrong = audit(view, "answer", "job_count is 7.", [{"name": "job_count", "value": 7, "citation": "call_1"}])
    assert wrong["verdict"] == "refuse"
    assert wrong["findings"][0]["ok"] is False


def test_auditor_overrides_a_critic_that_accepted(tmp_path):
    task = task_by_id("dev-001")
    result = run_team_v2(task, AuditorCatch(task, 40), tmp_path)
    accepts = [event for event in result.events if event["agent"] == "critic" and event["kind"] == "parsed"]
    assert accepts and accepts[-1]["payload"]["verdict"] == "accept"
    assert result.status == "refuse"
    assert result.answer == UNGROUNDED
    assert result.claims == []
    assert result.gate_overrode is True
    assert score_task(task, result)["success"] is False


# ---------------------------------------------------------------- critic cap


class _AlwaysRevise(ScriptedPolicyV2):
    def critic(self):
        return {"verdict": "revise", "reason": "Not good enough."}


def test_critic_revision_cap_and_turn_limit(tmp_path):
    task = task_by_id("dev-001")
    result = run_team_v2(task, _AlwaysRevise(task), tmp_path)
    revises = [event for event in result.events if event["kind"] == "revise"]
    assert 1 <= len(revises) <= MAX_CRITIC_RETRIES + 1
    assert result.n_steps <= MAX_TURNS
    # The last draft still goes through the auditor, which keeps the grounded answer.
    assert result.status == "answer"
    assert result.claims == task["gold_claims"]


def test_revise_once_keeps_the_gold_answer(tmp_path):
    task = task_by_id("dev-008")
    result = run_team_v2(task, ReviseOnceV2(task), tmp_path)
    assert sum(1 for event in result.events if event["kind"] == "revise") == 1
    assert result.status == "answer"
    assert result.claims == task["gold_claims"]


# ---------------------------------------------------------------- policy layer


class _FakeClient(LLMClient):
    provider = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete_text(self, messages, schema, *, role):
        self.calls.append({"role": role, "schema": schema, "messages": messages})
        return self.replies.pop(0)


def test_provider_interface_with_a_fake_client(tmp_path):
    task = task_by_id("dev-001")
    code = task["oracle_calls"][0]["arguments"]["code"]
    fake = _FakeClient(
        [
            json.dumps({"steps": [{"goal": "Count", "role": "engineer", "tool": "python"}]}),
            "Here you go: " + json.dumps(_tool("python", code=code)),
            json.dumps({"action": "final", "status": "answer", "answer": "job_count is 5.", "claims": task["gold_claims"]}),
            json.dumps({"verdict": "accept", "reason": "Grounded."}),
        ]
    )
    result = run_team_v2(task, fake, tmp_path)
    assert [call["role"] for call in fake.calls] == ["commander", "engineer", "analyst", "critic"]
    assert fake.calls[1]["schema"]["properties"]["tool"]["enum"] == ["python", "write_file"]
    assert result.status == "answer" and result.claims == task["gold_claims"]
    parsed = _FakeClient(['note {"verdict": "accept", "reason": "ok"}']).complete([], {}, role="critic")
    assert parsed == {"verdict": "accept", "reason": "ok"}


def test_v1_team_runs_through_the_same_interface(tmp_path):
    from orbit_agents.engine import run_team

    task = task_by_id("dev-001")
    policy = make_policy("scripted", task=task, team="v1")
    result = run_team(task, PolicyBackend(policy), tmp_path)
    assert result.status == "answer" and result.claims == task["gold_claims"]


def test_providers_include_anthropic_and_the_extension_point_is_a_stub():
    assert set(PROVIDERS) >= {"scripted", "transformers", "ollama", "llamacpp", "anthropic"}
    with pytest.raises(NotImplementedError):
        OpenAICompatiblePolicy()
    assert issubclass(OpenAICompatiblePolicy, LLMClient)


def test_anthropic_policy_needs_the_env_key(monkeypatch):
    monkeypatch.delenv("ORBIT_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ORBIT_ANTHROPIC_API_KEY"):
        AnthropicPolicy("claude-haiku-5-5", max_requests=5, max_cost_usd=1)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    AnthropicPolicy("claude-haiku-5-5", max_requests=5, max_cost_usd=1, transport=lambda *args: None)
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "ORBIT_ANTHROPIC_API_KEY" in makefile
    assert "make score-dry-run does not need a key" in makefile


def _anthropic_reply(text):
    return 200, {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": 400, "output_tokens": 40}}


def test_anthropic_policy_sends_role_prompts_only_and_honors_the_cap(monkeypatch):
    from distill_teacher import BudgetStop

    monkeypatch.setenv("ORBIT_ANTHROPIC_API_KEY", "test-key")
    sent = []

    def transport(url, headers, payload, timeout):
        del timeout
        sent.append({"url": url, "headers": headers, "payload": payload})
        return _anthropic_reply('{"verdict": "accept", "reason": "ok"}')

    policy = AnthropicPolicy(
        "claude-haiku-5-5", max_requests=2, max_cost_usd=1, transport=transport, sleep=lambda _s: None, log=lambda _e: None
    )
    messages = [{"role": "system", "content": "You are the critic."}, {"role": "user", "content": "Task: x"}]
    assert policy.complete(messages, {}, role="critic")["verdict"] == "accept"
    policy.complete_text(messages, {}, role="critic")
    with pytest.raises(BudgetStop):
        policy.complete_text(messages, {}, role="critic")
    assert len(sent) == 2
    body = json.dumps(sent[0]["payload"])
    assert "Checker contract" not in body
    assert sent[0]["payload"]["system"] == "You are the critic."
    assert "temperature" not in sent[0]["payload"]
    assert policy.usage()["requests"] == 2
    assert policy.usage()["estimated_cost_usd"] > 0


def test_policy_prompts_carry_no_checker_rules_or_frozen_test_content():
    from leakage import blocking_test_task

    test_rows = read_jsonl(Path("eval/test.jsonl"))
    blob = prompt_blob_v2()
    assert "Checker contract" not in blob
    assert "checker requires" not in blob.lower()
    for row in test_rows:
        assert row["question"] not in blob
        assert row["answer"] not in blob
        for claim in row["gold_claims"]:
            assert f'"name": "{claim["name"]}"' not in blob
            if "text" in claim:
                assert claim["text"] not in blob
    assert blocking_test_task(blob, test_rows) is None


def test_v1_prompts_and_the_frozen_manifest_are_untouched():
    import hashlib

    manifest = json.loads(Path("eval/manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(Path("eval/test.jsonl").read_bytes()).hexdigest() == manifest["test_sha256"]
    assert "commander" not in prompt_blob()


# ---------------------------------------------------------------- end to end


def test_scripted_v2_team_solves_every_task(tmp_path):
    for task in all_tasks():
        result = run_team_v2(task, ScriptedPolicyV2(task), tmp_path / task["id"])
        row = score_task(task, result)
        assert row["success"], task["id"]
        assert result.schema_invalid == 0
        roles = {event["agent"] for event in result.events}
        assert "auditor" in roles and "commander" in roles
        for call in result.tool_calls:
            assert call["tool"] in ROLE_TOOLS[call["by"]]


def test_committed_v2_traces_match_the_engine(tmp_path):
    from orbit_agents.trace import bundled_traces_v2

    written = bundled_traces_v2(tmp_path)
    assert [path.name for path in written] == ["accept.jsonl", "revise.jsonl", "refuse.jsonl"]
    statuses = {}
    for path in written:
        committed = Path("mission/traces/v2") / path.name
        assert path.read_bytes() == committed.read_bytes()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        header = rows[0]
        assert header["kind"] == "mission"
        assert header["fabricated"] is False
        assert header["source"] == "orbit_agents.engine"
        assert header["harness"] == "scripted_harness"
        assert header["team"] == "v2"
        assert header["flight_rules"]["max_turns"] == MAX_TURNS
        for index, row in enumerate(rows[1:]):
            assert row["index"] == index
            assert {"agent", "kind", "t_sec"} <= set(row)
        statuses[path.name] = (header["status"], header["n_retries"], header["gate_overrode"])
    assert statuses["accept.jsonl"] == ("answer", 0, False)
    assert statuses["revise.jsonl"] == ("answer", 1, False)
    assert statuses["refuse.jsonl"] == ("refuse", 0, True)
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index == json.loads(Path("mission/traces/v2/index.json").read_text(encoding="utf-8"))


def test_eval_run_team_v2_is_a_system_check(tmp_path):
    from orbit_agents.eval.run import main

    out = tmp_path / "v2.json"
    main(["--team", "v2", "--split", "test", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["kind"] == "system_check"
    assert report["team"] == "v2"
    assert report["n_success"] == report["n_tasks"] == 12


# ---------------------------------------------------------------- score once


def _scratch_root(tmp_path):
    root = tmp_path / "repo"
    (root / "eval").mkdir(parents=True)
    for name in ("manifest.json", "test.jsonl"):
        shutil.copy(Path("eval") / name, root / "eval" / name)
    for name in ("v1_qwen0.5b", "v2_qwen1.5b"):
        shutil.copytree(Path("results") / name, root / "results" / name)
    shutil.copy("README.md", root / "README.md")
    return root


class _OraclePolicy(LLMClient):
    """A fake API policy for the score command. It is not a model and never leaves the test."""

    provider = "fake"
    stops = ()

    def __init__(self):
        self.inner = None

    def complete_text(self, messages, schema, *, role):
        task_line = next(m["content"] for m in messages if m["role"] == "user").splitlines()[0]
        question = task_line.removeprefix("Task: ")
        if self.inner is None or self.inner.task["question"] != question:
            task = next(row for row in read_jsonl(Path("eval/test.jsonl")) if row["question"] == question)
            self.inner = ScriptedPolicyV2(task)
        return self.inner.complete_text(messages, schema, role=role)

    def usage(self):
        return {"requests": 0}


def test_score_team_dry_run_and_request_bound(tmp_path):
    from orbit_agents.eval.score_team import main, request_bound

    root = _scratch_root(tmp_path)
    plan = main(["--team", "v2", "--model", "claude-haiku-5-5", "--root", str(root), "--dry-run"])
    assert plan["frozen_test"]["test_sha256"] == json.loads(Path("eval/manifest.json").read_text())["test_sha256"]
    assert plan["requests"] == request_bound(12)
    assert plan["requests"]["upper_bound_without_retries"] == 1 + 12 * MAX_TURNS
    assert plan["max_requests"] == plan["requests"]["hard_cap"]
    assert not (root / "results" / "v3_api").exists()


def test_score_team_refuses_a_changed_frozen_test(tmp_path):
    from orbit_agents.eval.score_team import main

    root = _scratch_root(tmp_path)
    with (root / "eval" / "test.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(SystemExit, match="does not match"):
        main(["--team", "v2", "--root", str(root), "--dry-run"])


def test_results_row_appears_only_when_measured_and_scoring_runs_once(tmp_path):
    from orbit_agents.eval.readme import render
    from orbit_agents.eval.score_team import main

    root = _scratch_root(tmp_path)
    baseline = render(root)
    assert "v2 team (6 roles)" not in baseline
    folder = root / "results" / "v3_api" / "fake-model"
    folder.mkdir(parents=True)
    (folder / "team_v2.json").write_text(json.dumps({"status": "stopped", "kind": "model_eval", "model": "fake-model"}))
    assert render(root) == baseline
    report = main(["--team", "v2", "--provider", "ollama", "--model", "fake-model", "--root", str(root)], policy=_OraclePolicy())
    assert report["status"] == "measured" and report["n_tasks"] == 12
    assert report["previous_stops"] == 1
    block = render(root)
    assert "| v2 team (6 roles) | 12 of 12 |" in block
    assert "| v1 team (3 roles) | not scored | | | |" in block
    assert "no difference is claimed" in block
    with pytest.raises(SystemExit, match="already measured"):
        main(["--team", "v2", "--provider", "ollama", "--model", "fake-model", "--root", str(root)], policy=_OraclePolicy())


def test_readme_points_at_the_measured_v2_score():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "## v2: six roles" in readme
    section = readme.split("## v2: six roles", 1)[1].split("\n## ", 1)[0]
    assert "not been scored" not in section
    assert "make score-v2" in section
    assert "results/v3_api/" in section
