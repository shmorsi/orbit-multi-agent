"""Team v3: the six-role crew plus a coordinator, researcher, red team, and verifier.

The scripted policy is a harness. It is not a model score.
"""

import json
from pathlib import Path

from orbit_agents.engine import MAX_TURNS
from orbit_agents.eval.score import score_task
from orbit_agents.prompts import prompt_blob, prompt_blob_v2, prompt_blob_v3
from orbit_agents.schema import ROLE_TOOLS_V3, ROLES_V3, check_role_v3, role_schema_v3
from orbit_agents.scripted_v2 import ScriptedPolicyV2
from orbit_agents.scripted_v3 import PLANTED_VALUE, ScriptedPolicyV3
from orbit_agents.tasks import all_tasks, task_by_id
from orbit_agents.team_v2 import run_team_v2
from orbit_agents.team_v3 import FLIGHT_RULES_V3, ROUND_TURNS, run_team_v3


def _tool(tool, **arguments):
    return {"action": "tool", "tool": tool, "arguments": arguments}


def test_round_table_roles_and_permissions():
    assert ROLES_V3 == [
        "coordinator",
        "commander",
        "researcher",
        "scout",
        "engineer",
        "analyst",
        "critic",
        "redteam",
        "verifier",
        "auditor",
    ]
    assert ROLE_TOOLS_V3["coordinator"] == ()
    assert ROLE_TOOLS_V3["researcher"] == ("retrieve", "read_file", "web_search")
    assert ROLE_TOOLS_V3["redteam"] == ()
    assert ROLE_TOOLS_V3["verifier"] == ()
    assert FLIGHT_RULES_V3["round_turns"] == ROUND_TURNS
    assert FLIGHT_RULES_V3["builds_on"] == "v2"
    assert "commander" not in prompt_blob()
    assert "coordinator" not in prompt_blob_v2()
    assert "coordinator" in prompt_blob_v3()


def test_each_new_role_rejects_a_tool_it_does_not_own():
    final = {"action": "final", "status": "refuse", "answer": "No record.", "claims": []}
    assert "researcher may use only" in check_role_v3("researcher", _tool("python", code="print(1)"))
    assert "researcher cannot give the final answer" in check_role_v3("researcher", final)
    assert check_role_v3("researcher", _tool("retrieve", query="laser")) is None
    assert check_role_v3("researcher", _tool("read_file", path="logs/a.txt")) is None
    assert check_role_v3("researcher", _tool("web_search", query="hours")) is None
    assert role_schema_v3("researcher")["properties"]["tool"]["enum"] == ["retrieve", "read_file", "web_search"]
    assert role_schema_v3("researcher")["properties"]["action"]["enum"] == ["tool"]
    assert check_role_v3("coordinator", {"brief": "Use only the bundled corpus."}) is None
    assert check_role_v3("coordinator", {"brief": "Use only the bundled corpus.", "tool": "python"}) is not None
    planted = {
        "claim": {"name": "planted", "value": PLANTED_VALUE, "unit": "count", "citation": "call_1"},
        "why": "This count was not in the tool results.",
    }
    assert check_role_v3("redteam", planted) is None
    assert check_role_v3("redteam", {"why": "missing the claim"}) is not None
    assert check_role_v3("verifier", {"verdict": "reject", "reason": "The cited result does not contain it."}) is None
    assert check_role_v3("verifier", {"verdict": "accept", "reason": "no"}) is not None
    good = {"steps": [{"goal": "Find the log", "role": "researcher", "tool": "retrieve"}]}
    assert check_role_v3("commander", good) is None
    bad = {"steps": [{"goal": "Count jobs", "role": "researcher", "tool": "python"}]}
    assert "researcher may use only" in check_role_v3("commander", bad)


class _ResearcherTriesPython(ScriptedPolicyV3):
    def complete_text(self, messages, schema, *, role):
        if role == "researcher":
            return json.dumps(_tool("python", code="print(1)"))
        return super().complete_text(messages, schema, role=role)


def test_researcher_python_never_runs(tmp_path):
    task = task_by_id("dev-006")
    result = run_team_v3(task, _ResearcherTriesPython(task), tmp_path)
    assert result.tool_calls == []
    assert result.schema_invalid >= 1
    assert any(event["kind"] == "invalid" and "researcher may use only" in event["error"] for event in result.events)


def test_scripted_v3_solves_every_task_without_spending_extra_specialist_turns(tmp_path):
    for task in all_tasks():
        v2 = run_team_v2(task, ScriptedPolicyV2(task), tmp_path / "v2" / task["id"])
        v3 = run_team_v3(task, ScriptedPolicyV3(task), tmp_path / "v3" / task["id"])
        assert score_task(task, v3)["success"], task["id"]
        assert v3.schema_invalid == 0
        assert v3.n_steps == v2.n_steps <= MAX_TURNS
        assert v3.status == v2.status and v3.claims == v2.claims
        assert v3.schema_valid == v2.schema_valid + ROUND_TURNS
        for call in v3.tool_calls:
            assert call["tool"] in ROLE_TOOLS_V3[call["by"]]
        if any(call["tool"] in ROLE_TOOLS_V3["researcher"] for call in task["oracle_calls"]):
            assert any(call["by"] == "researcher" for call in v3.tool_calls), task["id"]
        assert all(call["by"] != "researcher" or call["tool"] != "python" for call in v3.tool_calls)


def test_red_team_claim_is_refused_in_code_and_left_out_of_the_answer(tmp_path):
    task = task_by_id("dev-008")
    result = run_team_v3(task, ScriptedPolicyV3(task), tmp_path)
    gate = next(event for event in result.events if event["kind"] == "gate")
    assert gate["agent"] == "verifier"
    assert gate["code_verdict"] == "refuse"
    assert gate["claim"]["value"] == PLANTED_VALUE
    verdict = next(event for event in result.events if event["agent"] == "verifier" and event["kind"] == "parsed")
    assert verdict["payload"]["verdict"] == "reject"
    assert result.status == "answer"
    assert all(claim.get("value") != PLANTED_VALUE for claim in result.claims)
    assert result.gate_overrode is False


def test_committed_v3_traces_match_the_engine(tmp_path):
    from orbit_agents.trace import bundled_traces_v3

    written = bundled_traces_v3(tmp_path)
    assert [path.name for path in written] == ["accept.jsonl", "revise.jsonl", "refuse.jsonl"]
    statuses = {}
    for path in written:
        committed = Path("mission/traces/v3") / path.name
        assert path.read_bytes() == committed.read_bytes()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        header = rows[0]
        assert header["team"] == "v3"
        assert header["fabricated"] is False
        assert header["harness"] == "scripted_harness"
        assert any(event["kind"] == "gate" and event["code_verdict"] == "refuse" for event in rows[1:])
        statuses[path.name] = (header["status"], header["n_retries"], header["gate_overrode"])
    assert statuses["accept.jsonl"] == ("answer", 0, False)
    assert statuses["revise.jsonl"] == ("answer", 1, False)
    assert statuses["refuse.jsonl"] == ("refuse", 0, True)
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index == json.loads(Path("mission/traces/v3/index.json").read_text(encoding="utf-8"))
    assert any(row["task_id"] == "dev-006" for row in index)


def test_eval_run_team_v3_is_a_system_check(tmp_path):
    from orbit_agents.eval.run import main

    out = tmp_path / "v3.json"
    main(["--team", "v3", "--split", "dev", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["kind"] == "system_check"
    assert report["team"] == "v3"
    assert report["n_success"] == report["n_tasks"]


def test_score_v3_dry_run_writes_nothing_and_counts_round_turns(tmp_path):
    import shutil

    from orbit_agents.eval.score_team import main, request_bound

    root = tmp_path / "repo"
    (root / "eval").mkdir(parents=True)
    for name in ("manifest.json", "test.jsonl"):
        shutil.copy(Path("eval") / name, root / "eval" / name)
    plan = main(["--team", "v3", "--model", "claude-haiku-5-5", "--root", str(root), "--dry-run"])
    assert plan["requests"] == request_bound(12, ROUND_TURNS)
    assert plan["requests"]["max_turn_requests"] == 12 * (MAX_TURNS + ROUND_TURNS)
    assert plan["label"] == "Round-table team"
    assert not (root / "results").exists()


def test_readme_names_the_round_table_as_not_scored():
    readme = Path("README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- RESULTS_START -->", 1)[1].split("<!-- RESULTS_END -->", 1)[0]
    assert "The round-table team is not scored." in block
    assert "Those v3 scores are not in yet." in block
    assert "| Round-table team | not scored | | | |" in block
    assert "Round-table team | 0" not in block
    assert "Round-table team | 1" not in block
