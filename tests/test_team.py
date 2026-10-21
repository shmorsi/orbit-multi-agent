import json

from orbit_agents.engine import STOPPED, claims_are_grounded, extract_json, repair_message, run_single, run_team
from orbit_agents.schema import check_role, normalize_payload
from orbit_agents.scripted import ReviseOnce, ScriptedBackend
from orbit_agents.tasks import all_tasks, task_by_id


def _workspace(task_id, mode):
    from pathlib import Path

    return Path("traces") / "test-team" / mode / task_id


def test_scripted_team_and_single_solve_every_task():
    for task in all_tasks():
        for mode, runner in ("team", run_team), ("single", run_single):
            result = runner(task, ScriptedBackend(task), _workspace(task["id"], mode))
            if task["expect_refusal"]:
                assert result.status == "refuse", task["id"]
                assert result.claims == []
                assert not any(character.isdigit() for character in result.answer)
            else:
                assert result.status == "answer", task["id"]
                assert result.claims == task["gold_claims"]


def test_zero_is_a_real_answer():
    task = task_by_id("dev-021")
    result = run_team(task, ScriptedBackend(task), _workspace("dev-021", "team"))
    assert result.status == "answer"
    assert any(claim.get("value") == 0 for claim in result.claims)


def test_revise_once_keeps_the_gold_answer():
    task = task_by_id("dev-001")
    result = run_team(task, ReviseOnce(task), _workspace("dev-001", "revise"))
    assert any(event["kind"] == "revise" for event in result.events)
    assert result.status == "answer"
    assert result.answer == task["answer"]
    assert result.claims == task["gold_claims"]


class _EndlessTool:
    def complete(self, *, role, messages):
        del messages
        if role == "planner":
            return '{"steps": [{"goal": "keep going", "tool": "calculator"}]}'
        return '{"action": "tool", "tool": "calculator", "arguments": {"expression": "1+1"}}'


class _RepairsWhenTold(ScriptedBackend):
    def complete(self, *, role, messages):
        if role == "worker" and not any("Validator says:" in message["content"] for message in messages):
            return "not json at all"
        return super().complete(role=role, messages=messages)


def test_tune_retry_includes_the_validator_error():
    task = task_by_id("dev-018")
    result = run_team(task, _RepairsWhenTold(task), _workspace("dev-018", "repair"))
    assert result.status == "answer"
    assert result.schema_invalid >= 1
    assert any("Validator says:" in message["content"] for example in result.examples for message in example["messages"])
    assert "no JSON object" in repair_message("no JSON object")


def test_extra_planner_key_still_solves_a_tune_task():
    class _Extra(ScriptedBackend):
        def complete(self, *, role, messages):
            raw = super().complete(role=role, messages=messages)
            if role != "planner":
                return raw
            payload = json.loads(raw)
            payload["note"] = "thinking"
            return json.dumps(payload)

    task = task_by_id("dev-018")
    result = run_team(task, _Extra(task), _workspace("dev-018", "extra"))
    assert result.status == "answer"
    assert result.schema_invalid == 0
    assert result.claims == task["gold_claims"]


def test_near_miss_numbers_stay_grounded_and_unknown_numbers_do_not():
    results = [{"result": {"ok": True, "stdout": "5\n"}}]
    claims = [{"name": "job_count", "value": 5, "citation": "call_1"}]
    assert claims_are_grounded("How many?", claims, results, "job_count is 5.0.")
    assert not claims_are_grounded("How many?", claims, results, "job_count is 50.")
    payload = normalize_payload(
        "worker",
        {
            "action": "final",
            "status": "answer",
            "answer": "job_count is 5.",
            "note": "drop me",
            "claims": [{"name": "job_count", "value": 5.0, "citation": "call 1", "unit": "count"}],
        },
    )
    assert check_role("worker", payload) is None
    assert payload["claims"][0]["value"] == 5
    assert payload["claims"][0]["citation"] == "call_1"
    assert "note" not in payload
    parsed = extract_json('{"steps": [{"goal": "Count rows", "tool": "python",}],}')
    assert parsed["steps"][0]["tool"] == "python"


def test_step_limit_stops_a_model_that_never_answers():
    task = task_by_id("dev-001")
    result = run_team(task, _EndlessTool(), _workspace("dev-001", "limit"))
    assert result.n_steps == 8
    assert result.status == "refuse"
    assert result.answer == STOPPED
