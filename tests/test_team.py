from orbit_agents.engine import STOPPED, run_single, run_team
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


def test_step_limit_stops_a_model_that_never_answers():
    task = task_by_id("dev-001")
    result = run_team(task, _EndlessTool(), _workspace("dev-001", "limit"))
    assert result.n_steps == 8
    assert result.status == "refuse"
    assert result.answer == STOPPED
