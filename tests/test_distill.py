"""Teacher distillation: rejection filter, request cap, and the frozen-test blocklist."""

import json
import urllib.request
from pathlib import Path

import pytest

from distill_teacher import (
    DEFAULT_ATTEMPTS,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_REQUESTS_PER_MINUTE,
    BudgetStop,
    DistillStop,
    FROZEN_FRAGMENTS,
    GeminiClient,
    QuotaStop,
    RequestBudget,
    TeacherHTTPError,
    build_parser,
    load_checkpoint,
    load_pool,
    main,
    rows_from_result,
    run_distill,
    select_tasks,
    trajectory_ok,
    uses_frozen_oracle,
)
from leakage import TEST, _read, overlap_error
from orbit_agents.engine import run_team
from orbit_agents.scripted import ScriptedBackend
from orbit_agents.tasks import task_by_id
from train import assert_training_file, data_paths


def _ok_body(text="{}"):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
    }


def _client(budget, transport, **kwargs):
    return GeminiClient(
        api_key="test-key",
        model="gemini-2.5-flash",
        budget=budget,
        transport=transport,
        log=lambda _event: None,
        **kwargs,
    )


def test_filter_keeps_checker_passes_and_drops_schema_or_checker_failures(tmp_path):
    task = task_by_id("dev-001")
    result = run_team(task, ScriptedBackend(task), tmp_path)
    assert trajectory_ok(task, result) == (True, "ok")
    rows = rows_from_result(task, result)
    assert {row["role"] for row in rows} >= {"planner", "worker", "critic"}
    assert overlap_error(rows, [], _read(TEST)) is None
    result.schema_invalid = 1
    assert trajectory_ok(task, result) == (False, "schema")
    result.schema_invalid = 0
    result.status = "refuse"
    result.claims = []
    result.answer = "no"
    assert trajectory_ok(task, result)[0] is False


def test_filter_keeps_a_correct_refusal(tmp_path):
    task = task_by_id("dev-015")
    result = run_team(task, ScriptedBackend(task), tmp_path)
    assert task["expect_refusal"] is True
    assert trajectory_ok(task, result) == (True, "ok")


def test_saved_rows_reject_a_copied_frozen_answer():
    test_rows = _read(TEST)
    answer = test_rows[0]["answer"]
    leaked = [
        {
            "id": "var-x",
            "role": "worker",
            "messages": [{"role": "assistant", "content": answer}],
        }
    ]
    assert "test answer leaked" in overlap_error(leaked, [], test_rows)


def test_cap_and_rate_limit_stop_at_the_ceiling():
    clock = {"t": 1000.0}
    sleeps = []

    def now():
        return clock["t"]

    def sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds

    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(url)
        assert "key=" not in url
        assert headers["x-goog-api-key"] == "test-key"
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        return 200, _ok_body()

    budget = RequestBudget(3, 8, clock=now, sleep=sleep)
    client = _client(budget, transport)
    for _ in range(3):
        assert client.generate([{"role": "user", "content": "Task: count the trays."}]) == "{}"
    with pytest.raises(BudgetStop):
        client.generate([{"role": "user", "content": "Task: again"}])
    assert len(calls) == 3
    assert sleeps == pytest.approx([7.5, 7.5])
    assert budget.total_tokens == 18
    assert budget.used == 3


def test_http_429_stops_and_is_not_retried():
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(seconds):
        clock["t"] += seconds

    calls = {"n": 0}

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        if calls["n"] == 1:
            return 200, _ok_body()
        return 429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}}

    client = _client(RequestBudget(10, 8, clock=now, sleep=sleep), transport)
    client.generate([{"role": "user", "content": "Task: one"}])
    with pytest.raises(QuotaStop):
        client.generate([{"role": "user", "content": "Task: two"}])
    with pytest.raises(QuotaStop):
        client.generate([{"role": "user", "content": "Task: three"}])
    assert calls["n"] == 2


def test_quota_body_on_403_stops_without_another_request():
    calls = {"n": 0}

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        return 403, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded for this project"}}

    client = _client(
        RequestBudget(5, 1000, clock=lambda: 0.0, sleep=lambda _seconds: None),
        transport,
    )
    with pytest.raises(QuotaStop):
        client.generate([{"role": "user", "content": "Task: one"}])
    with pytest.raises(QuotaStop):
        client.generate([{"role": "user", "content": "Task: two"}])
    assert calls["n"] == 1


def test_http_400_is_not_retried_inside_the_client():
    calls = {"n": 0}

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        return 400, {"error": {"message": "bad request"}}

    client = _client(
        RequestBudget(5, 1000, clock=lambda: 0.0, sleep=lambda _seconds: None),
        transport,
    )
    with pytest.raises(TeacherHTTPError):
        client.generate([{"role": "user", "content": "Task: one"}])
    with pytest.raises(TeacherHTTPError):
        client.generate([{"role": "user", "content": "Task: two"}])
    assert calls["n"] == 2


def test_resume_counts_spent_requests_against_the_cap():
    calls = {"n": 0}

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        return 200, _ok_body()

    budget = RequestBudget(3, 1000, clock=lambda: 50.0, sleep=lambda _seconds: None, used=2, last_request_at=50.0)
    client = _client(budget, transport)
    client.generate([{"role": "user", "content": "Task: one more"}])
    with pytest.raises(BudgetStop):
        client.generate([{"role": "user", "content": "Task: over the cap"}])
    assert calls["n"] == 1
    assert budget.used == 3


def test_checkpoint_resume_does_not_call_the_teacher_again(tmp_path):
    task = task_by_id("dev-015")
    test_rows = _read(TEST)
    state = run_distill(
        [task],
        test_rows,
        backend_for=lambda item: ScriptedBackend(item),
        attempts=1,
        workspace=tmp_path / "runs",
    )
    assert state["tasks"][task["id"]]["accepted"] is True
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    loaded = load_checkpoint(path)

    def backend_for(_task):
        raise AssertionError("a finished task must not call the teacher on resume")

    again = run_distill(
        [task],
        test_rows,
        backend_for=backend_for,
        attempts=1,
        workspace=tmp_path / "runs2",
        state=loaded,
    )
    assert again["tasks"][task["id"]]["accepted"] is True
    assert again["train_rows"]


def test_guard_stops_before_the_backend_if_a_frozen_task_is_forced(tmp_path):
    test_rows = _read(TEST)
    calls = []

    class Boom:
        def complete(self, *, role, messages):
            del role, messages
            calls.append(1)
            raise AssertionError("frozen test reached the backend")

    with pytest.raises(DistillStop) as caught:
        run_distill(
            [test_rows[0]],
            test_rows,
            backend_for=lambda _task: Boom(),
            attempts=1,
            workspace=tmp_path,
        )
    assert caught.value.reason == "leakage"
    assert calls == []


def test_pool_excludes_the_frozen_test_and_grows_past_the_templated_train_set(tmp_path):
    pool = load_pool(tmp_path / "pool")
    test_rows = _read(TEST)
    kept, dropped = select_tasks(pool, test_rows)
    train = [task for task in kept if task["subset"] == "train"]
    assert len(train) > 69
    assert any(task.get("expect_refusal") for task in kept)
    dropped_ids = {row["id"] for row in dropped}
    assert {"dev-005", "dev-017"} <= dropped_ids
    blob = json.dumps(kept)
    for task in test_rows:
        assert task["id"] not in blob
        assert task["question"] not in blob
    for fragment in FROZEN_FRAGMENTS:
        assert fragment not in blob
    for task in kept:
        assert uses_frozen_oracle(task, test_rows) is False
        assert not str(task["id"]).startswith("test-")


def test_dry_run_makes_no_network_call(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("dry-run called the network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    report = main(["--dry-run"])
    assert report["dry_run"] is True
    assert report["requests"] == 0
    assert report["test_sent_to_teacher"] is False
    assert report["n_train_tasks"] > 69
    assert report["n_refusals"] >= 1
    text = json.dumps(report)
    for task in _read(TEST):
        assert task["question"] not in text
        assert task["id"] not in text


def test_budget_defaults_match_the_distill_config():
    config = json.loads(Path("finetune/configs/qlora_distill.json").read_text(encoding="utf-8"))
    args = build_parser().parse_args([])
    assert config["teacher_max_requests"] == DEFAULT_MAX_REQUESTS == args.max_requests == 800
    assert config["teacher_requests_per_minute"] == DEFAULT_REQUESTS_PER_MINUTE == args.requests_per_minute == 8
    assert config["teacher_attempts"] == DEFAULT_ATTEMPTS == args.attempts == 3
    assert config["teacher_model"] == "gemini-2.5-flash"
    assert config["stop_on_http_status"] == 429
    assert config["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    train_path, tune_path = data_paths(config)
    assert train_path.name == "train.jsonl"
    assert tune_path.name == "tune.jsonl"
    assert "distill" in train_path.parts
    with pytest.raises(SystemExit):
        assert_training_file(Path("eval/test.jsonl"))
