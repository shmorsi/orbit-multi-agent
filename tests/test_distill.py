"""Teacher distillation: rejection filter, request cap, and the frozen-test blocklist."""

import json
import urllib.request
from pathlib import Path

import pytest

from distill_teacher import (
    DEFAULT_ATTEMPTS,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MODEL,
    DEFAULT_REQUESTS_PER_MINUTE,
    HIGH_DEMAND_STATUS,
    MINUTE_QUOTA_MAX_WAIT_SECONDS,
    PREFLIGHT_PROMPT,
    SERVER_ATTEMPTS,
    SERVER_BACKOFF_SECONDS,
    BudgetStop,
    DistillStop,
    FatalHTTPStop,
    FROZEN_FRAGMENTS,
    GeminiClient,
    NetworkStop,
    QuotaStop,
    RequestBudget,
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
    params = {
        "api_key": "test-key",
        "model": DEFAULT_MODEL,
        "budget": budget,
        "transport": transport,
        "log": lambda _event: None,
    }
    params.update(kwargs)
    return GeminiClient(**params)


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


def _fast_budget(max_requests=5):
    return RequestBudget(max_requests, 1000, clock=lambda: 0.0, sleep=lambda _seconds: None)


def test_http_400_stops_and_is_not_retried():
    calls = {"n": 0}

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        return 400, {"error": {"message": "bad request"}}

    client = _client(_fast_budget(), transport)
    with pytest.raises(FatalHTTPStop) as caught:
        client.generate([{"role": "user", "content": "Task: one"}])
    assert "HTTP 400" in str(caught.value)
    assert "bad request" in str(caught.value)
    assert "not retried" in str(caught.value)
    with pytest.raises(FatalHTTPStop):
        client.generate([{"role": "user", "content": "Task: two"}])
    assert calls["n"] == 1


def test_http_404_stops_with_the_model_message():
    calls = {"n": 0}
    message = (
        "models/gemini-2.5-flash is no longer available to new users, "
        "use models/gemini-3.8-flash"
    )

    def transport(url, headers, payload, timeout):
        del headers, payload, timeout
        calls["n"] += 1
        assert DEFAULT_MODEL in url
        return 404, {"error": {"code": 404, "message": message, "status": "NOT_FOUND"}}

    client = _client(_fast_budget(), transport)
    with pytest.raises(FatalHTTPStop) as caught:
        client.generate([{"role": "user", "content": "Task: one"}])
    assert message in str(caught.value)
    assert "HTTP 404" in str(caught.value)
    with pytest.raises(FatalHTTPStop):
        client.generate([{"role": "user", "content": "Task: two"}])
    assert calls["n"] == 1


def test_http_401_and_403_without_quota_stop_immediately():
    for status, message in ((401, "API key not valid"), (403, "permission denied")):
        calls = {"n": 0}

        def transport(url, headers, payload, timeout, status=status, message=message):
            del url, headers, payload, timeout
            calls["n"] += 1
            return status, {"error": {"message": message}}

        client = _client(_fast_budget(), transport)
        with pytest.raises(FatalHTTPStop) as caught:
            client.generate([{"role": "user", "content": "Task: one"}])
        assert f"HTTP {status}" in str(caught.value)
        assert message in str(caught.value)
        with pytest.raises(FatalHTTPStop):
            client.generate([{"role": "user", "content": "Task: two"}])
        assert calls["n"] == 1


def test_http_500_is_retried_a_bounded_number_of_times_then_stops():
    calls = {"n": 0}

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        return 500, {"error": {"message": "internal"}}

    client = _client(_fast_budget(max_requests=10), transport)
    with pytest.raises(NetworkStop) as caught:
        client.generate([{"role": "user", "content": "Task: one"}])
    assert "HTTP 500" in str(caught.value)
    assert calls["n"] == SERVER_ATTEMPTS
    with pytest.raises(NetworkStop):
        client.generate([{"role": "user", "content": "Task: two"}])
    assert calls["n"] == SERVER_ATTEMPTS


def _clocked_budget(max_requests=10, per_minute=1000.0):
    clock = {"t": 0.0}
    sleeps = []

    def now():
        return clock["t"]

    def sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds

    budget = RequestBudget(max_requests, per_minute, clock=now, sleep=sleep)
    return budget, sleeps


def _minute_429(delay=54):
    return {
        "error": {
            "status": "RESOURCE_EXHAUSTED",
            "message": (
                "Quota exceeded generate_content_free_tier_requests, "
                f"limit: 5 per minute. Please retry in {delay}s"
            ),
        }
    }


def test_http_503_waits_with_bounded_backoff_and_counts_each_request():
    calls = {"n": 0}
    logs = []
    budget, sleeps = _clocked_budget()

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        return 503, {"error": {"message": "high demand", "status": "UNAVAILABLE"}}

    client = _client(budget, transport, log=logs.append)
    with pytest.raises(NetworkStop) as caught:
        client.generate([{"role": "user", "content": "Task: one"}])
    assert "HTTP 503" in str(caught.value)
    assert "High demand" in str(caught.value)
    assert calls["n"] == len(SERVER_BACKOFF_SECONDS) + 1
    assert sleeps == list(SERVER_BACKOFF_SECONDS)
    assert budget.used == len(SERVER_BACKOFF_SECONDS) + 1
    request_logs = [row for row in logs if row["event"] == "request"]
    assert request_logs[-1]["status_counts"] == {"503": len(SERVER_BACKOFF_SECONDS) + 1}
    assert request_logs[-1]["configured_requests_per_minute"] == 1000
    assert request_logs[-1]["effective_requests_per_minute"] == pytest.approx(
        budget.used / (sum(SERVER_BACKOFF_SECONDS) / 60.0)
    )
    with pytest.raises(NetworkStop):
        client.generate([{"role": "user", "content": "Task: two"}])
    assert calls["n"] == len(SERVER_BACKOFF_SECONDS) + 1


def test_http_503_then_200_continues_after_one_backoff():
    calls = {"n": 0}
    budget, sleeps = _clocked_budget(per_minute=4)

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        if calls["n"] == 1:
            return 503, {"error": {"message": "high demand"}}
        return 200, _ok_body('{"ok": true}')

    client = _client(budget, transport)
    assert client.generate([{"role": "user", "content": "Task: one"}]) == '{"ok": true}'
    assert calls["n"] == 2
    assert budget.used == 2
    assert sleeps == [20]


def test_per_minute_429_waits_once_then_continues():
    calls = {"n": 0}
    logs = []
    budget, _sleeps = _clocked_budget(per_minute=4)

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        if calls["n"] in {1, 3}:
            return 429, _minute_429(54)
        return 200, _ok_body("{}")

    client = _client(budget, transport, log=logs.append)
    assert client.generate([{"role": "user", "content": "Task: one"}]) == "{}"
    assert client.generate([{"role": "user", "content": "Task: two"}]) == "{}"
    assert calls["n"] == 4
    assert budget.used == 4
    request_logs = [row for row in logs if row["event"] == "request"]
    assert request_logs[-1]["status_counts"]["429"] == 2
    assert request_logs[-1]["status_counts"]["200"] == 2
    assert request_logs[-1]["configured_requests_per_minute"] == 4
    waits = [row for row in logs if row["event"] == "wait"]
    assert [row["sleep_sec"] for row in waits] == [54, 54]


def test_second_consecutive_429_stops_without_a_second_wait():
    calls = {"n": 0}
    budget, sleeps = _clocked_budget()

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        return 429, _minute_429(54)

    client = _client(budget, transport)
    with pytest.raises(QuotaStop) as caught:
        client.generate([{"role": "user", "content": "Task: one"}])
    assert "second consecutive" in str(caught.value)
    assert sleeps == [54]
    assert calls["n"] == 2
    with pytest.raises(QuotaStop):
        client.generate([{"role": "user", "content": "Task: two"}])
    assert calls["n"] == 2


def test_per_day_429_stops_without_waiting():
    calls = {"n": 0}
    budget, sleeps = _clocked_budget()

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        return 429, {
            "error": {
                "message": "Quota exceeded generate_content_free_tier_requests, limit: 20 per day. Please retry in 3600s"
            }
        }

    client = _client(budget, transport)
    with pytest.raises(QuotaStop) as caught:
        client.generate([{"role": "user", "content": "Task: one"}])
    assert "per-day" in str(caught.value)
    assert sleeps == []
    assert calls["n"] == 1
    with pytest.raises(QuotaStop):
        client.generate([{"role": "user", "content": "Task: two"}])
    assert calls["n"] == 1


def test_long_retry_delay_stops_without_waiting():
    calls = {"n": 0}
    budget, sleeps = _clocked_budget()

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        return 429, _minute_429(121)

    client = _client(budget, transport)
    with pytest.raises(QuotaStop) as caught:
        client.generate([{"role": "user", "content": "Task: one"}])
    assert "pause limit" in str(caught.value)
    assert sleeps == []
    assert calls["n"] == 1


def test_retry_info_delay_is_honored_once():
    calls = {"n": 0}
    budget, sleeps = _clocked_budget()

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, {
                "error": {
                    "message": "Quota exceeded, limit: 5 per minute.",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "54s",
                        }
                    ],
                }
            }
        return 200, _ok_body("{}")

    client = _client(budget, transport)
    assert client.generate([{"role": "user", "content": "Task: one"}]) == "{}"
    assert sleeps == [54]
    assert calls["n"] == 2


def test_gemma_folds_the_system_prompt_into_the_user_turn():
    seen = {}

    def transport(url, headers, payload, timeout):
        del headers, timeout
        seen["url"] = url
        seen["payload"] = payload
        return 200, _ok_body("{}")

    client = _client(
        _fast_budget(),
        transport,
        model="gemma-4-31b-it",
    )
    client.generate(
        [
            {"role": "system", "content": "You are the planner."},
            {"role": "user", "content": "Task: count the trays."},
        ]
    )
    assert "gemma-4-31b-it" in seen["url"]
    assert "systemInstruction" not in seen["payload"]
    text = seen["payload"]["contents"][0]["parts"][0]["text"]
    assert text.startswith("You are the planner.")
    assert "Task: count the trays." in text


def test_gemini_keeps_the_system_instruction_field():
    seen = {}

    def transport(url, headers, payload, timeout):
        del url, headers, timeout
        seen["payload"] = payload
        return 200, _ok_body("{}")

    client = _client(_fast_budget(), transport, model="gemini-3.5-flash")
    client.generate(
        [
            {"role": "system", "content": "You are the planner."},
            {"role": "user", "content": "Task: count the trays."},
        ]
    )
    assert seen["payload"]["systemInstruction"]["parts"][0]["text"] == "You are the planner."
    assert seen["payload"]["contents"][0]["parts"][0]["text"] == "Task: count the trays."


def test_per_day_429_saves_a_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("ORBIT_TEACHER_API_KEY", "test-key")
    task = task_by_id("dev-001")
    monkeypatch.setattr("distill_teacher.load_pool", lambda _workspace: [dict(task)])
    calls = {"n": 0}

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        return 429, {
            "error": {
                "message": "Quota exceeded generate_content_free_tier_requests, limit: 20 per day."
            }
        }

    output = tmp_path / "out"
    with pytest.raises(SystemExit) as caught:
        main(
            ["--output", str(output), "--max-requests", "20", "--requests-per-minute", "1000"],
            transport=transport,
            clock=lambda: 0.0,
            sleep=lambda _seconds: None,
        )
    assert caught.value.code == 2
    assert calls["n"] == 1
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["requests"] == 1
    assert checkpoint["stopped_reason"] == "quota"
    assert checkpoint["tasks"] == {}
    assert checkpoint["status_counts"]["429"] == 1
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert stats["status_counts"]["429"] == 1


def test_http_500_then_200_returns_the_success():
    calls = {"n": 0}

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        if calls["n"] == 1:
            return 500, {"error": {"message": "internal"}}
        return 200, _ok_body('{"ok": true}')

    client = _client(_fast_budget(), transport)
    assert client.generate([{"role": "user", "content": "Task: one"}]) == '{"ok": true}'
    assert calls["n"] == 2


def test_client_error_stops_the_run_instead_of_the_next_attempt(tmp_path):
    calls = {"n": 0}

    class Backend:
        def complete(self, *, role, messages):
            del role, messages
            calls["n"] += 1
            raise FatalHTTPStop("HTTP 404 from model gemini-2.5-flash: model is no longer available")

    tasks = [task_by_id("dev-001"), task_by_id("dev-002")]
    state = run_distill(
        tasks,
        _read(TEST),
        backend_for=lambda _task: Backend(),
        attempts=3,
        workspace=tmp_path,
    )
    assert calls["n"] == 1
    assert state["stopped_reason"] == "http"
    assert "no longer available" in state["stopped_detail"]
    assert state["tasks"]["dev-001"]["attempts"] == 0
    assert "dev-002" not in state["tasks"]


def test_preflight_404_stops_before_any_task(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ORBIT_TEACHER_API_KEY", "test-key")
    task = task_by_id("dev-001")
    monkeypatch.setattr("distill_teacher.load_pool", lambda _workspace: [dict(task)])
    message = (
        "models/gemini-2.5-flash is no longer available to new users, "
        "use models/gemini-3.8-flash"
    )
    calls = []

    def transport(url, headers, payload, timeout):
        del headers, timeout
        calls.append((url, payload))
        return 404, {"error": {"code": 404, "message": message, "status": "NOT_FOUND"}}

    output = tmp_path / "out"
    with pytest.raises(SystemExit) as caught:
        main(
            ["--output", str(output), "--max-requests", "20", "--requests-per-minute", "1000"],
            transport=transport,
            clock=lambda: 0.0,
            sleep=lambda _seconds: None,
        )
    assert caught.value.code == 1
    assert len(calls) == 1
    assert DEFAULT_MODEL in calls[0][0]
    blob = json.dumps(calls[0][1])
    assert PREFLIGHT_PROMPT in blob
    assert task["id"] not in blob
    assert task["question"] not in blob
    captured = capsys.readouterr()
    assert message in captured.out
    assert message in captured.err
    assert "No further requests will be sent." in captured.err
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["requests"] == 1
    assert checkpoint["stopped_reason"] == "http"
    assert checkpoint["tasks"] == {}


def test_task_404_after_preflight_does_not_keep_spending(monkeypatch, tmp_path):
    monkeypatch.setenv("ORBIT_TEACHER_API_KEY", "test-key")
    task = task_by_id("dev-001")
    monkeypatch.setattr("distill_teacher.load_pool", lambda _workspace: [dict(task)])
    calls = {"n": 0}

    def transport(url, headers, payload, timeout):
        del url, headers, payload, timeout
        calls["n"] += 1
        if calls["n"] == 1:
            return 200, _ok_body("{}")
        return 404, {"error": {"message": "model is no longer available"}}

    with pytest.raises(SystemExit) as caught:
        main(
            ["--output", str(tmp_path / "out"), "--max-requests", "20", "--requests-per-minute", "1000"],
            transport=transport,
            clock=lambda: 0.0,
            sleep=lambda _seconds: None,
        )
    assert caught.value.code == 1
    assert calls["n"] == 2
    checkpoint = json.loads((tmp_path / "out" / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["requests"] == 2
    assert checkpoint["stopped_reason"] == "http"
    assert checkpoint["tasks"][task["id"]]["attempts"] == 0


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
    assert config["teacher_requests_per_minute"] == DEFAULT_REQUESTS_PER_MINUTE == args.requests_per_minute == 4
    assert config["server_backoff_seconds"] == list(SERVER_BACKOFF_SECONDS) == [20, 40, 80]
    assert config["minute_quota_max_wait_seconds"] == MINUTE_QUOTA_MAX_WAIT_SECONDS == 120
    assert config["high_demand_status"] == HIGH_DEMAND_STATUS == 503
    assert config["teacher_attempts"] == DEFAULT_ATTEMPTS == args.attempts == 3
    assert config["teacher_model"] == DEFAULT_MODEL == args.model == "gemini-3.8-flash"
    assert config["teacher_model_generation"] == 3.8
    assert config["server_error_attempts"] == SERVER_ATTEMPTS == 3
    assert config["stop_on_http_status"] == 429
    assert config["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    train_path, tune_path = data_paths(config)
    assert train_path.name == "train.jsonl"
    assert tune_path.name == "tune.jsonl"
    assert "distill" in train_path.parts
    with pytest.raises(SystemExit):
        assert_training_file(Path("eval/test.jsonl"))
