"""Incident commander demo. Oracles match the files. The trap is refused. Scores are untouched."""

import json
from pathlib import Path

from orbit_agents.engine import claims_are_grounded
from orbit_agents.incident import (
    INVENTORY_BLAME,
    breach_minutes,
    failed_orders,
    incident_tasks,
    inventory_errors,
    page_minutes,
    page_threshold,
    payment_checkout_errors,
    preceding_deploy,
    task_by_id,
    use_incident_corpus,
)
from orbit_agents.scripted_incident import IncidentPolicy
from orbit_agents.team_v3 import run_team_v3
from orbit_agents.tools import call_tool


def _answers() -> dict[str, int | str]:
    return {
        "payment_checkout_errors": payment_checkout_errors(),
        "breach_minutes": breach_minutes(),
        "failed_orders": failed_orders(),
        "deploy": preceding_deploy(),
    }


def test_incident_files_agree_with_the_tasks():
    facts = _answers()
    assert facts["payment_checkout_errors"] > 0
    assert facts["failed_orders"] > facts["payment_checkout_errors"]
    assert facts["breach_minutes"] >= page_minutes()
    assert page_threshold() > 0
    assert inventory_errors() == 0
    assert INVENTORY_BLAME != inventory_errors()
    assert "revenue" not in Path("data/scenarios/incident/corpus/metrics.csv").read_text(encoding="utf-8").lower()
    tasks = {task["id"]: task for task in incident_tasks()}
    assert set(tasks) == {
        "inc-report",
        "inc-payment-errors",
        "inc-breach",
        "inc-failed",
        "inc-deploy",
        "inc-revenue",
    }
    report = tasks["inc-report"]
    by_name = {claim["name"]: claim for claim in report["gold_claims"]}
    assert by_name["payment_checkout_errors"]["value"] == facts["payment_checkout_errors"]
    assert by_name["breach_minutes"]["value"] == facts["breach_minutes"]
    assert by_name["failed_orders"]["value"] == facts["failed_orders"]
    assert by_name["deploy"]["text"] == facts["deploy"]
    assert tasks["inc-revenue"]["expect_refusal"] is True
    assert tasks["inc-revenue"]["status"] == "refuse"
    assert tasks["inc-revenue"]["gold_claims"] == []
    assert "eval/test.jsonl" not in Path("data/scenarios/incident/tasks.jsonl").read_text(encoding="utf-8")


def test_scripted_oracles_match_the_files_and_the_trap_is_refused(tmp_path):
    facts = _answers()
    with use_incident_corpus():
        for task in incident_tasks():
            if task["status"] == "refuse":
                continue
            results = []
            for call in task["oracle_calls"]:
                results.append({"result": call_tool(call["tool"], call["arguments"], tmp_path)})
            assert claims_are_grounded(task["question"], task["gold_claims"], results, task["answer"])
        report = task_by_id("inc-report")
        result = run_team_v3(report, IncidentPolicy(report), tmp_path / "report")
        assert result.status == "answer"
        assert result.claims == report["gold_claims"]
        assert str(facts["payment_checkout_errors"]) in result.answer
        gate = next(event for event in result.events if event["kind"] == "gate")
        assert gate["code_verdict"] == "refuse"
        assert gate["claim"]["name"] == "inventory_errors"
        assert gate["claim"]["value"] == INVENTORY_BLAME
        assigns = [event for event in result.events if event["kind"] == "assign"]
        assert any(event["role"] == "engineer" and event["tool"] == "python" for event in assigns)
        revenue = task_by_id("inc-revenue")
        refused = run_team_v3(revenue, IncidentPolicy(revenue), tmp_path / "revenue")
        assert refused.status == "refuse"
        assert refused.claims == []
        assert "revenue" in refused.answer.lower()


def test_committed_incident_traces_are_scripted_protocol_demos(tmp_path):
    from orbit_agents.trace import bundled_traces_incident

    written = bundled_traces_incident(tmp_path)
    assert [path.name for path in written] == [
        "report.jsonl",
        "payment.jsonl",
        "breach.jsonl",
        "failed.jsonl",
        "deploy.jsonl",
        "revenue.jsonl",
    ]
    for path in written:
        committed = Path("mission/traces/incident") / path.name
        assert path.read_bytes() == committed.read_bytes()
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert header["fabricated"] is False
        assert header["source"] == "orbit_agents.engine"
        assert header["harness"] == "scripted_harness"
        assert header["scenario"] == "incident"
        assert "not a model score" in header["scenario_note"].lower()
    report = json.loads(Path("mission/traces/incident/report.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert report["status"] == "answer"
    body = Path("mission/traces/incident/report.jsonl").read_text(encoding="utf-8")
    assert "inventory service is the root cause" in body.lower()
    index = json.loads(Path("mission/traces/incident/index.json").read_text(encoding="utf-8"))
    assert index[0]["file"] == "report.jsonl"
    assert "scripted protocol demo" in index[0]["note"]
    script = Path("mission/replay.js").read_text(encoding="utf-8")
    assert "incident/report.jsonl" in script
    assert "Incident commander" in script
