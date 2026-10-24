"""Orbit Boardroom. Scripted demo. The auditor refuses a figure the CSV does not state."""

import json
from pathlib import Path

from orbit_agents.boardroom import (
    REVENUE_DELTA,
    activation_rate,
    check_claim,
    corpus_manifest,
    freeze_date,
    launch_crew,
    q4_operating_cost,
    q4_revenue,
    run_boardroom,
    task_by_id,
)
from orbit_agents.scripted_boardroom import BoardroomPolicy
from orbit_agents.trace import bundled_traces_boardroom


def test_auditor_refuses_the_planted_revenue_and_keeps_the_files():
    task = task_by_id("board-launch")
    result = run_boardroom(task, BoardroomPolicy(task))
    assert result.status == "answer"
    assert result.gate_overrode is True
    audit = next(event for event in result.events if event["kind"] == "audit")
    assert audit["verdict"] == "refuse"
    by_name = {finding["claim"]: finding for finding in audit["findings"]}
    revenue = by_name["q4_revenue_usd"]
    assert revenue["ok"] is False
    assert revenue["actual"] == q4_revenue()
    assert revenue["claimed"] == q4_revenue() + REVENUE_DELTA
    assert by_name["q4_operating_cost_usd"]["ok"] is True
    assert by_name["q4_operating_cost_usd"]["actual"] == q4_operating_cost()
    assert by_name["activation_rate_pct"]["actual"] == activation_rate()
    assert by_name["launch_crew"]["actual"] == launch_crew()
    assert by_name["freeze_date"]["actual"] == freeze_date()
    assert str(revenue["claimed"]) not in result.answer
    assert str(q4_revenue()) not in result.answer
    assert str(q4_operating_cost()) in result.answer
    assert "auditor refused q4_revenue_usd" in result.answer
    assert all(claim["name"] != "q4_revenue_usd" for claim in result.claims)
    roles = [event["agent"] for event in result.events if event["kind"] == "parsed"]
    assert roles == ["ceo", "cfo", "cto", "coo"]


def test_check_claim_uses_the_cited_file():
    matched = check_claim(
        {
            "name": "q4_revenue_usd",
            "value": q4_revenue(),
            "source": "financials.csv",
            "quarter": "2025Q4",
            "column": "revenue_usd",
        }
    )
    assert matched["ok"] is True
    missed = check_claim(
        {
            "name": "q4_revenue_usd",
            "value": q4_revenue() + REVENUE_DELTA,
            "source": "financials.csv",
            "quarter": "2025Q4",
            "column": "revenue_usd",
        }
    )
    assert missed["ok"] is False
    assert missed["actual"] == q4_revenue()


def test_corpus_manifest_matches_the_files():
    fresh = corpus_manifest()
    committed = json.loads(Path("data/scenarios/boardroom/manifest.json").read_text(encoding="utf-8"))
    assert fresh == committed
    assert committed["fictional"] is True
    corpus = Path("data/scenarios/boardroom/corpus")
    for relative, record in committed["files"].items():
        body = (corpus / relative).read_bytes()
        assert len(body) == record["bytes"]


def test_scripted_boardroom_trace_is_labeled_and_matches_the_engine(tmp_path):
    written = bundled_traces_boardroom(tmp_path)
    path = written[0]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    header = rows[0]
    assert header["fabricated"] is False
    assert header["harness"] == "scripted_harness"
    assert header["scenario"] == "boardroom"
    assert "fictional" in header["scenario_note"].lower()
    assert "not a model score" in header["scenario_note"].lower()
    assert header["status"] == "answer"
    assert header["gate_overrode"] is True
    audit = next(event for event in rows[1:] if event["kind"] == "audit")
    assert audit["verdict"] == "refuse"
    assert any(finding["claim"] == "q4_revenue_usd" and finding["ok"] is False for finding in audit["findings"])
    committed = Path("mission/traces/boardroom/memo.jsonl")
    assert path.read_bytes() == committed.read_bytes()
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index == json.loads(Path("mission/traces/boardroom/index.json").read_text(encoding="utf-8"))
    blob = path.read_text(encoding="utf-8").lower()
    for word in ("gdp", "inflation", "world bank", "macro_agent"):
        assert word not in blob
