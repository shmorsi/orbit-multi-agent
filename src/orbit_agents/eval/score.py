"""Score a run against gold that was computed from the tools."""

from __future__ import annotations

import re

from orbit_agents.engine import claims_are_grounded

DIGIT_RE = re.compile(r"\d")


def _claim_key(claim: dict) -> tuple:
    if "value" in claim:
        return (claim["name"], "value", int(claim["value"]), claim["citation"])
    return (claim["name"], "text", claim["text"], claim["citation"])


def score_task(task: dict, result) -> dict:
    valid = int(getattr(result, "schema_valid"))
    invalid = int(getattr(result, "schema_invalid"))
    total = valid + invalid
    schema_rate = (valid / total) if total else None
    if task["expect_refusal"]:
        success = (
            result.status == "refuse"
            and not result.claims
            and DIGIT_RE.search(result.answer) is None
        )
    else:
        gold = [_claim_key(claim) for claim in task["gold_claims"]]
        got = [_claim_key(claim) for claim in result.claims]
        success = (
            result.status == "answer"
            and gold == got
            and claims_are_grounded(task["question"], result.claims, _tool_results(result), result.answer)
        )
    return {
        "id": task["id"],
        "success": bool(success),
        "n_steps": int(result.n_steps),
        "schema_valid": valid,
        "schema_invalid": invalid,
        "schema_valid_rate": schema_rate,
        "n_tool_calls": len(result.tool_calls),
        "expect_refusal": bool(task["expect_refusal"]),
        "correct_refusal": bool(success) if task["expect_refusal"] else None,
        "status": result.status,
    }


def _tool_results(result) -> list[dict]:
    rows = []
    for event in result.events:
        if event["kind"] == "result" and event["agent"] == "tool":
            rows.append({"tool": event["tool"], "arguments": {}, "result": event["result"]})
    return rows
