"""Write the fictional checkout-outage corpus and its demo tasks.

The files are the evidence. tasks.jsonl stores the oracle that the scripted
harness follows. Nothing here is a frozen-test score.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

SCENARIO = Path(__file__).resolve().parent
CORPUS = SCENARIO / "corpus"
MANIFEST = SCENARIO / "manifest.json"
TASKS = SCENARIO / "tasks.jsonl"

# Payment error counts by minute. Rate is errors * 100 // 20. Above 5 only from 02:05 through 02:12.
PAYMENT_ERRORS = {
    "02:00": 0,
    "02:01": 1,
    "02:02": 0,
    "02:03": 1,
    "02:04": 1,
    "02:05": 2,
    "02:06": 4,
    "02:07": 6,
    "02:08": 5,
    "02:09": 4,
    "02:10": 3,
    "02:11": 3,
    "02:12": 2,
    "02:13": 1,
    "02:14": 0,
}
PAYMENT_REQUESTS = 20
# Checkout failures inside the window. Each line is one failed order.
CHECKOUT_FAILURES = (
    ("02:05", "PAYMENT_UPSTREAM", 2),
    ("02:06", "PAYMENT_UPSTREAM", 2),
    ("02:06", "GATEWAY_TIMEOUT", 1),
    ("02:07", "PAYMENT_UPSTREAM", 3),
    ("02:07", "GATEWAY_TIMEOUT", 1),
    ("02:08", "PAYMENT_UPSTREAM", 2),
    ("02:08", "GATEWAY_TIMEOUT", 1),
    ("02:09", "PAYMENT_UPSTREAM", 1),
    ("02:09", "GATEWAY_TIMEOUT", 1),
    ("02:10", "PAYMENT_UPSTREAM", 1),
    ("02:11", "PAYMENT_UPSTREAM", 1),
)
GATEWAY_ERRORS = {"02:06": 1, "02:07": 1, "02:08": 1, "02:09": 1}
RUNBOOK = """# Checkout on-call

Page the payment primary when the payment-service error rate stays above 5 percent for 3 consecutive minutes.

Error rate is the integer errors times 100 floor-divided by requests, from metrics.csv, one row per minute.

The incident window is 2026-03-14T02:05:00Z through 2026-03-14T02:12:59Z.

This book does not list revenue.
"""
STATUS = """# Status

Checkout is degraded on the night of the incident.
The error spike began at 2026-03-14T02:05:00Z.
Payment-service is the system under investigation.
No revenue figure is published on this page.
"""
DEPLOYS = """2026-03-14T01:12:00Z checkout-service deploy chk-220
2026-03-14T01:40:00Z payment-service deploy pay-1844
2026-03-14T02:04:00Z payment-service deploy pay-1847
2026-03-14T02:20:00Z payment-service rollback pay-1844
"""


def _minute_lines(service: str, code: str, counts: dict[str, int]) -> list[str]:
    lines = []
    for minute, count in counts.items():
        for index in range(count):
            second = index + 1
            lines.append(f"2026-03-14T{minute}:{second:02d}Z ERROR {service} {code}")
    return lines


def _checkout_lines() -> list[str]:
    lines = []
    order = 9000
    for minute, code, count in CHECKOUT_FAILURES:
        for index in range(count):
            order += 1
            second = index + 1
            lines.append(
                f"2026-03-14T{minute}:{second:02d}Z ERROR checkout {code} ORDER_FAILED order={order}"
            )
    return lines


def _metrics() -> str:
    rows = ["minute,service,requests,errors,latency_ms"]
    for minute, payment_errors in PAYMENT_ERRORS.items():
        stamp = f"2026-03-14T{minute}:00Z"
        hot = minute >= "02:05" and minute <= "02:12"
        checkout_errors = sum(count for item_minute, _code, count in CHECKOUT_FAILURES if item_minute == minute)
        rows.append(f"{stamp},checkout,40,{checkout_errors},{200 if hot else 70}")
        rows.append(f"{stamp},payment,{PAYMENT_REQUESTS},{payment_errors},{420 if hot else 90}")
        rows.append(f"{stamp},inventory,20,0,35")
        rows.append(f"{stamp},gateway,20,{GATEWAY_ERRORS.get(minute, 0)},{140 if hot else 80}")
    return "\n".join(rows) + "\n"


def _inventory_log() -> str:
    lines = [f"2026-03-14T{minute}:00Z INFO inventory healthy" for minute in PAYMENT_ERRORS]
    return "\n".join(lines) + "\n"


def write_corpus() -> None:
    """Write the evidence files. The manifest is hashed after this returns."""
    (CORPUS / "logs").mkdir(parents=True, exist_ok=True)
    (CORPUS / "docs").mkdir(parents=True, exist_ok=True)
    (CORPUS / "metrics.csv").write_text(_metrics(), encoding="utf-8")
    (CORPUS / "logs" / "checkout.log").write_text("\n".join(_checkout_lines()) + "\n", encoding="utf-8")
    (CORPUS / "logs" / "payment.log").write_text(
        "\n".join(_minute_lines("payment", "PAYMENT_TIMEOUT", PAYMENT_ERRORS)) + "\n",
        encoding="utf-8",
    )
    (CORPUS / "logs" / "inventory.log").write_text(_inventory_log(), encoding="utf-8")
    (CORPUS / "logs" / "gateway.log").write_text(
        "\n".join(_minute_lines("gateway", "GATEWAY_TIMEOUT", GATEWAY_ERRORS)) + "\n",
        encoding="utf-8",
    )
    (CORPUS / "logs" / "deploys.log").write_text(DEPLOYS, encoding="utf-8")
    (CORPUS / "docs" / "runbook.md").write_text(RUNBOOK, encoding="utf-8")
    (CORPUS / "docs" / "status.md").write_text(STATUS, encoding="utf-8")
    cache = {
        "pages": [
            {
                "query": "checkout status",
                "title": "Status",
                "snippet": "Checkout is degraded. Payment-service is under investigation. No revenue figure is published.",
            }
        ]
    }
    (CORPUS / "web_cache.json").write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")


def _task(task_id: str, question: str, plan: list, calls: list, claims: list, answer: str, status: str) -> dict:
    return {
        "id": task_id,
        "split": "demo",
        "subset": "incident",
        "question": question,
        "expect_refusal": status == "refuse",
        "plan": plan,
        "oracle_calls": calls,
        "gold_claims": claims,
        "answer": answer,
        "status": status,
    }


def write_tasks() -> None:
    """Oracle tasks. Counts are the strings the python cells print, filled after the files exist."""
    from orbit_agents.incident import (
        breach_minutes,
        failed_orders,
        payment_checkout_errors,
        preceding_deploy,
    )

    errors = payment_checkout_errors()
    minutes = breach_minutes()
    failed = failed_orders()
    deploy = preceding_deploy()
    count_errors = (
        "body = text('logs/checkout.log')\n"
        "print(sum(1 for line in body.splitlines() if 'PAYMENT_UPSTREAM' in line))"
    )
    count_minutes = (
        "body = text('metrics.csv')\n"
        "count = 0\n"
        "for line in body.splitlines()[1:]:\n"
        "    parts = line.split(',')\n"
        "    if len(parts) < 4 or parts[1] != 'payment':\n"
        "        continue\n"
        "    errors = int(parts[3])\n"
        "    requests = int(parts[2])\n"
        "    if requests > 0 and errors * 100 // requests > 5:\n"
        "        count += 1\n"
        "print(count)"
    )
    count_failed = (
        "body = text('logs/checkout.log')\n"
        "print(sum(1 for line in body.splitlines() if 'ORDER_FAILED' in line))"
    )
    read_deploy = {"tool": "read_file", "arguments": {"path": "logs/deploys.log"}}
    tasks = [
        _task(
            "inc-report",
            "On the night of the incident, how many checkout errors came from payment-service, "
            "for how many minutes did the payment-service error rate exceed the runbook page threshold, "
            "how many orders failed, and which payment-service deploy immediately preceded the spike?",
            [
                {"goal": "Count checkout errors from payment-service", "tool": "python"},
                {"goal": "Count minutes above the runbook page threshold", "tool": "python"},
                {"goal": "Count failed orders", "tool": "python"},
                {"goal": "Read the deploy that preceded the spike", "tool": "read_file"},
            ],
            [
                {"tool": "python", "arguments": {"code": count_errors}},
                {"tool": "python", "arguments": {"code": count_minutes}},
                {"tool": "python", "arguments": {"code": count_failed}},
                read_deploy,
            ],
            [
                {"name": "payment_checkout_errors", "citation": "call_1", "value": errors, "unit": "count"},
                {"name": "breach_minutes", "citation": "call_2", "value": minutes, "unit": "count"},
                {"name": "failed_orders", "citation": "call_3", "value": failed, "unit": "count"},
                {"name": "deploy", "citation": "call_4", "text": deploy},
            ],
            (
                f"payment_checkout_errors is {errors}. breach_minutes is {minutes}. "
                f"failed_orders is {failed}. deploy is {deploy}."
            ),
            "answer",
        ),
        _task(
            "inc-payment-errors",
            "How many checkout errors came from payment-service on the night of the incident?",
            [{"goal": "Count checkout errors from payment-service", "tool": "python"}],
            [{"tool": "python", "arguments": {"code": count_errors}}],
            [{"name": "payment_checkout_errors", "citation": "call_1", "value": errors, "unit": "count"}],
            f"payment_checkout_errors is {errors}.",
            "answer",
        ),
        _task(
            "inc-breach",
            "For how many minutes did the payment-service error rate exceed the runbook page threshold?",
            [{"goal": "Count minutes above the runbook page threshold", "tool": "python"}],
            [{"tool": "python", "arguments": {"code": count_minutes}}],
            [{"name": "breach_minutes", "citation": "call_1", "value": minutes, "unit": "count"}],
            f"breach_minutes is {minutes}.",
            "answer",
        ),
        _task(
            "inc-failed",
            "How many orders failed on the night of the incident?",
            [{"goal": "Count failed orders", "tool": "python"}],
            [{"tool": "python", "arguments": {"code": count_failed}}],
            [{"name": "failed_orders", "citation": "call_1", "value": failed, "unit": "count"}],
            f"failed_orders is {failed}.",
            "answer",
        ),
        _task(
            "inc-deploy",
            "Which payment-service deploy immediately preceded the error spike?",
            [{"goal": "Read the deploy log", "tool": "read_file"}],
            [read_deploy],
            [{"name": "deploy", "citation": "call_1", "text": deploy}],
            f"deploy is {deploy}.",
            "answer",
        ),
        _task(
            "inc-revenue",
            "How many dollars of revenue were lost during the checkout incident?",
            [],
            [],
            [],
            "The incident files have no revenue figure.",
            "refuse",
        ),
    ]
    TASKS.write_text("".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")


def main() -> None:
    """Rewrite the incident corpus, its manifest, and the demo tasks."""
    write_corpus()
    os.environ["ORBIT_CORPUS"] = str(CORPUS)
    os.environ["ORBIT_CORPUS_MANIFEST"] = str(MANIFEST)
    from orbit_agents.corpus import write_manifest

    write_manifest()
    write_tasks()
    print(json.dumps({"corpus": str(CORPUS), "tasks": str(TASKS)}))


if __name__ == "__main__":
    main()
