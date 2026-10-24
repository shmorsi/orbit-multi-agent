"""Fictional checkout-outage demo. Not a scored eval split.

Answers are read back out of the incident files. The scripted harness follows
tasks.jsonl. Measured makerspace scores are a different corpus.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from orbit_agents.paths import ROOT
from orbit_agents.tasks import read_jsonl

SCENARIO = ROOT / "data" / "scenarios" / "incident"
CORPUS = SCENARIO / "corpus"
MANIFEST = SCENARIO / "manifest.json"
TASKS = SCENARIO / "tasks.jsonl"
# A count the incident files do not contain. The red team cites it as an inventory root cause.
INVENTORY_BLAME = 17


def incident_tasks() -> list[dict]:
    """Demo tasks for the incident scenario. These are not dev, tune, or test."""
    return read_jsonl(TASKS)


def task_by_id(task_id: str) -> dict:
    """One incident task."""
    for task in incident_tasks():
        if task["id"] == task_id:
            return task
    raise KeyError(task_id)


@contextmanager
def use_incident_corpus():
    """Point the sandboxed tools at the incident files, then restore the makerspace."""
    previous_corpus = os.environ.get("ORBIT_CORPUS")
    previous_manifest = os.environ.get("ORBIT_CORPUS_MANIFEST")
    os.environ["ORBIT_CORPUS"] = str(CORPUS)
    os.environ["ORBIT_CORPUS_MANIFEST"] = str(MANIFEST)
    try:
        yield
    finally:
        if previous_corpus is None:
            os.environ.pop("ORBIT_CORPUS", None)
        else:
            os.environ["ORBIT_CORPUS"] = previous_corpus
        if previous_manifest is None:
            os.environ.pop("ORBIT_CORPUS_MANIFEST", None)
        else:
            os.environ["ORBIT_CORPUS_MANIFEST"] = previous_manifest


def _read(relative: str) -> str:
    return (CORPUS / relative).read_text(encoding="utf-8")


def page_threshold() -> int:
    """The runbook's error-rate percent. The page fires when a minute is strictly above it."""
    for line in _read("docs/runbook.md").splitlines():
        if "above" in line and "percent" in line:
            return int(line.split("above", 1)[1].split("percent", 1)[0].strip())
    raise RuntimeError("runbook has no page threshold")


def page_minutes() -> int:
    """How many consecutive minutes the runbook requires before a page."""
    for line in _read("docs/runbook.md").splitlines():
        if "consecutive minutes" in line:
            return int(line.split("for", 1)[1].split("consecutive", 1)[0].strip())
    raise RuntimeError("runbook has no minute count")


def payment_checkout_errors() -> int:
    """Checkout log lines whose upstream is payment-service."""
    return sum(1 for line in _read("logs/checkout.log").splitlines() if "PAYMENT_UPSTREAM" in line)


def failed_orders() -> int:
    """Checkout log lines that record a failed order."""
    return sum(1 for line in _read("logs/checkout.log").splitlines() if "ORDER_FAILED" in line)


def breach_minutes() -> int:
    """Minutes whose payment error rate is above the runbook percent."""
    threshold = page_threshold()
    count = 0
    for line in _read("metrics.csv").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 4 or parts[1] != "payment":
            continue
        errors = int(parts[3])
        requests = int(parts[2])
        if requests > 0 and errors * 100 // requests > threshold:
            count += 1
    return count


def spike_began() -> str:
    """Timestamp the status note gives for the start of the spike."""
    for line in _read("docs/status.md").splitlines():
        if "spike began at" in line:
            return line.split("spike began at", 1)[1].strip().rstrip(".")
    raise RuntimeError("status note has no spike time")


def preceding_deploy() -> str:
    """Payment-service deploy id with the latest timestamp strictly before the spike."""
    spike = spike_began()
    chosen = ""
    chosen_at = ""
    for line in _read("logs/deploys.log").splitlines():
        if "payment-service deploy " not in line:
            continue
        stamp, _rest = line.split(" ", 1)
        deploy = line.rsplit(" ", 1)[1]
        if stamp < spike and stamp > chosen_at:
            chosen_at = stamp
            chosen = deploy
    if not chosen:
        raise RuntimeError("no payment deploy precedes the spike")
    return chosen


def inventory_errors() -> int:
    """Inventory log lines that are errors. The red-team root cause needs this to be zero."""
    return sum(1 for line in _read("logs/inventory.log").splitlines() if "ERROR" in line)
