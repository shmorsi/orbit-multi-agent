"""Programmatic train and tune tasks. This module does not read eval/test.jsonl.

Gold values come from the same checkers the eval uses. A later leakage filter,
not this file, decides which questions are too close to the frozen test.
"""

from __future__ import annotations

import zlib
from pathlib import Path

from orbit_agents.eval.build_tasks import expand
from orbit_agents.eval.catalog import (
    _calc,
    _count,
    _py,
    _sum,
    _task,
    _write,
)

MACHINES = ("printer-a", "printer-b", "printer-c", "laser")
MATERIALS = ("pla", "petg", "plywood", "acrylic")
OUTCOMES = ("ok", "fail")
DAYS = (
    "2024-06-03",
    "2024-06-04",
    "2024-06-05",
    "2024-06-06",
    "2024-06-07",
    "2024-06-08",
    "2024-06-09",
    "2024-06-10",
    "2024-06-11",
    "2024-06-12",
)
PEOPLE = ("mina", "jules", "noor", "ari")
LOG_NEEDLES = ("START", "OK", "FAIL", "J01", "J02", "J06", "J10", "J14", "E_BED", "E_NOZZLE")
REFUSALS = (
    "What is the kiln firing schedule for the clay studio?",
    "Who paid the makerspace electricity bill in June?",
    "What is the volume of the resin vat?",
    "Which member owns the blue toolbox?",
    "How many visitors used the pen plotter?",
    "What is the combination for the filament cabinet?",
    "Which vendor services the air compressor?",
    "What is the laser tube hour count?",
    "How many spools of nylon are on the shelf?",
    "What is the membership price for a month?",
    "Who has the key to the spray booth?",
    "What is the guest wifi password this week?",
)

_CELL = {
    "machine": "machine cell",
    "material": "material cell",
    "outcome": "outcome cell",
    "day": "day cell",
    "person": "person cell",
}


def _phrase(where: dict, compare: dict | None = None) -> str:
    parts = [f"whose {_CELL[key]} is {value}" for key, value in where.items()]
    if compare:
        parts.append(f"whose {compare['field']} cell is greater than {int(compare['value'])}")
    if not parts:
        raise ValueError("a variant needs a filter")
    return " and ".join(parts)


def _goal(text: str) -> str:
    return text if len(text) <= 140 else text[:137] + "..."


def _spec(family: str, question: str, steps: list, refusal: bool = False) -> dict:
    row = _task("pending", "train", question, steps, refusal=refusal)
    row["family"] = family
    return row


def _count_jobs(question: str, where: dict, compare: dict | None, family: str) -> dict:
    return _spec(family, question, [_py(_goal(question), _count("jobs", where, compare), "job_count")])


def _sum_jobs(question: str, field: str, where: dict, family: str) -> dict:
    claim = "grams" if field == "grams" else "minutes"
    return _spec(family, question, [_py(_goal(question), _sum("jobs", field, where), claim)])


def iter_specs() -> list[dict]:
    """Catalog-style rows. Ids and subsets are assigned in materialize_variants."""
    rows: list[dict] = []
    singles = (
        ("machine", MACHINES),
        ("material", MATERIALS),
        ("outcome", OUTCOMES),
        ("day", DAYS),
    )
    for key, values in singles:
        for value in values:
            where = {key: value}
            phrase = _phrase(where)
            rows.append(_count_jobs(f"From the jobs table, count rows {phrase}.", where, None, "count_jobs"))
            if key != "outcome":
                rows.append(
                    _sum_jobs(
                        f"From the jobs table, add the minutes cell for every row {phrase}.",
                        "minutes",
                        where,
                        "sum_jobs",
                    )
                )
                rows.append(
                    _sum_jobs(
                        f"From the jobs table, add the grams cell for every row {phrase}.",
                        "grams",
                        where,
                        "sum_jobs",
                    )
                )
    for machine in MACHINES:
        for material in MATERIALS:
            where = {"machine": machine, "material": material}
            phrase = _phrase(where)
            rows.append(_count_jobs(f"From the jobs table, count rows {phrase}.", where, None, "count_jobs"))
            rows.append(
                _sum_jobs(
                    f"From the jobs table, add the grams cell for every row {phrase}.",
                    "grams",
                    where,
                    "sum_jobs",
                )
            )
        for outcome in OUTCOMES:
            where = {"machine": machine, "outcome": outcome}
            phrase = _phrase(where)
            rows.append(_count_jobs(f"From the jobs table, count rows {phrase}.", where, None, "count_jobs"))
            rows.append(
                _sum_jobs(
                    f"From the jobs table, add the minutes cell for every row {phrase}.",
                    "minutes",
                    where,
                    "sum_jobs",
                )
            )
    for material in MATERIALS:
        for outcome in OUTCOMES:
            where = {"material": material, "outcome": outcome}
            phrase = _phrase(where)
            rows.append(_count_jobs(f"From the jobs table, count rows {phrase}.", where, None, "count_jobs"))
            rows.append(
                _sum_jobs(
                    f"From the jobs table, add the minutes cell for every row {phrase}.",
                    "minutes",
                    where,
                    "sum_jobs",
                )
            )
    for machine in MACHINES:
        for material in MATERIALS:
            for outcome in OUTCOMES:
                where = {"machine": machine, "material": material, "outcome": outcome}
                rows.append(
                    _count_jobs(
                        f"From the jobs table, count rows {_phrase(where)}.",
                        where,
                        None,
                        "count_jobs",
                    )
                )
    for day in DAYS:
        for outcome in OUTCOMES:
            where = {"day": day, "outcome": outcome}
            rows.append(
                _count_jobs(f"From the jobs table, count rows {_phrase(where)}.", where, None, "count_jobs")
            )
    for limit in (15, 30, 45, 60, 90):
        compare = {"field": "minutes", "op": "gt", "value": limit}
        rows.append(
            _count_jobs(
                f"From the jobs table, count rows {_phrase({}, compare)}.",
                {},
                compare,
                "threshold_jobs",
            )
        )
        for machine in MACHINES:
            where = {"machine": machine}
            rows.append(
                _count_jobs(
                    f"From the jobs table, count rows {_phrase(where, compare)}.",
                    where,
                    compare,
                    "threshold_jobs",
                )
            )
    for person in PEOPLE:
        where = {"person": person}
        phrase = _phrase(where)
        rows.append(
            _spec(
                "count_shifts",
                f"From the shifts table, count rows {phrase}.",
                [_py(_goal(phrase), _count("shifts", where), "shift_count")],
            )
        )
        rows.append(
            _spec(
                "sum_shifts",
                f"From the shifts table, add the hours cell for every row {phrase}.",
                [_py(_goal(phrase), _sum("shifts", "hours", where), "hours")],
            )
        )
        for limit in (3, 5, 6, 7):
            compare = {"field": "hours", "op": "gt", "value": limit}
            rows.append(
                _spec(
                    "threshold_shifts",
                    f"From the shifts table, count rows {_phrase(where, compare)}.",
                    [_py(_goal(phrase), _count("shifts", where, compare), "long_shifts")],
                )
            )
    for limit in (3, 5, 6, 7):
        compare = {"field": "hours", "op": "gt", "value": limit}
        rows.append(
            _spec(
                "threshold_shifts",
                f"From the shifts table, count rows {_phrase({}, compare)}.",
                [_py("Count long shifts", _count("shifts", {}, compare), "long_shifts")],
            )
        )
    for needle in LOG_NEEDLES:
        rows.append(
            _spec(
                "log",
                f"In logs/printer-a.log, how many lines contain the token {needle}?",
                [
                    _py(
                        f"Count {needle} lines",
                        {"op": "count_text", "file": "logs/printer-a.log", "needle": needle},
                        "line_count",
                    )
                ],
            )
        )
    for machine in MACHINES:
        rows.append(
            _spec(
                "quotient",
                (
                    f"Total the minutes of {machine} jobs, count those jobs, then give the integer "
                    "quotient of the total divided by the count."
                ),
                [
                    _py(f"Sum {machine} minutes", _sum("jobs", "minutes", {"machine": machine})),
                    _py(f"Count {machine} jobs", _count("jobs", {"machine": machine})),
                    _calc(f"Divide {machine} minutes by the job count", 0, 1, "minutes_per_job"),
                ],
            )
        )
        slug = machine.replace("-", "")
        rows.append(
            _spec(
                "write",
                f"Count jobs whose machine is {machine}, then write that count into workspace file {slug}.txt.",
                [
                    _py(f"Count {machine} jobs", _count("jobs", {"machine": machine}), "job_count"),
                    _write(f"Save the {machine} count", 0, f"{machine} jobs: {{value}}", f"{slug}.txt"),
                ],
            )
        )
    for question in REFUSALS:
        rows.append(_spec("refusal", question, [], refusal=True))
    return rows


def _subset(question: str) -> str:
    return "tune" if zlib.adler32(question.encode("utf-8")) % 8 == 0 else "train"


def materialize_variants(workspace: Path) -> list[dict]:
    """Expand every programmatic variant. Questions already in `seen` are skipped by the caller."""
    grouped = {"train": [], "tune": []}
    for spec in iter_specs():
        grouped[_subset(spec["question"])].append(spec)
    rows = []
    for subset, specs in grouped.items():
        width = max(3, len(str(len(specs))))
        for index, spec in enumerate(specs, start=1):
            task_id = f"var-{subset}-{index:0{width}d}"
            spec = dict(spec)
            spec["id"] = task_id
            spec["subset"] = subset
            record = expand(spec, workspace / task_id)
            record["family"] = spec["family"]
            rows.append(record)
    return rows
