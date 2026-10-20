"""Task catalog. Numeric gold is computed from the corpus, not typed by hand."""

from __future__ import annotations

REFUSAL = "The bundled corpus has no record that answers this."


def _count(table: str, where: dict | None = None, compare: dict | None = None) -> dict:
    spec = {"op": "count_rows", "table": table, "where": where or {}}
    if compare:
        spec["compare"] = compare
    return spec


def _sum(table: str, field: str, where: dict | None = None) -> dict:
    return {"op": "sum_field", "table": table, "field": field, "where": where or {}}


def _py(goal: str, spec: dict, claim: str | None = None) -> dict:
    step = {"kind": "python", "goal": goal, "spec": spec}
    if claim:
        step["claim"] = claim
    return step


def _retrieve(goal: str, query: str, file: str, text: str, claim: str) -> dict:
    return {
        "kind": "retrieve",
        "goal": goal,
        "query": query,
        "file": file,
        "text": text,
        "claim": claim,
    }


def _read(goal: str, path: str, text: str, claim: str) -> dict:
    return {"kind": "read_file", "goal": goal, "path": path, "text": text, "claim": claim}


def _web(goal: str, query: str, text: str, claim: str) -> dict:
    return {"kind": "web_search", "goal": goal, "query": query, "text": text, "claim": claim}


def _calc(goal: str, left: int, right: int, claim: str) -> dict:
    return {"kind": "calculator", "goal": goal, "left": left, "right": right, "claim": claim}


def _write(goal: str, source: int, template: str, path: str) -> dict:
    return {"kind": "write_file", "goal": goal, "source": source, "template": template, "path": path}


def _task(task_id: str, subset: str, question: str, steps: list | None = None, refusal: bool = False) -> dict:
    return {
        "id": task_id,
        "subset": subset,
        "question": question,
        "steps": steps or [],
        "refusal": refusal,
    }


CATALOG = [
    _task("dev-001", "train", "How many jobs in the jobs table used machine printer-a?", [
        _py("Count printer-a jobs", _count("jobs", {"machine": "printer-a"}), "job_count"),
    ]),
    _task("dev-002", "train", "How many laser jobs in the jobs table have outcome fail?", [
        _py("Count failed laser jobs", _count("jobs", {"machine": "laser", "outcome": "fail"}), "fail_count"),
    ]),
    _task("dev-003", "train", "What is the total grams recorded for machine printer-b in the jobs table?", [
        _py("Sum printer-b grams", _sum("jobs", "grams", {"machine": "printer-b"}), "grams"),
    ]),
    _task("dev-004", "train", "How many jobs in the jobs table used material pla?", [
        _py("Count PLA jobs", _count("jobs", {"material": "pla"}), "job_count"),
    ]),
    _task("dev-005", "train", "How many lines in logs/printer-a.log contain E_BED?", [
        _py("Count E_BED lines", {"op": "count_text", "file": "logs/printer-a.log", "needle": "E_BED"}, "line_count"),
    ]),
    _task("dev-006", "train", "Who covers the laser, according to the coverage note?", [
        _retrieve(
            "Retrieve the laser coverage sentence",
            "who covers the laser",
            "docs/shifts.md",
            "Jules covers the laser.",
            "cover",
        ),
    ]),
    _task("dev-007", "train", "How many shifts in the shifts table lasted more than 4 hours?", [
        _py(
            "Count long shifts",
            _count("shifts", compare={"field": "hours", "op": "gt", "value": 4}),
            "long_shifts",
        ),
    ]),
    _task(
        "dev-008",
        "train",
        "How many laser jobs ran longer than the second-person limit written in the laser note?",
        [
            _retrieve(
                "Retrieve the laser duration rule",
                "laser second person in the room",
                "docs/laser.md",
                "second person",
                "rule",
            ),
            _py(
                "Count laser jobs over that limit",
                _count("jobs", {"machine": "laser"}, {"field": "minutes", "op": "gt", "value": 15}),
                "long_jobs",
            ),
        ],
    ),
    _task("dev-009", "train", "For machine printer-a, how many jobs failed, and how many grams did those failed jobs list?", [
        _py("Count failed printer-a jobs", _count("jobs", {"machine": "printer-a", "outcome": "fail"}), "fail_count"),
        _py("Sum grams on those failed jobs", _sum("jobs", "grams", {"machine": "printer-a", "outcome": "fail"}), "fail_grams"),
    ]),
    _task(
        "dev-010",
        "train",
        "What does the offline page cache say M112 is, and how many FAIL lines are in logs/printer-a.log?",
        [
            _web("Read the offline M112 page", "what does gcode M112 mean", "emergency stop", "meaning"),
            _py("Count FAIL lines", {"op": "count_text", "file": "logs/printer-a.log", "needle": "FAIL"}, "fail_lines"),
        ],
    ),
    _task("dev-011", "train", "How many jobs used machine printer-c? Write that count into the workspace file summary.txt.", [
        _py("Count printer-c jobs", _count("jobs", {"machine": "printer-c"}), "job_count"),
        _write("Save the count in the workspace", 0, "printer-c jobs: {value}", "summary.txt"),
    ]),
    _task(
        "dev-012",
        "train",
        "For machine printer-a, what is the integer quotient of total minutes divided by the job count?",
        [
            _py("Sum printer-a minutes", _sum("jobs", "minutes", {"machine": "printer-a"})),
            _py("Count printer-a jobs", _count("jobs", {"machine": "printer-a"})),
            _calc("Divide total minutes by the job count", 0, 1, "minutes_per_job"),
        ],
    ),
    _task("dev-013", "train", "How many jobs used material petg and have outcome fail?", [
        _py("Count failed PETG jobs", _count("jobs", {"material": "petg", "outcome": "fail"}), "fail_count"),
    ]),
    _task("dev-014", "train", "What is the total minutes of laser jobs whose outcome is ok?", [
        _py("Sum minutes of successful laser jobs", _sum("jobs", "minutes", {"machine": "laser", "outcome": "ok"}), "minutes"),
    ]),
    _task("dev-015", "train", "How many jobs did machine printer-z run?", refusal=True),
    _task("dev-016", "train", "What is the door code for the makerspace?", refusal=True),
    _task("dev-017", "tune", "What is the total minutes recorded for machine printer-c?", [
        _py("Sum printer-c minutes", _sum("jobs", "minutes", {"machine": "printer-c"}), "minutes"),
    ]),
    _task("dev-018", "tune", "How many grams did the failed printer-c jobs list?", [
        _py("Sum grams of failed printer-c jobs", _sum("jobs", "grams", {"machine": "printer-c", "outcome": "fail"}), "grams"),
    ]),
    _task("dev-019", "tune", "How many shifts for person noor lasted more than 4 hours?", [
        _py(
            "Count Noor's long shifts",
            _count("shifts", {"person": "noor"}, {"field": "hours", "op": "gt", "value": 4}),
            "long_shifts",
        ),
    ]),
    _task("dev-020", "tune", "Open the coverage note and say who covers printer-a.", [
        _read("Read the coverage note", "docs/shifts.md", "Mina covers printer-a.", "cover"),
    ]),
    _task(
        "dev-021",
        "tune",
        "How many PLA jobs used more grams than the PLA gram cap written in the printers note?",
        [
            _retrieve("Retrieve the PLA gram cap", "PLA job grams cap", "docs/printers.md", "50 grams", "cap"),
            _py(
                "Count PLA jobs over that cap",
                _count("jobs", {"material": "pla"}, {"field": "grams", "op": "gt", "value": 50}),
                "over_cap",
            ),
        ],
    ),
    _task("dev-022", "tune", "How many laser jobs used material acrylic?", [
        _py("Count acrylic laser jobs", _count("jobs", {"machine": "laser", "material": "acrylic"}), "job_count"),
    ]),
    _task("dev-023", "tune", "How many jobs are already recorded for 2025-01-01?", refusal=True),
    _task("dev-024", "tune", "Which machine has the lowest rental price?", refusal=True),
    _task("test-001", "test", "How many printer-b jobs have outcome fail?", [
        _py("Count failed printer-b jobs", _count("jobs", {"machine": "printer-b", "outcome": "fail"}), "fail_count"),
    ]),
    _task("test-002", "test", "What is the total grams recorded for machine printer-a?", [
        _py("Sum printer-a grams", _sum("jobs", "grams", {"machine": "printer-a"}), "grams"),
    ]),
    _task("test-003", "test", "How many lines in logs/printer-a.log contain E_NOZZLE?", [
        _py("Count E_NOZZLE lines", {"op": "count_text", "file": "logs/printer-a.log", "needle": "E_NOZZLE"}, "line_count"),
    ]),
    _task("test-004", "test", "Who covers printer-c in the coverage note?", [
        _retrieve(
            "Retrieve the printer-c coverage sentence",
            "who covers printer-c",
            "docs/shifts.md",
            "Ari covers printer-c.",
            "cover",
        ),
    ]),
    _task("test-005", "test", "How many printer-c jobs lasted 90 minutes or longer?", [
        _py(
            "Count long printer-c jobs",
            _count("jobs", {"machine": "printer-c"}, {"field": "minutes", "op": "ge", "value": 90}),
            "long_jobs",
        ),
    ]),
    _task("test-006", "test", "Read the laser note and quote the phrase that requires a second person.", [
        _read("Read the laser note", "docs/laser.md", "second person", "rule"),
    ]),
    _task(
        "test-007",
        "test",
        "What does the offline page cache say G28 does, and how many laser jobs used material plywood?",
        [
            _web("Read the offline G28 page", "what does gcode G28 mean", "homes all axes", "meaning"),
            _py("Count plywood laser jobs", _count("jobs", {"machine": "laser", "material": "plywood"}), "job_count"),
        ],
    ),
    _task("test-008", "test", "How many printer-b jobs have outcome ok? Save that count in the workspace file ok.txt.", [
        _py("Count successful printer-b jobs", _count("jobs", {"machine": "printer-b", "outcome": "ok"}), "job_count"),
        _write("Save the count in the workspace", 0, "printer-b ok jobs: {value}", "ok.txt"),
    ]),
    _task("test-009", "test", "What is the total hours recorded for person ari in the shifts table?", [
        _py("Sum Ari's hours", _sum("shifts", "hours", {"person": "ari"}), "hours"),
    ]),
    _task("test-010", "test", "How many jobs in the jobs table are dated 2024-06-06?", [
        _py("Count jobs on that day", _count("jobs", {"day": "2024-06-06"}), "job_count"),
    ]),
    _task("test-011", "test", "How many jobs did the waterjet run?", refusal=True),
    _task("test-012", "test", "Which filament brand was delivered last Tuesday?", refusal=True),
]
