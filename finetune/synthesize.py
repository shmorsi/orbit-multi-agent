"""Corpus scenarios for fine-tuning. This module does not read eval/test.jsonl."""

from __future__ import annotations

from orbit_agents.eval.build_tasks import expand
from orbit_agents.eval.catalog import _calc, _count, _py, _read, _retrieve, _sum, _task, _web, _write

TRAIN = (
    ("gen-train-001", "From the jobs table, count rows whose material cell is petg and whose outcome cell is ok.", [
        _py("Count successful PETG jobs", _count("jobs", {"material": "petg", "outcome": "ok"}), "petg_ok"),
    ]),
    ("gen-train-002", "From the jobs table, add the minutes cell for every row whose material cell is pla.", [
        _py("Sum PLA minutes", _sum("jobs", "minutes", {"material": "pla"}), "pla_minutes"),
    ]),
    ("gen-train-003", "From the jobs table, add the grams cell for every row whose machine cell is laser.", [
        _py("Sum laser grams", _sum("jobs", "grams", {"machine": "laser"}), "laser_grams"),
    ]),
    ("gen-train-004", "From the jobs table, count rows whose day cell is 2024-06-03.", [
        _py("Count jobs on 3 June", _count("jobs", {"day": "2024-06-03"}), "day_count"),
    ]),
    ("gen-train-005", "From the jobs table, count rows whose day cell is 2024-06-04.", [
        _py("Count jobs on 4 June", _count("jobs", {"day": "2024-06-04"}), "day_count"),
    ]),
    ("gen-train-006", "From the jobs table, count rows whose day cell is 2024-06-05.", [
        _py("Count jobs on 5 June", _count("jobs", {"day": "2024-06-05"}), "day_count"),
    ]),
    ("gen-train-007", "From the jobs table, count rows whose day cell is 2024-06-07.", [
        _py("Count jobs on 7 June", _count("jobs", {"day": "2024-06-07"}), "day_count"),
    ]),
    ("gen-train-008", "From the jobs table, count rows whose day cell is 2024-06-08.", [
        _py("Count jobs on 8 June", _count("jobs", {"day": "2024-06-08"}), "day_count"),
    ]),
    ("gen-train-009", "From the jobs table, count rows whose day cell is 2024-06-09.", [
        _py("Count jobs on 9 June", _count("jobs", {"day": "2024-06-09"}), "day_count"),
    ]),
    ("gen-train-010", "From the jobs table, count rows whose day cell is 2024-06-10.", [
        _py("Count jobs on 10 June", _count("jobs", {"day": "2024-06-10"}), "day_count"),
    ]),
    ("gen-train-011", "From the jobs table, count rows whose day cell is 2024-06-11.", [
        _py("Count jobs on 11 June", _count("jobs", {"day": "2024-06-11"}), "day_count"),
    ]),
    ("gen-train-012", "From the jobs table, count rows whose day cell is 2024-06-12.", [
        _py("Count jobs on 12 June", _count("jobs", {"day": "2024-06-12"}), "day_count"),
    ]),
    ("gen-train-013", "From the shifts table, add the hours cell for every row whose person cell is mina.", [
        _py("Sum Mina hours", _sum("shifts", "hours", {"person": "mina"}), "mina_hours"),
    ]),
    ("gen-train-014", "From the shifts table, add the hours cell for every row whose person cell is jules.", [
        _py("Sum Jules hours", _sum("shifts", "hours", {"person": "jules"}), "jules_hours"),
    ]),
    ("gen-train-015", "From the shifts table, add the hours cell for every row whose person cell is noor.", [
        _py("Sum Noor hours", _sum("shifts", "hours", {"person": "noor"}), "noor_hours"),
    ]),
    ("gen-train-016", "From the shifts table, count rows whose hours cell is greater than 6.", [
        _py("Count shifts over 6 hours", _count("shifts", compare={"field": "hours", "op": "gt", "value": 6}), "long_shifts"),
    ]),
    ("gen-train-017", "From the jobs table, count rows whose machine cell is printer-c and whose material cell is petg.", [
        _py("Count printer-c PETG jobs", _count("jobs", {"machine": "printer-c", "material": "petg"}), "job_count"),
    ]),
    ("gen-train-018", "From the jobs table, count rows whose machine cell is printer-a and whose material cell is petg.", [
        _py("Count printer-a PETG jobs", _count("jobs", {"machine": "printer-a", "material": "petg"}), "job_count"),
    ]),
    ("gen-train-019", "From the jobs table, add the grams cell where material is petg and outcome is ok.", [
        _py("Sum grams of successful PETG", _sum("jobs", "grams", {"material": "petg", "outcome": "ok"}), "petg_grams"),
    ]),
    ("gen-train-020", "From the jobs table, add the minutes cell for every row whose material cell is acrylic.", [
        _py("Sum acrylic minutes", _sum("jobs", "minutes", {"material": "acrylic"}), "acrylic_minutes"),
    ]),
    ("gen-train-021", "From the jobs table, count rows whose material cell is plywood and whose outcome cell is fail.", [
        _py("Count failed plywood jobs", _count("jobs", {"material": "plywood", "outcome": "fail"}), "fail_count"),
    ]),
    ("gen-train-022", "In logs/printer-a.log, how many lines contain the token START?", [
        _py("Count START lines", {"op": "count_text", "file": "logs/printer-a.log", "needle": "START"}, "line_count"),
    ]),
    ("gen-train-023", "In logs/printer-a.log, how many lines contain the token OK?", [
        _py("Count OK lines", {"op": "count_text", "file": "logs/printer-a.log", "needle": "OK"}, "line_count"),
    ]),
    ("gen-train-024", "Open docs/printers.md and copy the sentence that limits a single printer-a job to 90 minutes.", [
        _read(
            "Read the printer-a duration rule",
            "docs/printers.md",
            "Printer-a must not run a single job longer than 90 minutes.",
            "duration_rule",
        ),
    ]),
    ("gen-train-025", "Open docs/shifts.md and copy the sentence that defines a long shift.", [
        _read(
            "Read the long-shift definition",
            "docs/shifts.md",
            "A shift longer than 4 hours is a long shift.",
            "shift_rule",
        ),
    ]),
    ("gen-train-026", "Retrieve the printers note sentence that says failed jobs still count the listed grams.", [
        _retrieve(
            "Retrieve the failed-grams sentence",
            "failed jobs still count grams",
            "docs/printers.md",
            "Failed jobs still count the grams listed on the job row.",
            "grams_rule",
        ),
    ]),
    ("gen-train-027", "Open docs/laser.md and copy the sentence that names both plywood and acrylic.", [
        _read(
            "Read the laser materials sentence",
            "docs/laser.md",
            "The laser cuts plywood and acrylic.",
            "materials",
        ),
    ]),
    ("gen-train-028", "Total the minutes of printer-b jobs, count those jobs, then give the integer quotient of the total divided by the count.", [
        _py("Sum printer-b minutes", _sum("jobs", "minutes", {"machine": "printer-b"})),
        _py("Count printer-b jobs", _count("jobs", {"machine": "printer-b"})),
        _calc("Divide printer-b minutes by the job count", 0, 1, "minutes_per_job"),
    ]),
    ("gen-train-029", "Count jobs whose material is pla and whose outcome is ok, then write that count into workspace file pla-ok.txt.", [
        _py("Count successful PLA jobs", _count("jobs", {"material": "pla", "outcome": "ok"}), "pla_ok"),
        _write("Save the PLA ok count", 0, "pla ok jobs: {value}", "pla-ok.txt"),
    ]),
    ("gen-train-030", "From the jobs table, count rows whose machine cell is printer-c and whose material cell is pla.", [
        _py("Count printer-c PLA jobs", _count("jobs", {"machine": "printer-c", "material": "pla"}), "job_count"),
    ]),
    ("gen-train-031", "From the jobs table, add the minutes cell where machine is laser and outcome is ok.", [
        _py("Sum successful laser minutes", _sum("jobs", "minutes", {"machine": "laser", "outcome": "ok"}), "laser_minutes"),
    ]),
    ("gen-train-032", "From the shifts table, count rows whose person cell is mina.", [
        _py("Count Mina shifts", _count("shifts", {"person": "mina"}), "shift_count"),
    ]),
    ("gen-train-033", "From the jobs table, add the grams cell for every row whose outcome cell is fail.", [
        _py("Sum grams on failed jobs", _sum("jobs", "grams", {"outcome": "fail"}), "fail_grams"),
    ]),
    ("gen-train-034", "From the jobs table, add the minutes cell where material is pla and outcome is ok.", [
        _py("Sum minutes of successful PLA", _sum("jobs", "minutes", {"material": "pla", "outcome": "ok"}), "pla_minutes"),
    ]),
    ("gen-train-035", "Look up gcode M112 in the offline page cache and quote the snippet.", [
        _web("Read the offline M112 page", "what does gcode M112 mean", "M112 is the emergency stop command.", "meaning"),
    ]),
    ("gen-train-036", "In logs/printer-a.log, how many lines contain the token J14?", [
        _py("Count J14 lines", {"op": "count_text", "file": "logs/printer-a.log", "needle": "J14"}, "line_count"),
    ]),
    ("gen-train-037", "From the jobs table, add the grams cell for every row whose material cell is pla.", [
        _py("Sum PLA grams", _sum("jobs", "grams", {"material": "pla"}), "pla_grams"),
    ]),
    ("gen-train-038", "From the shifts table, count rows whose hours cell is greater than 5.", [
        _py("Count shifts over 5 hours", _count("shifts", compare={"field": "hours", "op": "gt", "value": 5}), "long_shifts"),
    ]),
    ("gen-train-039", "From the jobs table, count rows whose machine cell is printer-b and whose material cell is pla.", [
        _py("Count printer-b PLA jobs", _count("jobs", {"machine": "printer-b", "material": "pla"}), "job_count"),
    ]),
    ("gen-train-040", "From the jobs table, add the minutes cell for every row whose day cell is 2024-06-03.", [
        _py("Sum minutes on 3 June", _sum("jobs", "minutes", {"day": "2024-06-03"}), "day_minutes"),
    ]),
    ("gen-train-041", "Read the printers note for the 90 minute rule, then count printer-a jobs whose minutes cell is greater than 90.", [
        _retrieve(
            "Retrieve the printer-a duration rule",
            "printer-a single job 90 minutes",
            "docs/printers.md",
            "90 minutes",
            "limit",
        ),
        _py(
            "Count printer-a jobs over 90 minutes",
            _count("jobs", {"machine": "printer-a"}, {"field": "minutes", "op": "gt", "value": 90}),
            "over_limit",
        ),
    ]),
    ("gen-train-042", "Add Noor's hours, count Noor's shift rows, then give the integer quotient of the hours divided by the row count.", [
        _py("Sum Noor hours", _sum("shifts", "hours", {"person": "noor"})),
        _py("Count Noor shifts", _count("shifts", {"person": "noor"})),
        _calc("Divide Noor hours by the shift count", 0, 1, "hours_per_shift"),
    ]),
    ("gen-train-043", "From the jobs table, count rows whose machine cell is laser and whose material cell is plywood and whose outcome cell is ok.", [
        _py(
            "Count successful plywood laser jobs",
            _count("jobs", {"machine": "laser", "material": "plywood", "outcome": "ok"}),
            "job_count",
        ),
    ]),
    ("gen-train-044", "From the jobs table, add the grams cell where machine is printer-c and material is pla.", [
        _py("Sum printer-c PLA grams", _sum("jobs", "grams", {"machine": "printer-c", "material": "pla"}), "pla_grams"),
    ]),
    ("gen-train-045", "From the shifts table, count rows whose person cell is jules.", [
        _py("Count Jules shifts", _count("shifts", {"person": "jules"}), "shift_count"),
    ]),
    ("gen-train-046", "In logs/printer-a.log, how many lines contain the token FAIL?", [
        _py("Count FAIL lines", {"op": "count_text", "file": "logs/printer-a.log", "needle": "FAIL"}, "line_count"),
    ]),
    ("gen-train-047", "From the jobs table, count rows whose outcome cell is ok and whose material cell is petg.", [
        _py("Count ok PETG jobs again by outcome", _count("jobs", {"outcome": "ok", "material": "petg"}), "petg_ok"),
    ]),
    ("gen-train-048", "Open docs/shifts.md and copy the sentence that says hours are time in the room.", [
        _read("Read the hours definition", "docs/shifts.md", "Hours are time in the room.", "hours_rule"),
    ]),
)

TRAIN_REFUSALS = (
    ("gen-train-049", "What is the makerspace wifi password?"),
    ("gen-train-050", "How many jobs did the CNC router run?"),
    ("gen-train-051", "Which resin brand arrived on 2024-07-01?"),
    ("gen-train-052", "What is the alarm code for the laser room?"),
    ("gen-train-053", "How many visitors signed the June guest book?"),
    ("gen-train-054", "What is the purchase price of printer-a?"),
)

TUNE = (
    ("gen-tune-001", "From the jobs table, add the grams cell for every row whose day cell is 2024-06-04.", [
        _py("Sum grams on 4 June", _sum("jobs", "grams", {"day": "2024-06-04"}), "day_grams"),
    ]),
    ("gen-tune-002", "From the shifts table, count rows whose person cell is noor.", [
        _py("Count Noor shift rows", _count("shifts", {"person": "noor"}), "shift_count"),
    ]),
    ("gen-tune-003", "In logs/printer-a.log, how many lines contain the token J10?", [
        _py("Count J10 lines", {"op": "count_text", "file": "logs/printer-a.log", "needle": "J10"}, "line_count"),
    ]),
    ("gen-tune-004", "From the jobs table, add the minutes cell where machine is printer-b and outcome is ok.", [
        _py("Sum successful printer-b minutes", _sum("jobs", "minutes", {"machine": "printer-b", "outcome": "ok"}), "ok_minutes"),
    ]),
    ("gen-tune-005", "From the jobs table, count rows whose material cell is pla and whose outcome cell is fail.", [
        _py("Count failed PLA jobs", _count("jobs", {"material": "pla", "outcome": "fail"}), "fail_count"),
    ]),
    ("gen-tune-006", "Open docs/printers.md and copy the sentence that sets the PLA gram cap.", [
        _read("Read the PLA gram sentence", "docs/printers.md", "A PLA job should use at most 50 grams.", "gram_cap"),
    ]),
    ("gen-tune-007", "From the jobs table, add the minutes cell where machine is printer-c and outcome is fail.", [
        _py("Sum failed printer-c minutes", _sum("jobs", "minutes", {"machine": "printer-c", "outcome": "fail"}), "fail_minutes"),
    ]),
    ("gen-tune-008", "Add the grams of printer-b jobs whose material is petg, then write that sum into workspace file petg-grams.txt.", [
        _py("Sum printer-b PETG grams", _sum("jobs", "grams", {"machine": "printer-b", "material": "petg"}), "petg_grams"),
        _write("Save the PETG gram sum", 0, "printer-b petg grams: {value}", "petg-grams.txt"),
    ]),
)

TUNE_REFUSALS = (
    ("gen-tune-009", "What was the visitor count on 2024-06-12?"),
    ("gen-tune-010", "What is the serial number of printer-c?"),
    ("gen-tune-011", "Which courier delivered the plywood?"),
)


def scenarios(subset: str) -> list[dict]:
    rows = []
    source = TRAIN if subset == "train" else TUNE
    refusals = TRAIN_REFUSALS if subset == "train" else TUNE_REFUSALS
    for task_id, question, steps in source:
        rows.append(_task(task_id, subset, question, steps))
    for task_id, question in refusals:
        rows.append(_task(task_id, subset, question, refusal=True))
    return rows


def materialize(subset: str, workspace) -> list[dict]:
    return [expand(_task_row, workspace / _task_row["id"]) for _task_row in scenarios(subset)]
