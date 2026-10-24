"""Independent checkers and the python the scripted worker runs. Both must agree."""

from __future__ import annotations

from orbit_agents.corpus import load_table, read_text

_CMP = {"gt": ">", "ge": ">=", "eq": "=="}


def _row_ok(row: dict, spec: dict) -> bool:
    for key, value in spec.get("where", {}).items():
        if row[key] != value:
            return False
    compare = spec.get("compare")
    if not compare:
        return True
    got = int(row[compare["field"]])
    limit = int(compare["value"])
    op = compare["op"]
    if op == "gt":
        return got > limit
    if op == "ge":
        return got >= limit
    if op == "eq":
        return got == limit
    raise ValueError(op)


def evaluate(spec: dict) -> int | str:
    """Run one checker op over the corpus. The scripted python cell must print the same value."""
    op = spec["op"]
    if op == "count_rows":
        return sum(1 for row in load_table(spec["table"]) if _row_ok(row, spec))
    if op == "sum_field":
        return sum(int(row[spec["field"]]) for row in load_table(spec["table"]) if _row_ok(row, spec))
    if op == "count_text":
        body = read_text(spec["file"])
        return sum(1 for line in body.splitlines() if spec["needle"] in line)
    if op == "doc_contains":
        if spec["text"] not in read_text(spec["file"]):
            raise RuntimeError(f"{spec['file']} does not contain the expected note")
        return spec["text"]
    raise ValueError(op)


def render_python(spec: dict) -> str:
    """Render the python cell the scripted worker runs for this checker spec."""
    op = spec["op"]
    if op == "count_text":
        return (
            f"body = text({spec['file']!r})\n"
            f"print(sum(1 for line in body.splitlines() if {spec['needle']!r} in line))"
        )
    clauses = []
    for key, value in spec.get("where", {}).items():
        clauses.append(f"row[{key!r}] == {value!r}")
    compare = spec.get("compare")
    if compare:
        clauses.append(
            f"int(row[{compare['field']!r}]) {_CMP[compare['op']]} {int(compare['value'])}"
        )
    condition = " and ".join(clauses) if clauses else "True"
    table = spec["table"]
    if op == "count_rows":
        return f"rows = table({table!r})\nprint(sum(1 for row in rows if {condition}))"
    if op == "sum_field":
        field = spec["field"]
        return (
            f"rows = table({table!r})\n"
            f"print(sum(int(row[{field!r}]) for row in rows if {condition}))"
        )
    raise ValueError(op)
