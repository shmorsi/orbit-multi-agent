"""Child process for the Python tool. The parent kills it after one second."""

from __future__ import annotations

import io
import json
import sys

from orbit_agents.corpus import load_table, read_text


def main() -> None:
    code = sys.stdin.read()
    buffer = io.StringIO()

    def _print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        buffer.write(sep.join(str(arg) for arg in args))
        buffer.write(end)

    safe_globals = {
        "__builtins__": {},
        "table": load_table,
        "text": read_text,
        "print": _print,
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "int": int,
        "float": float,
        "str": str,
        "range": range,
        "enumerate": enumerate,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "zip": zip,
        "abs": abs,
        "any": any,
        "all": all,
        "round": round,
    }
    try:
        exec(compile(code, "<sandbox>", "exec"), safe_globals, {})  # noqa: S102
    except Exception as exc:  # faults stay inside the tool result
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        payload = {"ok": True, "stdout": buffer.getvalue()}
    sys.stdout.write(json.dumps(payload))


if __name__ == "__main__":
    main()
