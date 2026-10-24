"""Sandboxed tools. Python and search never touch the network or the repo outside the corpus."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from orbit_agents.corpus import TABLES, active_corpus, read_text
from orbit_agents.schema import TOOL_SCHEMAS, validate

_ALLOWED_NODES = {
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.While,
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    ast.ListComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Call,
    ast.keyword,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.IfExp,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Attribute,
}
_SAFE_ATTRS = {"splitlines", "split", "strip", "lower", "upper", "startswith", "endswith", "items", "keys", "get"}
_ALLOWED_CALLS = {
    "table",
    "text",
    "print",
    "len",
    "sum",
    "min",
    "max",
    "int",
    "float",
    "str",
    "range",
    "enumerate",
    "sorted",
    "list",
    "dict",
    "set",
    "tuple",
    "zip",
    "abs",
    "any",
    "all",
    "round",
}
_CALC_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.Load,
}


def _reject_python(code: str) -> str | None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            return f"blocked syntax: {type(node).__name__}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in _SAFE_ATTRS:
                return "blocked attribute"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _ALLOWED_CALLS:
                pass
            elif isinstance(func, ast.Attribute) and func.attr in _SAFE_ATTRS and not func.attr.startswith("_"):
                pass
            else:
                return "blocked call"
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            return "blocked name"
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, str, bool, type(None))):
            return "blocked constant"
    return None


def run_python(code: str) -> dict:
    """Run checked code in a child process so the one-second limit works on Windows and POSIX."""
    error = _reject_python(code)
    if error:
        return {"ok": False, "error": error}
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "orbit_agents.sandbox_exec"],
            input=code,
            text=True,
            capture_output=True,
            timeout=1,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "TimeoutError: python step exceeded one second"}
    raw = completed.stdout.strip()
    if not raw:
        detail = completed.stderr.strip() or "python step returned no result"
        return {"ok": False, "error": detail}
    try:
        payload = json.loads(raw.splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "error": "python step returned no result"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "python step returned no result"}
    return payload


def _eval_calc(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_calc(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_calc(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_calc(node.left)
        right = _eval_calc(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left // right
    raise ValueError("expression must use integers and + - * //")


def run_calculator(expression: str) -> dict:
    """Evaluate one integer expression. True division is rejected."""
    if not expression or any(ch not in "0123456789+-*/() " for ch in expression):
        return {"ok": False, "error": "expression may use only integers, spaces, and + - * // ()"}
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return {"ok": False, "error": f"syntax error: {exc.msg}"}
    for node in ast.walk(tree):
        if type(node) not in _CALC_NODES:
            return {"ok": False, "error": f"blocked syntax: {type(node).__name__}"}
    try:
        result = _eval_calc(tree.body)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "result": result}


def run_read_file(path: str) -> dict:
    """Read one corpus file by relative path."""
    try:
        body = read_text(path)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "path": path, "text": body[:2000]}


def _workspace_path(workspace: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("write path must stay inside the run workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    path = (workspace / relative).resolve()
    if workspace.resolve() not in path.parents and path != workspace.resolve():
        raise ValueError("write path must stay inside the run workspace")
    return path


def run_write_file(workspace: Path, path: str, content: str) -> dict:
    """Write one file under this run's workspace."""
    try:
        target = _workspace_path(workspace, path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}


def _tokens(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch == "-" else " " for ch in text)
    return {token for token in cleaned.split() if len(token) >= 3}


def _search_cards() -> list[tuple[str, str, int]]:
    """Notes first, then a card per table and log. Equal overlap keeps the note."""
    root = active_corpus()
    cards: list[tuple[str, str, int]] = []
    docs = root / "docs"
    if docs.is_dir():
        for path in sorted(docs.glob("*.md")):
            relative = path.relative_to(root).as_posix()
            for block in path.read_text(encoding="utf-8").split("\n\n"):
                text = block.strip()
                if text and not text.startswith("#"):
                    cards.append((relative, text, 0))
    for name, relative in sorted(TABLES.items()):
        table_path = root / relative
        if not table_path.is_file():
            continue
        header = table_path.read_text(encoding="utf-8").splitlines()[0].strip()
        text = (
            f"Table {name} is file {relative} with columns {header}. "
            f"Read it in python with table('{name}')."
        )
        cards.append((relative, text, 1))
    logs = root / "logs"
    if logs.is_dir():
        for path in sorted(logs.glob("*.log")):
            relative = path.relative_to(root).as_posix()
            text = f"Log file {relative}. Read it in python with text('{relative}') or read_file."
            cards.append((relative, text, 1))
    return cards


def run_retrieve(query: str) -> dict:
    """Keyword search over the bundled notes, tables, and logs."""
    query_tokens = _tokens(query)
    ranked = sorted(
        _search_cards(),
        key=lambda item: (-len(query_tokens & _tokens(item[1])), item[2], item[0], item[1]),
    )
    if not ranked or not (query_tokens & _tokens(ranked[0][1])):
        return {"ok": True, "query": query, "matches": []}
    source, text, _group = ranked[0]
    return {"ok": True, "query": query, "matches": [{"path": source, "text": text}]}


def run_web_search(query: str) -> dict:
    """Offline cache only. This function does not import a network client."""
    cache = active_corpus() / "web_cache.json"
    if not cache.is_file():
        return {"ok": False, "error": "no offline web cache in this corpus"}
    payload = json.loads(cache.read_text(encoding="utf-8"))
    query_tokens = _tokens(query)
    hits = []
    for page in payload["pages"]:
        page_tokens = _tokens(page["query"] + " " + page["snippet"])
        if query.strip().lower() == page["query"].lower() or len(query_tokens & page_tokens) >= 2:
            hits.append({"title": page["title"], "snippet": page["snippet"]})
    return {"ok": True, "query": query, "results": hits[:2], "offline": True}


def call_tool(name: str, arguments: dict, workspace: Path) -> dict:
    """Dispatch one sandboxed tool call. Unknown tools return ok false."""
    try:
        validate(TOOL_SCHEMAS[name], arguments)
    except Exception as exc:  # schema errors are tool failures, not crashes
        return {"ok": False, "error": str(exc)}
    if name == "python":
        return run_python(arguments["code"])
    if name == "calculator":
        return run_calculator(arguments["expression"])
    if name == "read_file":
        return run_read_file(arguments["path"])
    if name == "write_file":
        return run_write_file(workspace, arguments["path"], arguments["content"])
    if name == "retrieve":
        return run_retrieve(arguments["query"])
    if name == "web_search":
        return run_web_search(arguments["query"])
    return {"ok": False, "error": f"unknown tool {name}"}
