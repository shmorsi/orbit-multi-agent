import socket
from pathlib import Path

from orbit_agents.tools import call_tool, run_python, run_web_search


def test_calculator_and_true_division():
    workspace = Path("traces") / "test-tools"
    assert call_tool("calculator", {"expression": "2 + 2"}, workspace)["result"] == 4
    blocked = call_tool("calculator", {"expression": "2 / 2"}, workspace)
    assert blocked["ok"] is False


def test_python_rejects_import_and_times_out():
    blocked = run_python("import os\nprint(os.getcwd())")
    assert blocked["ok"] is False
    assert "blocked" in blocked["error"]
    timed = run_python("while True:\n    pass")
    assert timed["ok"] is False
    assert "TimeoutError" in timed["error"]


def test_paths_stay_inside_the_corpus_and_workspace(tmp_path):
    escaped = call_tool("read_file", {"path": "../pyproject.toml"}, tmp_path)
    assert escaped["ok"] is False
    written = call_tool("write_file", {"path": "../escape.txt", "content": "no"}, tmp_path)
    assert written["ok"] is False
    assert not Path("escape.txt").exists()
    kept = call_tool("write_file", {"path": "note.txt", "content": "yes"}, tmp_path)
    assert kept["ok"] is True
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "yes"


def test_web_search_is_offline_cache_only():
    source = Path("src/orbit_agents/tools.py").read_text(encoding="utf-8")
    start = source.index("def run_web_search")
    body = source[start:source.index("\ndef call_tool")]
    assert "socket" not in body
    assert "urllib" not in body
    real_socket = socket.socket

    def boom(*_args, **_kwargs):
        raise AssertionError("web search opened a socket")

    socket.socket = boom
    try:
        found = run_web_search("emergency stop")
    finally:
        socket.socket = real_socket
    assert found["offline"] is True
    assert found["results"]
