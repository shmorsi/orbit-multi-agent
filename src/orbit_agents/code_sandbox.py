"""Pytest in a temp workspace. The child has a timeout and no network.

The reviewer is code. It rejects a patch whose tests fail, whose run times out,
or whose diff touches a test file.
"""

from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys
from pathlib import Path

PYTEST_TIMEOUT_SECONDS = 8
_TEST_RE = re.compile(r"(^|/)(tests|hidden|hidden_tests)/|(^|/)(test_[^/]*\.py|conftest\.py)$")
_COUNT_RE = re.compile(r"(\d+) (passed|failed|error)")

# Installed before pytest starts. connect and name lookup fail closed.
_CHILD = """
import socket

def _blocked(*args, **kwargs):
    raise OSError("network disabled in the Orbit code sandbox")

socket.socket.connect = _blocked
socket.socket.connect_ex = lambda self, address: 1
socket.create_connection = _blocked
socket.getaddrinfo = _blocked

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""


def _summary(stdout: str, stderr: str, *, timed_out: bool) -> str:
    if timed_out:
        return "timed out"
    text = (stdout or stderr).strip()
    if not text:
        return "no output"
    return text.splitlines()[-1][:240]


def _counts(stdout: str) -> dict[str, int]:
    found = {"passed": 0, "failed": 0, "error": 0}
    for count, word in _COUNT_RE.findall(stdout or ""):
        if word in found:
            found[word] = int(count)
    return found


def _env(workspace: Path) -> dict[str, str]:
    """Drop credentials and proxies. The child still needs a normal interpreter."""
    env = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PROXY", "CREDENTIAL")):
            continue
        if key.startswith("ORBIT_"):
            continue
        env[key] = value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["ORBIT_SANDBOX"] = "1"
    env.pop("PYTHONPATH", None)
    env["PYTHONSAFEPATH"] = "1"
    return env


def run_pytest(workspace: Path, *, timeout: float = PYTEST_TIMEOUT_SECONDS, select: str = "tests") -> dict:
    """Run pytest in workspace. A timeout or a socket attempt does not escape the child."""
    try:
        # -o addopts= drops a parent project's extra -q. Two -q flags hide the
        # "N passed" line, and the reviewer would treat a green run as a failure.
        # An absolute rootdir and confcutdir keep this checkout's pytest config out.
        root = str(workspace.resolve())
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _CHILD,
                "-q",
                "--tb=line",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                "-o",
                "pythonpath=",
                "--rootdir",
                root,
                "--confcutdir",
                root,
                select,
            ],
            cwd=workspace,
            env=_env(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        counts = _counts(stdout)
        return {
            "ok": False,
            "timed_out": True,
            "returncode": None,
            "stdout": stdout[-2000:],
            "stderr": (stderr or "timeout")[-2000:],
            "summary": _summary(stdout, stderr, timed_out=True),
            "n_passed": counts["passed"],
            "n_failed": counts["failed"],
            "n_errors": counts["error"],
        }
    counts = _counts(proc.stdout)
    ok = proc.returncode == 0 and counts["passed"] >= 1 and counts["failed"] == 0 and counts["error"] == 0
    return {
        "ok": ok,
        "timed_out": False,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-2000:],
        "summary": _summary(proc.stdout or "", proc.stderr or "", timed_out=False),
        "n_passed": counts["passed"],
        "n_failed": counts["failed"],
        "n_errors": counts["error"],
    }


def is_test_path(path: str) -> bool:
    """True when a repo-relative path is a test the reviewer must protect."""
    cleaned = path.replace("\\", "/").lstrip("./")
    return _TEST_RE.search(cleaned) is not None


def touches_tests(diff: str) -> bool:
    """True when a unified diff adds, removes, or rewrites a test path."""
    for line in diff.splitlines():
        if not (line.startswith("+++ ") or line.startswith("--- ")):
            continue
        raw = line[4:].split("\t", 1)[0].strip()
        if raw == "/dev/null":
            continue
        if raw.startswith("a/") or raw.startswith("b/"):
            raw = raw[2:]
        if is_test_path(raw):
            return True
    return False


def safe_relative(path: str) -> str:
    """A repo-relative edit path. Absolute paths and parent segments are refused."""
    if not path or path.startswith(("/", "\\")) or ".." in Path(path).parts:
        raise ValueError("edit path escapes the workspace")
    cleaned = path.replace("\\", "/").lstrip("./")
    if not cleaned or cleaned.startswith("../"):
        raise ValueError("edit path escapes the workspace")
    return cleaned


def snapshot(workspace: Path) -> dict[str, str]:
    """Text of every file currently in the workspace, keyed by relative path."""
    found = {}
    for path in workspace.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        found[path.relative_to(workspace).as_posix()] = path.read_text(encoding="utf-8")
    return found


def apply_edits(workspace: Path, edits: list[dict]) -> None:
    """Write full file contents. The caller checks the diff afterwards."""
    for edit in edits:
        relative = safe_relative(edit["path"])
        target = workspace / relative
        if not target.resolve().is_relative_to(workspace.resolve()):
            raise ValueError("edit path escapes the workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit["content"], encoding="utf-8")


def unified_diff(before: dict[str, str], workspace: Path) -> str:
    """Diff from the original snapshot to the workspace, stable path order."""
    after = snapshot(workspace)
    paths = sorted(set(before) | set(after))
    chunks = []
    for relative in paths:
        old = before.get(relative, "")
        new = after.get(relative, "")
        if old == new:
            continue
        chunks.append(
            "".join(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
        )
    return "".join(chunks)


def review_patch(*, diff: str, pytest_result: dict) -> dict:
    """Reject a failing run, a timeout, an empty run, or a diff that touches tests."""
    if touches_tests(diff):
        return {
            "verdict": "reject",
            "reason": "The diff touches a test file. The reviewer rejected the patch.",
        }
    if pytest_result.get("timed_out"):
        return {
            "verdict": "reject",
            "reason": "The test run timed out. The reviewer rejected the patch.",
        }
    if not pytest_result.get("ok"):
        return {
            "verdict": "reject",
            "reason": "Tests failed. The reviewer rejected the patch.",
        }
    return {
        "verdict": "accept",
        "reason": "Tests passed and the diff does not touch tests.",
    }


def claims_grounded(claims: list, pytest_result: dict | None, review: dict | None) -> bool:
    """A tests_passed claim stays only when pytest passed and the reviewer accepted.

    An empty claim list is grounded when nothing was shipped. This is the code
    gate, the same idea as the makerspace grounding gate: the model does not get
    the last word.
    """
    if review is None or pytest_result is None:
        return not claims
    if not claims:
        return True
    if review.get("verdict") != "accept" or not pytest_result.get("ok"):
        return False
    for claim in claims:
        if claim.get("name") != "tests_passed":
            return False
        if claim.get("citation") != "pytest":
            return False
        if claim.get("value") is not True:
            return False
    return True
