"""Frozen Orbit Code split. The hash covers the tasks and the mini repos.

Hidden tests and oracle files are in the hash. They are not copied into the
agent workspace and they are not placed in the prompt.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from orbit_agents.paths import ROOT

PROTOCOL = "code-eval-v1"
SCENARIO = Path("data") / "scenarios" / "code"
TASKS_REL = SCENARIO / "tasks.jsonl"
REPOS_REL = SCENARIO / "repos"
MANIFEST_REL = SCENARIO / "manifest.json"
EXCLUDE_DIRS = {"hidden", "oracle", "wrong", "__pycache__"}


def scenario_root(root: Path = ROOT) -> Path:
    return root / SCENARIO


def tasks_path(root: Path = ROOT) -> Path:
    return root / TASKS_REL


def repos_path(root: Path = ROOT) -> Path:
    return root / REPOS_REL


def load_tasks(root: Path = ROOT) -> list[dict]:
    """Every frozen code task. The split tag on the manifest is code-eval-v1."""
    path = tasks_path(root)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_by_id(task_id: str, root: Path = ROOT) -> dict:
    for task in load_tasks(root):
        if task["id"] == task_id:
            return task
    raise KeyError(task_id)


def iter_repo_files(root: Path = ROOT) -> list[Path]:
    """Sorted repo files. The bundle hash walks this order."""
    repos = repos_path(root)
    files = [path for path in repos.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    return sorted(files, key=lambda path: path.relative_to(repos).as_posix())


def bundle_bytes(root: Path = ROOT) -> bytes:
    """Canonical bytes for the frozen split. Manifest is not included."""
    chunks = [b"tasks.jsonl\n", tasks_path(root).read_bytes(), b"\n"]
    repos = repos_path(root)
    for path in iter_repo_files(root):
        rel = path.relative_to(repos).as_posix().encode()
        chunks.extend([rel, b"\n", path.read_bytes(), b"\n"])
    return b"".join(chunks)


def bundle_sha256(root: Path = ROOT) -> str:
    return hashlib.sha256(bundle_bytes(root)).hexdigest()


def file_record(path: Path) -> dict:
    body = path.read_bytes()
    return {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def manifest_payload(root: Path = ROOT) -> dict:
    """The manifest body, without reading an existing manifest."""
    files = {TASKS_REL.as_posix(): file_record(tasks_path(root))}
    repos = repos_path(root)
    for path in iter_repo_files(root):
        rel = (REPOS_REL / path.relative_to(repos)).as_posix()
        files[rel] = file_record(path)
    tasks = load_tasks(root)
    return {
        "protocol": PROTOCOL,
        "split": "test",
        "split_tag": PROTOCOL,
        "n_tasks": len(tasks),
        "bundle_sha256": bundle_sha256(root),
        "files": files,
    }


def write_manifest(root: Path = ROOT) -> Path:
    path = root / MANIFEST_REL
    path.write_text(json.dumps(manifest_payload(root), indent=2) + "\n", encoding="utf-8")
    return path


def _git_show(root: Path, spec: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(root), "show", spec], capture_output=True, check=False)


def verify_benchmark(root: Path = ROOT) -> dict:
    """Check the working tree against the manifest and, when present, the git tag.

    A mismatch refuses the score. A missing tag is reported and does not invent a match.
    """
    manifest_path = root / MANIFEST_REL
    if not manifest_path.is_file():
        raise SystemExit(f"missing {MANIFEST_REL.as_posix()}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fresh = manifest_payload(root)
    if manifest.get("protocol") != PROTOCOL or manifest.get("split_tag") != PROTOCOL:
        raise SystemExit("code benchmark manifest protocol does not match code-eval-v1")
    if manifest.get("bundle_sha256") != fresh["bundle_sha256"] or manifest.get("files") != fresh["files"]:
        raise SystemExit("data/scenarios/code does not match its manifest. Do not score a changed frozen split.")
    if manifest.get("n_tasks") != fresh["n_tasks"]:
        raise SystemExit("code benchmark n_tasks does not match the task file")
    probe = _git_show(root, f"{PROTOCOL}:{MANIFEST_REL.as_posix()}")
    tag = "missing"
    if probe.returncode == 0:
        tagged = json.loads(probe.stdout.decode("utf-8"))
        for rel, meta in tagged.get("files", {}).items():
            blob = _git_show(root, f"{PROTOCOL}:{rel}")
            if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != meta.get("sha256"):
                raise SystemExit(f"{rel} differs from tag {PROTOCOL}. Do not score a changed frozen split.")
            work = root / rel
            if not work.is_file() or hashlib.sha256(work.read_bytes()).hexdigest() != meta.get("sha256"):
                raise SystemExit(f"{rel} differs from tag {PROTOCOL}. Do not score a changed frozen split.")
        if tagged.get("bundle_sha256") != fresh["bundle_sha256"]:
            raise SystemExit(f"bundle hash differs from tag {PROTOCOL}")
        tag = "matches"
    return {
        "protocol": PROTOCOL,
        "split": "test",
        "split_tag": PROTOCOL,
        "bundle_sha256": fresh["bundle_sha256"],
        "n_tasks": fresh["n_tasks"],
        "tag": tag,
    }


def materialize(task: dict, dest: Path, root: Path = ROOT) -> Path:
    """Copy the repo the agent may edit. Hidden tests, the oracle, and the wrong draft stay out."""
    src = repos_path(root) / task["repo"]
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(src)
        if any(part in EXCLUDE_DIRS for part in relative.parts):
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return dest


def source_files(task: dict, root: Path = ROOT) -> list[tuple[str, str]]:
    """Buggy source only. Tests, hidden tests, and the oracle are not in the prompt."""
    src = repos_path(root) / task["repo"]
    found = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(src)
        if any(part in EXCLUDE_DIRS or part == "tests" for part in relative.parts):
            continue
        if path.name == "ISSUE.txt":
            continue
        if path.suffix != ".py":
            continue
        found.append((relative.as_posix(), path.read_text(encoding="utf-8")))
    return found


def task_prompt(task: dict, root: Path = ROOT) -> str:
    """The issue and the source. Hidden tests are not included."""
    parts = [
        f"Issue: {task['issue']}",
        "Edit the source so the issue is fixed. Do not edit tests. Reply with one JSON object.",
    ]
    for path, text in source_files(task, root):
        parts.append(f"FILE {path}\n{text}")
    return "\n\n".join(parts)


def read_variant(task: dict, name: str, root: Path = ROOT) -> str:
    """Read oracle/ or wrong/ for the scripted harness. The agent workspace does not contain these."""
    path = repos_path(root) / task["repo"] / name / task["entry"]
    return path.read_text(encoding="utf-8")
