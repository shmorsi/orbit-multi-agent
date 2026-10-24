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
PROTOCOL_V2 = "code-eval-v2"
SCENARIO = Path("data") / "scenarios" / "code"
TASKS_REL = SCENARIO / "tasks.jsonl"
REPOS_REL = SCENARIO / "repos"
MANIFEST_REL = SCENARIO / "manifest.json"
EXCLUDE_DIRS = {"hidden", "oracle", "wrong", "partial", "__pycache__"}
SPLITS = {
    PROTOCOL: {"scenario": SCENARIO, "results": Path("results") / "code"},
    PROTOCOL_V2: {
        "scenario": Path("data") / "scenarios" / "code_v2",
        "results": Path("results") / "code_v2",
    },
}


def split_spec(protocol: str = PROTOCOL) -> dict:
    """Paths for one frozen code split. code-eval-v1 stays the default for the old callers."""
    try:
        spec = SPLITS[protocol]
    except KeyError:
        raise SystemExit(f"unknown code split {protocol}") from None
    scenario = spec["scenario"]
    return {
        "protocol": protocol,
        "scenario": scenario,
        "tasks": scenario / "tasks.jsonl",
        "repos": scenario / "repos",
        "manifest": scenario / "manifest.json",
        "results": spec["results"],
    }


def scenario_root(root: Path = ROOT, protocol: str = PROTOCOL) -> Path:
    return root / split_spec(protocol)["scenario"]


def tasks_path(root: Path = ROOT, protocol: str = PROTOCOL) -> Path:
    return root / split_spec(protocol)["tasks"]


def repos_path(root: Path = ROOT, protocol: str = PROTOCOL) -> Path:
    return root / split_spec(protocol)["repos"]


def load_tasks(root: Path = ROOT, protocol: str = PROTOCOL) -> list[dict]:
    """Every frozen task on one code split. The default split is code-eval-v1."""
    path = tasks_path(root, protocol)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_by_id(task_id: str, root: Path = ROOT, protocol: str = PROTOCOL) -> dict:
    for task in load_tasks(root, protocol):
        if task["id"] == task_id:
            return task
    raise KeyError(task_id)


def iter_repo_files(root: Path = ROOT, protocol: str = PROTOCOL) -> list[Path]:
    """Sorted repo files. The bundle hash walks this order."""
    repos = repos_path(root, protocol)
    files = [path for path in repos.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    return sorted(files, key=lambda path: path.relative_to(repos).as_posix())


def bundle_bytes(root: Path = ROOT, protocol: str = PROTOCOL) -> bytes:
    """Canonical bytes for the frozen split. Manifest is not included."""
    chunks = [b"tasks.jsonl\n", tasks_path(root, protocol).read_bytes(), b"\n"]
    repos = repos_path(root, protocol)
    for path in iter_repo_files(root, protocol):
        rel = path.relative_to(repos).as_posix().encode()
        chunks.extend([rel, b"\n", path.read_bytes(), b"\n"])
    return b"".join(chunks)


def bundle_sha256(root: Path = ROOT, protocol: str = PROTOCOL) -> str:
    return hashlib.sha256(bundle_bytes(root, protocol)).hexdigest()


def file_record(path: Path) -> dict:
    body = path.read_bytes()
    return {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def manifest_payload(root: Path = ROOT, protocol: str = PROTOCOL) -> dict:
    """The manifest body, without reading an existing manifest."""
    spec = split_spec(protocol)
    files = {spec["tasks"].as_posix(): file_record(tasks_path(root, protocol))}
    repos = repos_path(root, protocol)
    for path in iter_repo_files(root, protocol):
        rel = (spec["repos"] / path.relative_to(repos)).as_posix()
        files[rel] = file_record(path)
    tasks = load_tasks(root, protocol)
    return {
        "protocol": protocol,
        "split": "test",
        "split_tag": protocol,
        "n_tasks": len(tasks),
        "bundle_sha256": bundle_sha256(root, protocol),
        "files": files,
    }


def write_manifest(root: Path = ROOT, protocol: str = PROTOCOL) -> Path:
    path = root / split_spec(protocol)["manifest"]
    path.write_text(json.dumps(manifest_payload(root, protocol), indent=2) + "\n", encoding="utf-8")
    return path


def _git_show(root: Path, spec: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(root), "show", spec], capture_output=True, check=False)


def verify_benchmark(root: Path = ROOT, protocol: str = PROTOCOL) -> dict:
    """Check the working tree against the manifest and, when present, the git tag.

    A mismatch refuses the score. A missing tag is reported and does not invent a match.
    """
    spec = split_spec(protocol)
    manifest_path = root / spec["manifest"]
    if not manifest_path.is_file():
        raise SystemExit(f"missing {spec['manifest'].as_posix()}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fresh = manifest_payload(root, protocol)
    if manifest.get("protocol") != protocol or manifest.get("split_tag") != protocol:
        raise SystemExit(f"code benchmark manifest protocol does not match {protocol}")
    if manifest.get("bundle_sha256") != fresh["bundle_sha256"] or manifest.get("files") != fresh["files"]:
        raise SystemExit(f"{spec['scenario'].as_posix()} does not match its manifest. Do not score a changed frozen split.")
    if manifest.get("n_tasks") != fresh["n_tasks"]:
        raise SystemExit("code benchmark n_tasks does not match the task file")
    probe = _git_show(root, f"{protocol}:{spec['manifest'].as_posix()}")
    tag = "missing"
    if probe.returncode == 0:
        tagged = json.loads(probe.stdout.decode("utf-8"))
        for rel, meta in tagged.get("files", {}).items():
            blob = _git_show(root, f"{protocol}:{rel}")
            if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != meta.get("sha256"):
                raise SystemExit(f"{rel} differs from tag {protocol}. Do not score a changed frozen split.")
            work = root / rel
            if not work.is_file() or hashlib.sha256(work.read_bytes()).hexdigest() != meta.get("sha256"):
                raise SystemExit(f"{rel} differs from tag {protocol}. Do not score a changed frozen split.")
        if tagged.get("bundle_sha256") != fresh["bundle_sha256"]:
            raise SystemExit(f"bundle hash differs from tag {protocol}")
        tag = "matches"
    return {
        "protocol": protocol,
        "split": "test",
        "split_tag": protocol,
        "bundle_sha256": fresh["bundle_sha256"],
        "n_tasks": fresh["n_tasks"],
        "tag": tag,
    }


def _task_protocol(task: dict) -> str:
    return task.get("protocol") or PROTOCOL


def materialize(task: dict, dest: Path, root: Path = ROOT) -> Path:
    """Copy the repo the agent may edit. Hidden tests, the oracle, and the wrong draft stay out."""
    src = repos_path(root, _task_protocol(task)) / task["repo"]
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
    src = repos_path(root, _task_protocol(task)) / task["repo"]
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
    path = repos_path(root, _task_protocol(task)) / task["repo"] / name / task["entry"]
    return path.read_text(encoding="utf-8")
