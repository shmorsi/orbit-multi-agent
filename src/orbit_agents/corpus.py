"""Load the bundled corpus and refuse files that do not match the manifest.

ORBIT_CORPUS and ORBIT_CORPUS_MANIFEST point a demo at another fictional corpus.
Unset, every path stays the makerspace corpus. The frozen test uses that default.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

from orbit_agents.paths import CORPUS, MANIFEST

TABLES = {"jobs": "jobs.csv", "shifts": "shifts.csv"}


def active_corpus() -> Path:
    """Corpus root for this process. The child python tool inherits the same env."""
    override = os.environ.get("ORBIT_CORPUS", "").strip()
    if override:
        return Path(override).resolve()
    return CORPUS


def active_manifest() -> Path:
    """Manifest that matches active_corpus()."""
    override = os.environ.get("ORBIT_CORPUS_MANIFEST", "").strip()
    if override:
        return Path(override).resolve()
    return MANIFEST


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def corpus_files() -> list[Path]:
    """Every file under the active corpus, in stable order."""
    root = active_corpus()
    return sorted(path for path in root.rglob("*") if path.is_file())


def corpus_guide() -> str:
    """Files and tables the worker can read. This does not open or hash the corpus."""
    root = active_corpus()
    files = ", ".join(path.relative_to(root).as_posix() for path in corpus_files())
    tables = ", ".join(f"{name} ({relative})" for name, relative in sorted(TABLES.items()))
    return (
        "Bundled files: "
        + files
        + ". Tables: "
        + tables
        + ". In python, table(name) returns the rows of a table and text(path) returns one of these files. "
        + "print the value you computed. Do not paste file text into code. "
        + "read_file uses the same relative path."
    )


def write_manifest() -> dict:
    """Rewrite the active manifest from the current corpus bytes."""
    root = active_corpus()
    files = {}
    for path in corpus_files():
        relative = path.relative_to(root).as_posix()
        files[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    label = "makerspace" if root == CORPUS.resolve() else root.name
    payload = {"corpus": label, "files": files}
    active_manifest().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_manifest() -> dict:
    """Read the corpus manifest, or exit if it has not been written."""
    path = active_manifest()
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run python -m orbit_agents.eval.build_tasks")
    return json.loads(path.read_text(encoding="utf-8"))


def verify() -> dict:
    """Raise if the corpus file set or any hash disagrees with the manifest."""
    manifest = load_manifest()
    expected = manifest["files"]
    root = active_corpus()
    found = {path.relative_to(root).as_posix(): path for path in corpus_files()}
    if set(found) != set(expected):
        raise RuntimeError("corpus file set does not match data/manifest.json")
    for relative, meta in expected.items():
        path = found[relative]
        digest = file_sha256(path)
        if digest != meta["sha256"] or path.stat().st_size != meta["bytes"]:
            raise RuntimeError(f"corpus hash mismatch for {relative}")
    return manifest


def resolve_corpus(relative: str) -> Path:
    """Resolve one relative corpus path after the manifest check."""
    verify()
    root = active_corpus()
    if not relative or relative.startswith(("/", "\\")) or ".." in Path(relative).parts:
        raise ValueError("path must stay inside the bundled corpus")
    path = (root / relative).resolve()
    if root not in path.parents and path != root:
        raise ValueError("path must stay inside the bundled corpus")
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path


def load_table(name: str) -> list[dict[str, str]]:
    """Load a named CSV table from the corpus."""
    if name not in TABLES:
        raise ValueError(f"unknown table {name}")
    path = resolve_corpus(TABLES[name])
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_text(relative: str) -> str:
    """Read one bundled text file."""
    return resolve_corpus(relative).read_text(encoding="utf-8")
