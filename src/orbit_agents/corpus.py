"""Load the bundled corpus and refuse files that do not match the manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from orbit_agents.paths import CORPUS, MANIFEST

TABLES = {"jobs": "jobs.csv", "shifts": "shifts.csv"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def corpus_files() -> list[Path]:
    return sorted(path for path in CORPUS.rglob("*") if path.is_file())


def corpus_guide() -> str:
    """Files and tables the worker can read. This does not open or hash the corpus."""
    files = ", ".join(path.relative_to(CORPUS).as_posix() for path in corpus_files())
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
    files = {}
    for path in corpus_files():
        relative = path.relative_to(CORPUS).as_posix()
        files[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    payload = {"corpus": "makerspace", "files": files}
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"missing {MANIFEST}; run python -m orbit_agents.eval.build_tasks")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify() -> dict:
    manifest = load_manifest()
    expected = manifest["files"]
    found = {path.relative_to(CORPUS).as_posix(): path for path in corpus_files()}
    if set(found) != set(expected):
        raise RuntimeError("corpus file set does not match data/manifest.json")
    for relative, meta in expected.items():
        path = found[relative]
        digest = file_sha256(path)
        if digest != meta["sha256"] or path.stat().st_size != meta["bytes"]:
            raise RuntimeError(f"corpus hash mismatch for {relative}")
    return manifest


def resolve_corpus(relative: str) -> Path:
    verify()
    if not relative or relative.startswith(("/", "\\")) or ".." in Path(relative).parts:
        raise ValueError("path must stay inside the bundled corpus")
    path = (CORPUS / relative).resolve()
    if CORPUS.resolve() not in path.parents and path != CORPUS.resolve():
        raise ValueError("path must stay inside the bundled corpus")
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path


def load_table(name: str) -> list[dict[str, str]]:
    if name not in TABLES:
        raise ValueError(f"unknown table {name}")
    path = resolve_corpus(TABLES[name])
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_text(relative: str) -> str:
    return resolve_corpus(relative).read_text(encoding="utf-8")
