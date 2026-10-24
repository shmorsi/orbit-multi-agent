"""code-eval-v2 stays frozen, harder than v1, and separate from the v1 record."""

import importlib.util
import shutil
from pathlib import Path

from orbit_agents.codebench import PROTOCOL_V2, load_tasks, source_files, verify_benchmark

V2_SHA256 = "045e19bc206a6a3e17ea1165976122690166577c176d06e20aa7040eaff84b15"


def _build():
    spec = importlib.util.spec_from_file_location("code_v2_build", Path("data/scenarios/code_v2/build.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_hash_task_count_and_traps():
    frozen = verify_benchmark(protocol=PROTOCOL_V2)
    tasks = load_tasks(protocol=PROTOCOL_V2)
    assert frozen["bundle_sha256"] == V2_SHA256
    assert frozen["n_tasks"] == len(tasks) == 32
    assert frozen["protocol"] == PROTOCOL_V2
    traps = {task["trap"] for task in tasks}
    assert {"hidden_only", "partial", "both_files", "test_edit"} <= traps
    assert sum(1 for task in tasks if len(source_files(task)) >= 2) >= 16
    assert verify_benchmark()["bundle_sha256"] == "18665421458c6661299b60c1aa982394c9d3b773660a16bb3d7066f090a7028f"
    try:
        _build().self_check()
    finally:
        shutil.rmtree(Path(".sandbox-v2"), ignore_errors=True)
