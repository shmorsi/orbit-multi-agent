import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


def test_frozen_test_matches_tag():
    probe = subprocess.run(["git", "rev-parse", "--verify", "team-eval-v1"], capture_output=True, text=True)
    if probe.returncode != 0:
        if os.environ.get("REQUIRE_EVAL_TAG") == "1":
            pytest.fail("tag team-eval-v1 is missing")
        pytest.skip("tag team-eval-v1 is not present yet")
    blob = subprocess.check_output(["git", "show", "team-eval-v1:eval/test.jsonl"])
    digest = hashlib.sha256(blob).hexdigest()
    worktree = hashlib.sha256(Path("eval/test.jsonl").read_bytes()).hexdigest()
    manifest = json.loads(Path("eval/manifest.json").read_text(encoding="utf-8"))
    assert digest == worktree
    assert manifest["test_sha256"] == digest
    assert manifest["protocol"] == "team-eval-v1"
