import json
import re
from pathlib import Path

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
RESULTS_RE = re.compile(r"<!-- RESULTS_START -->(.*)<!-- RESULTS_END -->", re.S)


def _json_numbers(path: Path) -> set[float]:
    found = set()

    def walk(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.add(float(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(json.loads(path.read_text(encoding="utf-8")))
    return found


def _numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in NUMBER_RE.finditer(text)]


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-9 * max(1.0, abs(right))


def test_readme_numbers_come_from_committed_files():
    readme = Path("README.md").read_text(encoding="utf-8")
    results_match = RESULTS_RE.search(readme)
    assert results_match, "README needs a RESULTS block"
    results_text = results_match.group(1)
    outside = RESULTS_RE.sub(" ", readme)
    result_numbers = set()
    result_files = list(Path("results").glob("*.json"))
    result_files.extend(Path("results/v1_qwen0.5b").glob("*.json"))
    result_files.extend(Path("results/v2_qwen1.5b").glob("*.json"))
    result_files.extend(Path("results/v3_qwen1.5b").glob("*.json"))
    result_files.extend(Path("results/v3_api").glob("*/*.json"))
    for path in result_files:
        result_numbers |= _json_numbers(path)
    allowed_outside = set(result_numbers)
    for path in (
        Path("eval/manifest.json"),
        Path("data/stats.json"),
        Path("data/manifest.json"),
        Path("finetune/data/manifest.json"),
        Path("finetune/configs/qlora_distill.json"),
    ):
        allowed_outside |= _json_numbers(path)

    def missing(text, allowed):
        return [number for number in _numbers(text) if not any(_close(number, item) for item in allowed)]

    assert missing(results_text, result_numbers) == []
    assert missing(outside, allowed_outside) == []
    assert "pending" in results_text
    team = json.loads(Path("results/team_base.json").read_text(encoding="utf-8"))
    finetune = json.loads(Path("results/finetune_eval.json").read_text(encoding="utf-8"))
    assert team["status"] == "pending"
    assert finetune["status"] == "pending"
    assert team["metrics"] is None
    assert finetune["metrics"] is None
    assert "%" not in results_text


def test_results_block_matches_the_generator():
    from orbit_agents.eval.readme import render, updated_readme

    readme = Path("README.md").read_text(encoding="utf-8")
    assert updated_readme() == readme, "run python -m orbit_agents.eval.readme"
    assert "| Team distill | not scored | | | |" in render()


def test_generator_adds_measured_v3_rows_only(tmp_path):
    import shutil

    from orbit_agents.eval.readme import render

    for name in ("v1_qwen0.5b", "v2_qwen1.5b"):
        shutil.copytree(Path("results") / name, tmp_path / "results" / name)
    v3 = tmp_path / "results" / "v3_qwen1.5b"
    v3.mkdir()
    assert "Those v3 scores are not in yet." in render(tmp_path)
    measured = json.loads(Path("results/v2_qwen1.5b/team_base.json").read_text(encoding="utf-8"))
    (v3 / "team_distill.json").write_text(json.dumps(measured), encoding="utf-8")
    (v3 / "team_base.json").write_text(json.dumps({"status": "pending", "kind": "model_eval"}), encoding="utf-8")
    block = render(tmp_path)
    assert "| Team distill | 2 of 12 |" in block
    assert "| Team base | not scored | | | |" in block
    assert "no v3 difference is claimed" in block
    assert "scored only under harness v3 below" in block
