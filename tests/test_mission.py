import json
from pathlib import Path

from orbit_agents.trace import bundled_traces


def test_committed_traces_match_the_engine(tmp_path):
    written = bundled_traces(tmp_path)
    for path in written:
        committed = Path("mission/traces") / path.name
        assert path.read_bytes() == committed.read_bytes()
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert header["fabricated"] is False
        assert header["source"] == "orbit_agents.engine"
        assert header["harness"] == "scripted_harness"
        assert "max_turns" in header["flight_rules"]


def test_page_has_playback_and_reduced_motion():
    html = Path("mission/index.html").read_text(encoding="utf-8")
    css = Path("mission/style.css").read_text(encoding="utf-8")
    script = Path("mission/replay.js").read_text(encoding="utf-8")
    assert 'id="play"' in html
    assert 'id="pause"' in html
    assert 'id="scrub"' in html
    assert 'id="phases"' in html
    assert 'id="telemetry"' in html
    assert 'id="success-chart"' in html
    assert 'id="schema-chart"' in html
    assert 'id="loss-chart"' in html
    assert 'id="paired-chart"' in html
    assert 'id="traps"' in html
    assert "Task-success intervals include zero" in html
    assert "team distill" in html.lower()
    assert "prefers-reduced-motion" in css
    assert "/api/run" in script
    assert "Inter Tight" in css
    assert Path("mission/fonts/inter-tight-latin.woff2").stat().st_size > 1000
    assert "SIL OPEN FONT LICENSE" in Path("mission/fonts/OFL.txt").read_text(encoding="utf-8")


def test_board_matches_the_measured_v2_files():
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "mission/build_board.py", "--check"])
    text = Path("mission/board.js").read_text(encoding="utf-8")
    payload = json.loads(text.removeprefix("window.ORBIT_BOARD = ").removesuffix(";\n"))
    by_id = {run["id"]: run for run in payload["runs"]}
    assert by_id["single_base"]["n_success"] == 2
    assert by_id["team_base"]["n_success"] == 2
    assert by_id["team_lora"]["n_success"] == 3
    assert by_id["team_lora"]["n_correct_refusals"] == 1
    assert by_id["team_distill"]["pending"] is True
    assert by_id["team_distill"]["n_success"] is None
    assert payload["paired"]["task_success_rate"]["low"] < 0 < payload["paired"]["task_success_rate"]["high"]
    assert payload["loss"][0]["step"] == 10
    assert payload["loss"][-1]["step"] == 150


def test_no_macro_language_in_the_product():
    roots = [Path("src/orbit_agents"), Path("mission"), Path("README.md"), Path("data/corpus")]
    banned = ("gdp", "inflation", "world bank", "macro_agent")
    for root in roots:
        files = [root] if root.is_file() else list(root.rglob("*"))
        for path in files:
            if not path.is_file() or path.suffix.lower() in {".woff2", ".png", ".gif", ".mp4"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for word in banned:
                assert word not in text, f"{word} in {path}"
