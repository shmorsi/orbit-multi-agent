from pathlib import Path


def test_recorded_replay_is_saved():
    mp4 = Path("docs/mission-control.mp4")
    gif = Path("docs/mission-control.gif")
    assert mp4.is_file()
    assert gif.is_file()
    assert mp4.stat().st_size > 50_000
    assert gif.stat().st_size > 20_000
    assert b"ftyp" in mp4.read_bytes()[:32]
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "docs/mission-control.mp4" in readme
    assert "docs/mission-control.gif" in readme
    script = Path("demo/record_replay.py").read_text(encoding="utf-8")
    assert "revise.jsonl" in script
    assert "Page.captureScreenshot" in script
