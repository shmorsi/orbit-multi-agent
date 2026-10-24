"""Landing page, citation, and local links. No network."""

import re
from pathlib import Path

README = Path("README.md").read_text(encoding="utf-8")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def test_citation_names_the_author_and_the_project_page():
    text = Path("CITATION.cff").read_text(encoding="utf-8")
    assert "cff-version: 1.2.0" in text
    assert "family-names: Morsi" in text
    assert "given-names: Sherif" in text
    assert "license: MIT" in text
    assert "https://github.com/shmorsi/orbit-multi-agent" in text
    assert "https://shmorsi.github.io/orbit/" in text


def test_readme_is_a_landing_page():
    for heading in (
        "## Why this matters",
        "## Key results",
        "## Architecture",
        "## v2: six roles",
        "## Quickstart",
        "## Reproducibility and integrity",
        "## What I'd do next",
    ):
        assert heading in README
    assert "```mermaid" in README.split("## Why this matters", 1)[0]
    assert "aria-label=" in README
    assert "coming soon" not in README.lower()
    assert "TODO" not in README
    images = IMG_RE.findall(README)
    assert images
    for alt, src in images:
        assert alt.strip(), src
        if src.startswith(("http://", "https://")):
            continue
        assert Path(src).is_file(), src
    assert 'src="docs/mission-control.mp4"' in README


def test_dev_guide_names_locked_paths_and_the_generator():
    text = Path("docs/DEVELOPMENT.md").read_text(encoding="utf-8")
    for path in ("eval/test.jsonl", "results/", "finetune/PROTOCOL.md", "mission/board.js"):
        assert path in text
    assert "orbit_agents.eval.readme" in text
    assert "ORBIT_ANTHROPIC_API_KEY" in text
    assert "coming soon" not in text.lower()


def test_readme_local_links_resolve():
    for match in LINK_RE.finditer(README):
        target = match.group(1)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#", 1)[0]
        assert Path(path).exists(), target
