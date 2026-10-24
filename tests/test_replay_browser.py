"""Mission Control in a real browser. Skips unless Chrome or ORBIT_BROWSER=1 is available."""

import os
import threading
from pathlib import Path

import pytest

from orbit_agents.serve import bind


def _launch(playwright):
    if os.environ.get("ORBIT_BROWSER") == "1":
        return playwright.chromium.launch(headless=True)
    chrome = next(
        (path for path in ("/usr/local/bin/google-chrome", "/usr/bin/google-chrome") if Path(path).is_file()),
        None,
    )
    if chrome is None:
        pytest.skip("set ORBIT_BROWSER=1 to run the Mission Control browser test")
    return playwright.chromium.launch(executable_path=chrome, headless=True)


def _playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        if os.environ.get("ORBIT_BROWSER") == "1":
            raise
        pytest.skip("playwright is not installed")
    return sync_playwright


def _scrub(page, value):
    page.evaluate(
        """(value) => {
          const scrub = document.getElementById("scrub");
          scrub.value = value == null ? scrub.max : String(value);
          scrub.dispatchEvent(new Event("input", { bubbles: true }));
        }""",
        value,
    )


def test_replay_keeps_v1_and_v2_and_shows_the_round_table():
    sync_playwright = _playwright()
    server = bind(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = _launch(playwright)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(f"http://127.0.0.1:{port}/mission/?trace=revise.jsonl", wait_until="networkidle")
            page.wait_for_function("() => Number(document.getElementById('scrub').max) > 0")
            page.wait_for_selector("#seat-planner")
            assert page.locator("#seat-v2-commander").count() == 0
            assert page.locator("#seat-v3-coordinator").count() == 0
            page.click("#play")
            page.click("#pause")
            assert page.locator("#steps").inner_text() != "00"
            _scrub(page, None)
            assert "critic" in page.locator("#log").inner_text().lower()

            page.locator("#trace").select_option("v2/revise.jsonl")
            page.wait_for_selector("#seat-v2-commander")
            assert page.locator("#seat-planner").is_hidden()
            _scrub(page, None)
            assert page.locator("#seat-v2-auditor").count() == 1
            assert "auditor" in page.locator("#log").inner_text().lower()

            page.locator("#trace").select_option("v3/refuse.jsonl")
            page.wait_for_selector("#seat-v3-coordinator")
            assert page.locator("#seat-v2-commander").count() == 0
            assert page.locator("#links").count() == 1
            _scrub(page, 4)
            assert page.locator("#links line").count() >= 1
            crew = page.locator("#crew").inner_text()
            assert crew
            _scrub(page, None)
            why = page.locator("#why-text").inner_text()
            assert "red team" in why.lower()
            assert "not the team's answer" in why.lower()
            assert "flash-refuse" in (page.locator("#stage").get_attribute("class") or "")
            assert page.locator("#scrub").get_attribute("aria-label") == "Scrub the replay"

            page.locator("#trace").select_option("revise.jsonl")
            page.wait_for_selector("#seat-planner")
            assert page.locator("#seat-v3-coordinator").count() == 0
            assert page.locator("#seat-planner").is_visible()

            page.set_viewport_size({"width": 390, "height": 800})
            page.locator("#trace").select_option("v3/accept.jsonl")
            page.wait_for_selector("#seat-v3-researcher")
            _scrub(page, None)
            page.locator("#stage").scroll_into_view_if_needed()
            box = page.locator("#seat-v3-researcher").bounding_box()
            stage = page.locator("#stage").bounding_box()
            assert box is not None and stage is not None and box["width"] > 0
            center_x = box["x"] + box["width"] / 2
            center_y = box["y"] + box["height"] / 2
            assert stage["x"] <= center_x <= stage["x"] + stage["width"]
            assert stage["y"] <= center_y <= stage["y"] + stage["height"]
            browser.close()
    finally:
        server.shutdown()
