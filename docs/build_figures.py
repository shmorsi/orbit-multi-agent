"""Draw the README diagrams from the measured results and the engine constants.

The type is Inter Tight, already bundled for Mission Control, outlined to paths so the
picture does not depend on a font being installed. The Qwen figures come from
results/v2_qwen1.5b. When a measured API pair exists, the third column reads
results/v3_api. Nothing in this file is a new measurement.

    python docs/build_figures.py
    python docs/build_figures.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fontTools.pens.svgPathPen import SVGPathPen  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402
from fontTools.varLib.instancer import instantiateVariableFont  # noqa: E402

from orbit_agents.engine import MAX_CRITIC_RETRIES, MAX_SCHEMA_TRIES, MAX_TURNS  # noqa: E402
from orbit_agents.eval.readme import display_number  # noqa: E402
from orbit_agents.schema import PLANNER_SCHEMA, TOOL_NAMES  # noqa: E402

FONT = ROOT / "mission" / "fonts" / "inter-tight-latin.woff2"
HERO = ROOT / "docs" / "hero.svg"
ARCHITECTURE = ROOT / "docs" / "architecture.svg"

BG = "#000000"
INK = "#f5f5f7"
MUTED = "#a8a8ad"
DIM = "#8e8e93"
LINE = "rgba(255,255,255,0.14)"
LINE_STRONG = "rgba(255,255,255,0.34)"
ACCENT = "#8ec5ff"
GRID = "rgba(255,255,255,0.035)"

STARS = (
    (72, 36, 1.1),
    (188, 92, 0.7),
    (1040, 48, 0.9),
    (1128, 140, 0.6),
    (64, 500, 0.8),
    (1144, 470, 1.0),
    (960, 24, 0.5),
    (300, 520, 0.6),
)


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Face:
    """One static weight of Inter Tight, with glyph outlines in font units."""

    def __init__(self, weight: float):
        font = instantiateVariableFont(TTFont(FONT), {"wght": weight}, inplace=False)
        self.upem = font["head"].unitsPerEm
        self.cmap = font.getBestCmap()
        self.glyphs = font.getGlyphSet()
        self._cache: dict[str, tuple[str, int]] = {}

    def glyph(self, char: str) -> tuple[str, int]:
        name = self.cmap.get(ord(char))
        if name is None:
            raise ValueError(f"Inter Tight has no glyph for {char!r}")
        cached = self._cache.get(name)
        if cached is None:
            glyph = self.glyphs[name]
            pen = SVGPathPen(self.glyphs)
            glyph.draw(pen)
            cached = (pen.getCommands() or "", glyph.width)
            self._cache[name] = cached
        return cached

    def width(self, text: str, size: float, tracking: float) -> float:
        if not text:
            return 0.0
        scale = size / self.upem
        total = 0.0
        for index, char in enumerate(text):
            _path, advance = self.glyph(char)
            total += advance * scale
            if index != len(text) - 1:
                total += tracking
        return total


class Canvas:
    def __init__(self, width: int, height: int, title: str, desc: str):
        self.width = width
        self.height = height
        self.title = title
        self.desc = desc
        self.parts: list[str] = []
        self.faces = {weight: Face(weight) for weight in (440, 560, 640)}

    def grid(self) -> None:
        self.parts.append(f'<rect width="{self.width}" height="{self.height}" fill="{BG}"/>')
        step = 80
        lines = []
        x = 0
        while x <= self.width:
            lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{self.height}" stroke="{GRID}"/>')
            x += step
        y = 0
        while y <= self.height:
            lines.append(f'<line x1="0" y1="{y}" x2="{self.width}" y2="{y}" stroke="{GRID}"/>')
            y += step
        self.parts.append(f'<g stroke-width="1">{"".join(lines)}</g>')
        dots = []
        for x, y, radius in STARS:
            if x < self.width and y < self.height:
                dots.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{INK}" opacity="0.45"/>')
        self.parts.append("".join(dots))

    def rule(self, x1: float, x2: float, y: float, stroke: str = LINE) -> None:
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1"/>'
        )

    def rect(self, x: float, y: float, w: float, h: float, stroke: str) -> None:
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )

    def text(
        self,
        content: str,
        x: float,
        y: float,
        size: float,
        *,
        weight: float = 440,
        fill: str = INK,
        tracking: float = 0.0,
        anchor: str = "start",
    ) -> float:
        face = self.faces[weight]
        width = face.width(content, size, tracking)
        if anchor == "middle":
            x -= width / 2
        elif anchor == "end":
            x -= width
        scale = size / face.upem
        paths = []
        pen_x = 0
        for index, char in enumerate(content):
            commands, advance = face.glyph(char)
            if commands:
                paths.append(f'<path d="{commands}" transform="translate({pen_x:.2f},0)"/>')
            pen_x += advance
            if index != len(content) - 1:
                pen_x += tracking / scale
        self.parts.append(
            f'<g fill="{fill}" transform="translate({x:.2f},{y:.2f}) scale({scale:.6f},{-scale:.6f})">{"".join(paths)}</g>'
        )
        return width

    def arrow(self, x1: float, x2: float, y: float, stroke: str = DIM) -> None:
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2 - 8:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1"/>'
            f'<path d="M {x2 - 8:.1f} {y - 3.5:.1f} L {x2:.1f} {y:.1f} L {x2 - 8:.1f} {y + 3.5:.1f}" '
            f'fill="none" stroke="{stroke}" stroke-width="1"/>'
        )

    def svg(self) -> str:
        body = "\n  ".join(self.parts)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'role="img" aria-labelledby="title desc">\n'
            f"  <title>{_xml(self.title)}</title>\n"
            f"  <desc>{_xml(self.desc)}</desc>\n"
            f"  {body}\n"
            "</svg>\n"
        )


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _phrase(interval: dict) -> str:
    if interval["low"] > 0:
        return "interval above zero"
    if interval["high"] < 0:
        return "interval below zero"
    return "interval includes zero"


def story(root: Path = ROOT) -> dict[str, str]:
    """The hero copy. Qwen figures come from the v2 result files."""
    folder = root / "results" / "v2_qwen1.5b"
    base = _read(folder / "team_base.json")
    lora = _read(folder / "team_lora.json")
    pair = _read(folder / "comparison.json")["pairs"]["team_lora_minus_team_base"]
    return {
        "base_schema": display_number(base["schema_valid_rate"]),
        "lora_schema": display_number(lora["schema_valid_rate"]),
        "base_success": f"{base['n_success']} of {base['n_tasks']}",
        "lora_success": f"{lora['n_success']} of {lora['n_tasks']}",
        "schema_phrase": _phrase(pair["schema_valid_rate"]),
        "success_phrase": _phrase(pair["task_success_rate"]),
        "model": str(base["model"]),
    }


def api_story(root: Path = ROOT) -> dict[str, str] | None:
    """Measured v1 and v2 API scores, if both files and their comparison exist."""
    folder = root / "results" / "v3_api"
    if not folder.is_dir():
        return None
    for model_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
        left_path = model_dir / "team_v1.json"
        right_path = model_dir / "team_v2.json"
        comparison_path = model_dir / "comparison.json"
        if not (left_path.is_file() and right_path.is_file() and comparison_path.is_file()):
            continue
        left = _read(left_path)
        right = _read(right_path)
        comparison = _read(comparison_path)
        if left.get("status") != "measured" or right.get("status") != "measured":
            continue
        if comparison.get("status") != "measured":
            continue
        pair = comparison.get("pairs", {}).get("team_v2_minus_team_v1")
        if not pair:
            continue
        return {
            "model": str(left["model"]),
            "v1_success": f"{left['n_success']} of {left['n_tasks']}",
            "v2_success": f"{right['n_success']} of {right['n_tasks']}",
            "success_phrase": _phrase(pair["task_success_rate"]),
            "schema_phrase": _phrase(pair["schema_valid_rate"]),
        }
    return None


def hero_svg(root: Path = ROOT) -> str:
    facts = story(root)
    api = api_story(root)
    if api is None:
        desc = (
            f"Schema-valid rate {facts['base_schema']} to {facts['lora_schema']}, {facts['schema_phrase']}. "
            f"Task success {facts['base_success']} to {facts['lora_success']}, {facts['success_phrase']}. "
            f"Model {facts['model']}. Claude v1, Claude v2, and the T4 distill run are pending, not measured."
        )
        third = (840, "03  PENDING", "Claude v1 and v2", None, "not measured", "T4 QLoRA distill", False)
    else:
        desc = (
            f"Schema-valid rate {facts['base_schema']} to {facts['lora_schema']}, {facts['schema_phrase']}. "
            f"Task success {facts['base_success']} to {facts['lora_success']}, {facts['success_phrase']}. "
            f"Model {facts['model']}. API task success {api['v1_success']} to {api['v2_success']}, "
            f"{api['success_phrase']}. API schema-valid {api['schema_phrase']}. "
            "The T4 distill run and the round table are pending, not measured."
        )
        third = (
            840,
            "03  API TASK SUCCESS",
            api["v1_success"],
            api["v2_success"],
            api["success_phrase"],
            api["model"],
            False,
        )
    canvas = Canvas(1200, 560, "Orbit", desc)
    canvas.grid()
    canvas.rule(56, 1144, 48)
    canvas.text("ORBIT", 56, 82, 13, weight=640, tracking=3.4)
    canvas.text("MULTI-AGENT MISSION CONTROL", 1144, 82, 12, weight=560, fill=MUTED, tracking=2.2, anchor="end")
    canvas.text("Answers only what the tools support.", 56, 178, 52, weight=640, tracking=-1.0)
    canvas.text(
        "A local planner, worker, and critic. Every turn is one JSON object.",
        56,
        222,
        18,
        weight=440,
        fill=MUTED,
    )
    canvas.text("Six sandboxed tools. A gate that refuses the rest.", 56, 250, 18, weight=440, fill=MUTED)
    canvas.rule(56, 1144, 286)

    columns = (
        (
            56,
            "01  SCHEMA-VALID",
            facts["base_schema"],
            facts["lora_schema"],
            facts["schema_phrase"],
            "Qwen2.5-1.5B  QLoRA  harness v2",
            True,
        ),
        (
            448,
            "02  TASK SUCCESS",
            facts["base_success"],
            facts["lora_success"],
            facts["success_phrase"],
            "frozen test  one score each",
            False,
        ),
        third,
    )
    for x, label, left, right, phrase, detail, accent in columns:
        canvas.text(label, x, 330, 12, weight=560, fill=ACCENT if accent else DIM, tracking=1.8)
        if accent:
            canvas.rule(x, x + 36, 344, ACCENT)
        width = canvas.text(left, x, 404, 30, weight=560, fill=INK, tracking=-0.4)
        if right is not None:
            canvas.arrow(x + width + 14, x + width + 48, 394, ACCENT if accent else DIM)
            canvas.text(right, x + width + 58, 404, 30, weight=560, fill=ACCENT if accent else INK, tracking=-0.4)
        canvas.text(phrase, x, 444, 15, weight=440, fill=MUTED)
        canvas.text(detail, x, 472, 14, weight=440, fill=DIM)
    canvas.rule(56, 1144, 504)
    canvas.text(
        "FULL VALUES IN results/v2_qwen1.5b    THREE DECIMAL PLACES",
        56,
        536,
        11,
        weight=560,
        fill=DIM,
        tracking=1.6,
    )
    return canvas.svg()


def architecture_svg() -> str:
    steps = int(PLANNER_SCHEMA["properties"]["steps"]["maxItems"])
    desc = (
        "Task to planner to worker to critic to the grounding gate to a JSONL trace. "
        f"Tools: {', '.join(TOOL_NAMES)}. "
        f"Limits: {MAX_TURNS} turns, {MAX_CRITIC_RETRIES} revise passes, {MAX_SCHEMA_TRIES} schema tries, "
        f"planner at most {steps} steps."
    )
    canvas = Canvas(1200, 680, "Orbit architecture", desc)
    canvas.grid()
    canvas.text("ARCHITECTURE", 56, 52, 12, weight=560, fill=DIM, tracking=2.4)
    canvas.text("One bus. Four checks.", 56, 96, 36, weight=640, tracking=-0.6)
    canvas.text(
        "JSON schema on every turn. Sandboxed tools. Critic revisions. A grounding gate.",
        56,
        132,
        16,
        weight=440,
        fill=MUTED,
    )

    nodes = (
        ("01", "Task", "the question", False),
        ("02", "Planner", f"at most {steps} steps", False),
        ("03", "Worker", "one call or a final", False),
        ("04", "Critic", "accept or revise", False),
        ("05", "Gate", "supported claim", True),
        ("06", "Trace", "JSONL", False),
    )
    box_w, box_h = 160, 88
    gap = 28
    x0, y0 = 48, 176
    centers = []
    for index, (num, name, note, accent) in enumerate(nodes):
        x = x0 + index * (box_w + gap)
        stroke = ACCENT if accent else LINE_STRONG
        canvas.rect(x, y0, box_w, box_h, stroke)
        canvas.text(
            num, x + box_w / 2, y0 + 28, 11, weight=560, fill=ACCENT if accent else DIM, tracking=1.4, anchor="middle"
        )
        canvas.text(
            name, x + box_w / 2, y0 + 52, 18, weight=560, fill=ACCENT if accent else INK, tracking=-0.2, anchor="middle"
        )
        canvas.text(note, x + box_w / 2, y0 + 74, 11, weight=440, fill=MUTED, anchor="middle")
        centers.append(x + box_w / 2)
        if index:
            previous = x0 + (index - 1) * (box_w + gap) + box_w
            canvas.arrow(previous + 6, x - 6, y0 + box_h / 2, ACCENT if accent else DIM)

    worker_x = x0 + 2 * (box_w + gap)
    critic_x = x0 + 3 * (box_w + gap)
    arc_y = y0 + box_h + 8
    start_x = critic_x + box_w / 2
    end_x = worker_x + box_w / 2
    control_y = arc_y + 36
    canvas.parts.append(
        f'<path d="M {start_x:.1f} {arc_y:.1f} Q {(start_x + end_x) / 2:.1f} {control_y:.1f} {end_x + 10:.1f} {arc_y:.1f}" '
        f'fill="none" stroke="{DIM}" stroke-width="1"/>'
    )
    canvas.arrow(end_x + 10, end_x - 4, arc_y, DIM)
    canvas.text(
        f"revise, at most {MAX_CRITIC_RETRIES}",
        (start_x + end_x) / 2,
        control_y + 22,
        12,
        weight=440,
        fill=MUTED,
        anchor="middle",
    )

    panel_y = 360
    canvas.rect(48, panel_y, 1104, 112, LINE)
    canvas.text("SANDBOXED TOOLS", 72, panel_y + 28, 11, weight=560, fill=DIM, tracking=1.8)
    chip_gap = 12
    chip_sizes = [(name, canvas.faces[560].width(name, 14, 0.2) + 28) for name in TOOL_NAMES]
    chip_row = sum(width for _name, width in chip_sizes) + chip_gap * (len(chip_sizes) - 1)
    chip_x = 48 + (1104 - chip_row) / 2
    for name, chip_w in chip_sizes:
        canvas.rect(chip_x, panel_y + 46, chip_w, 32, LINE_STRONG)
        canvas.text(name, chip_x + 14, panel_y + 68, 14, weight=560, fill=INK, tracking=0.2)
        chip_x += chip_w + chip_gap
    canvas.text(
        "web_search reads a bundled cache. No tool in this path opens a socket.",
        72,
        panel_y + 100,
        13,
        weight=440,
        fill=MUTED,
    )

    notes = (
        (
            56,
            "JSON SCHEMA",
            False,
            "Planner, worker, and critic",
            f"each return one object. {MAX_SCHEMA_TRIES} tries on a bad turn.",
        ),
        (448, "CRITIC", False, "Accept a grounded claim,", f"or send it back, at most {MAX_CRITIC_RETRIES} times."),
        (840, "GROUNDING GATE", True, "A cited result, a printed count,", "or a matching line count. Else refuse."),
    )
    for x, label, accent, line_one, line_two in notes:
        canvas.text(label, x, 516, 12, weight=560, fill=ACCENT if accent else DIM, tracking=1.6)
        canvas.text(line_one, x, 544, 14, weight=440, fill=MUTED)
        canvas.text(line_two, x, 566, 14, weight=440, fill=MUTED)
    canvas.rule(56, 1144, 604)
    canvas.text(
        f"{MAX_TURNS} TURNS     {MAX_CRITIC_RETRIES} REVISE PASSES     {MAX_SCHEMA_TRIES} SCHEMA TRIES     PLANNER AT MOST {steps} STEPS",
        56,
        640,
        12,
        weight=560,
        fill=DIM,
        tracking=1.4,
    )
    return canvas.svg()


def build(root: Path = ROOT) -> dict[Path, str]:
    return {HERO: hero_svg(root), ARCHITECTURE: architecture_svg()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write docs/hero.svg and docs/architecture.svg")
    parser.add_argument("--check", action="store_true", help="Fail if the committed SVGs are stale")
    args = parser.parse_args(argv)
    fresh = build()
    stale = [path for path, text in fresh.items() if not path.is_file() or path.read_text(encoding="utf-8") != text]
    if args.check:
        if stale:
            raise SystemExit("README figures are stale; run python docs/build_figures.py")
        print(json.dumps({"figures": "current"}))
        return
    for path, text in fresh.items():
        path.write_text(text, encoding="utf-8")
    print(json.dumps({"wrote": [str(path.relative_to(ROOT)) for path in fresh]}))


if __name__ == "__main__":
    main()
