"""Record Mission Control replaying a real engine trace.

Writes docs/mission-control.mp4 and docs/mission-control.gif.
The page is the local server, the trace is mission/traces/revise.jsonl,
and the frames come from one headless Chrome session.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit_agents.serve import bind  # noqa: E402

CHROME_CANDIDATES = (
    os.environ.get("CHROME_BIN"),
    "/usr/local/bin/google-chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def _chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    found = shutil.which("google-chrome") or shutil.which("chromium")
    if not found:
        raise SystemExit("Chrome is not installed")
    return found


class SocketCDP:
    def __init__(self, ws_url: str):
        parsed = urlparse(ws_url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=20)
        key = base64.b64encode(os.urandom(16)).decode()
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self.sock.sendall(request.encode())
        data = b""
        while b"\r\n\r\n" not in data:
            data += self.sock.recv(4096)
        status = data.split(b"\r\n", 1)[0]
        if b" 101 " not in status:
            raise RuntimeError(status.decode())
        self.buffer = data.split(b"\r\n\r\n", 1)[1]
        self.next_id = 0

    def _send_frame(self, payload: bytes, opcode: int = 0x1) -> None:
        mask = os.urandom(4)
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack(">H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack(">Q", length))
        header.extend(mask)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def read_message(self) -> bytes:
        chunks = []
        while True:
            def take(count: int) -> bytes:
                while len(self.buffer) < count:
                    incoming = self.sock.recv(1 << 20)
                    if not incoming:
                        raise RuntimeError("devtools socket closed")
                    self.buffer += incoming
                out, self.buffer = self.buffer[:count], self.buffer[count:]
                return out

            first, second = take(2)
            opcode = first & 0x0F
            length = second & 0x7F
            if length == 126:
                length = struct.unpack(">H", take(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", take(8))[0]
            if second & 0x80:
                mask = take(4)
                data = bytes(byte ^ mask[index % 4] for index, byte in enumerate(take(length)))
            else:
                data = take(length)
            if opcode == 0x8:
                raise RuntimeError("devtools closed the socket")
            if opcode == 0x9:
                self._send_frame(data, opcode=0xA)
                continue
            if opcode in {0x0, 0x1, 0x2}:
                chunks.append(data)
                if first & 0x80:
                    return b"".join(chunks)
            elif first & 0x80:
                return b"".join(chunks)

    def call(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        self.next_id += 1
        ident = self.next_id
        body = {"id": ident, "method": method}
        if params:
            body["params"] = params
        self._send_frame(json.dumps(body).encode())
        self.sock.settimeout(timeout)
        while True:
            message = json.loads(self.read_message())
            if message.get("id") != ident:
                continue
            if "error" in message:
                raise RuntimeError(json.dumps(message["error"]))
            return message.get("result") or {}


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode())


def _event_count(trace: Path) -> int:
    count = 0
    for line in trace.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("kind") != "mission":
            count += 1
    return count


def _evaluate(cdp: SocketCDP, expression: str):
    result = cdp.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
    return result.get("result", {}).get("value")


def record(
    trace_name: str,
    also_artifacts: Path | None,
    *,
    mp4_name: str = "mission-control.mp4",
    gif_name: str = "mission-control.gif",
    step_ms: int = 2200,
    title_ms: int = 4000,
    hold_s: float = 8,
    loose: bool = False,
    width: int = 1280,
    height: int = 800,
    gif_scale: int = 800,
    gif_fps: int = 8,
    capture_every: float = 0.5,
    device_scale: int = 1,
) -> None:
    trace = ROOT / "mission" / "traces" / trace_name
    events = _event_count(trace)
    duration = title_ms / 1000 + (max(events - 1, 1) * step_ms / 1000) + hold_s
    if not loose and not 30 <= duration <= 60:
        raise SystemExit(f"recording length {duration:.1f}s is outside 30-60s")
    server = bind(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    page = (
        f"http://127.0.0.1:{port}/mission/index.html"
        f"?trace={trace_name}&autoplay=1&stepms={step_ms}&titlems={title_ms}&record=1"
    )
    debug = 9333
    profile = Path("/tmp/orbit_agents-chrome-profile")
    shutil.rmtree(profile, ignore_errors=True)
    profile.mkdir(parents=True)
    chrome_log = Path("/tmp/orbit_agents-chrome.log")
    chrome = subprocess.Popen(
        [
            _chrome(),
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--hide-scrollbars",
            "--remote-allow-origins=*",
            f"--remote-debugging-port={debug}",
            f"--user-data-dir={profile}",
            f"--window-size={width},{height}",
            "about:blank",
        ],
        stdout=chrome_log.open("w"),
        stderr=subprocess.STDOUT,
    )
    frames = ROOT / "demo" / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)
    try:
        version = None
        for _ in range(40):
            try:
                version = _get(f"http://127.0.0.1:{debug}/json/version")
                break
            except Exception:
                time.sleep(0.25)
        if version is None:
            raise SystemExit("Chrome did not open a debugging port")
        browser = SocketCDP(version["webSocketDebuggerUrl"])
        created = browser.call("Target.createTarget", {"url": page})
        target_id = created["targetId"]
        listed = None
        for _ in range(40):
            targets = _get(f"http://127.0.0.1:{debug}/json/list")
            listed = next((item for item in targets if item.get("id") == target_id), None)
            if listed and listed.get("webSocketDebuggerUrl"):
                break
            time.sleep(0.25)
        if not listed:
            raise SystemExit("Chrome did not expose the page socket")
        page_cdp = SocketCDP(listed["webSocketDebuggerUrl"])
        page_cdp.call("Page.enable")
        page_cdp.call("Runtime.enable")
        page_cdp.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": height, "deviceScaleFactor": device_scale, "mobile": False},
        )
        deadline = time.monotonic() + 15
        ready = ""
        while time.monotonic() < deadline:
            ready = _evaluate(page_cdp, "document.getElementById('question') && document.getElementById('question').textContent") or ""
            if ready and not ready.startswith("Loading"):
                break
            time.sleep(0.2)
        if not ready or ready.startswith("Loading"):
            raise SystemExit(f"replay page did not load a trace: {ready!r}")
        started = time.monotonic()
        index = 0
        while time.monotonic() - started < duration:
            shot = page_cdp.call("Page.captureScreenshot", {"format": "png"})
            index += 1
            (frames / f"frame_{index:04d}.png").write_bytes(base64.b64decode(shot["data"]))
            elapsed = time.monotonic() - started
            sleep_for = (index * capture_every) - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
        badge = _evaluate(page_cdp, "document.getElementById('badge').textContent")
        log_rows = _evaluate(page_cdp, "document.querySelectorAll('#log li').length")
        if badge != "Mission success":
            raise SystemExit(f"replay ended on {badge!r} with {log_rows} log rows")
        if not log_rows or log_rows < events:
            raise SystemExit(f"log showed {log_rows} rows, expected {events}")
    finally:
        server.shutdown()
        chrome.terminate()
        try:
            chrome.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome.kill()
    if index < 2:
        raise SystemExit("recorder captured fewer than two frames")
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    mp4 = docs / mp4_name
    gif = docs / gif_name
    rate = 1 / capture_every
    rate_arg = str(int(rate)) if rate == int(rate) else f"{rate:.4f}"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            rate_arg,
            "-i",
            str(frames / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4),
        ]
    )
    palette = frames / "palette.png"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4),
            "-vf",
            f"fps={gif_fps},scale={gif_scale}:-1:flags=lanczos,palettegen",
            str(palette),
        ]
    )
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4),
            "-i",
            str(palette),
            "-lavfi",
            f"fps={gif_fps},scale={gif_scale}:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(gif),
        ]
    )
    probed = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(mp4),
        ],
        text=True,
    ).strip()
    length = float(probed)
    if not loose and not 30 <= length <= 60:
        raise SystemExit(f"mp4 duration {length:.2f}s is outside 30-60s")
    first = (frames / "frame_0001.png").read_bytes()
    last = (frames / f"frame_{index:04d}.png").read_bytes()
    if first == last:
        raise SystemExit("first and last frames are identical")
    if also_artifacts is not None:
        also_artifacts.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mp4, also_artifacts)
    print(json.dumps({"mp4": str(mp4), "gif": str(gif), "seconds": round(length, 2), "frames": index, "badge": "Mission success"}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the Mission Control replay")
    parser.add_argument("--trace", default="revise.jsonl")
    parser.add_argument("--artifacts", type=Path, default=None, help="Optional extra copy of the mp4")
    parser.add_argument("--mp4", default="mission-control.mp4")
    parser.add_argument("--gif", default="mission-control.gif")
    parser.add_argument("--step-ms", type=int, default=2200)
    parser.add_argument("--title-ms", type=int, default=4000)
    parser.add_argument("--hold", type=float, default=8)
    parser.add_argument("--loose", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--gif-scale", type=int, default=800)
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--capture-every", type=float, default=0.5)
    parser.add_argument("--device-scale", type=int, default=1)
    args = parser.parse_args()
    record(
        args.trace,
        args.artifacts,
        mp4_name=args.mp4,
        gif_name=args.gif,
        step_ms=args.step_ms,
        title_ms=args.title_ms,
        hold_s=args.hold,
        loose=args.loose,
        width=args.width,
        height=args.height,
        gif_scale=args.gif_scale,
        gif_fps=args.gif_fps,
        capture_every=args.capture_every,
        device_scale=args.device_scale,
    )


if __name__ == "__main__":
    main()
