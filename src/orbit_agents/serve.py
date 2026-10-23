"""Local mission-control server. It streams engine events and never invents a trace."""

from __future__ import annotations

import json
import queue
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from orbit_agents.engine import run_team
from orbit_agents.paths import ROOT
from orbit_agents.tasks import ad_hoc_task, task_by_id

DEFAULT_PORT = 8741
MAX_QUESTION = 500
_LOCK = threading.Lock()


def resolve_task(payload: dict) -> tuple[dict | None, str | None]:
    backend = payload.get("backend", "scripted")
    if backend not in {"scripted", "transformers", "ollama", "llamacpp"}:
        return None, "unknown backend"
    task_id = payload.get("task_id")
    if task_id:
        task = task_by_id(str(task_id))
        if task is None:
            return None, "unknown task id"
        return task, None
    question = str(payload.get("question", "")).strip()
    if not question or len(question) > MAX_QUESTION:
        return None, "question must be present and short"
    if backend == "scripted":
        return None, "the scripted harness only runs a bundled task id"
    return ad_hoc_task(question), None


def make_backend(name: str, task: dict, model: str, adapter: str | None):
    if name == "scripted":
        from orbit_agents.scripted import ScriptedBackend

        return ScriptedBackend(task)
    if name == "transformers":
        from orbit_agents.backends import TransformersBackend

        return TransformersBackend(model, adapter=adapter)
    if name == "ollama":
        from orbit_agents.backends import OllamaBackend

        return OllamaBackend(model)
    from orbit_agents.backends import LlamaCppBackend

    return LlamaCppBackend(model)


def stream_events(payload: dict):
    task, error = resolve_task(payload)
    if error:
        yield json.dumps({"kind": "error", "error": error}) + "\n"
        return
    backend_name = payload.get("backend", "scripted")
    model = payload.get("model") or "Qwen/Qwen2.5-0.5B-Instruct"
    adapter = payload.get("adapter")
    events: queue.Queue = queue.Queue()

    def on_event(event: dict) -> None:
        events.put(event)

    def work() -> None:
        try:
            if not _LOCK.acquire(blocking=False):
                events.put({"kind": "error", "error": "another run is already using this machine"})
                return
            try:
                backend = make_backend(backend_name, task, model, adapter)
                workspace = ROOT / "traces" / "live" / str(task["id"])
                result = run_team(task, backend, workspace, on_event=on_event)
            finally:
                _LOCK.release()
            from orbit_agents.trace import mission_header

            harness = "scripted_harness" if backend_name == "scripted" else backend_name
            events.put(mission_header(task, result, harness=harness))
        except Exception as exc:
            events.put({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = events.get()
        if item is None:
            break
        yield json.dumps(item, ensure_ascii=True) + "\n"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/.git") or "/.git/" in path or ".." in path:
            self.send_error(404)
            return
        if path in {"/", "/mission"}:
            self.path = "/mission/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            payload = {}
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        for line in stream_events(payload):
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()


def bind(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(port: int = DEFAULT_PORT) -> None:
    server = bind(port)
    print(f"Orbit: Multi-Agent Mission Control at http://127.0.0.1:{server.server_address[1]}/mission/")
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve mission control and live runs")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    serve(args.port)


if __name__ == "__main__":
    main()
