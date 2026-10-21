import json
import threading
import urllib.request
from pathlib import Path

from orbit_agents.backends import (
    LlamaCppBackend,
    OllamaBackend,
    eval_loads_in_4bit,
    inference_dtype,
    pick_device,
    torchao_is_compatible,
)
from orbit_agents.doctor import main as doctor_main
from orbit_agents.serve import bind


def test_cuda_eval_stays_4bit_and_other_devices_do_not():
    assert eval_loads_in_4bit("cuda") is True
    assert eval_loads_in_4bit("mps") is False
    assert eval_loads_in_4bit("cpu") is False


def test_old_torchao_is_rejected():
    assert torchao_is_compatible(None) is True
    assert torchao_is_compatible("0.10.0") is False
    assert torchao_is_compatible("0.16.0") is False
    assert torchao_is_compatible("0.16.1") is True


def test_pick_device_prefers_mps_then_cpu():
    class Cuda:
        def is_available(self):
            return False

    class MPS:
        def is_available(self):
            return True

    class Backends:
        mps = MPS()

    class Torch:
        cuda = Cuda()
        backends = Backends()
        float32 = "float32"
        float16 = "float16"

    assert pick_device(None) == "cpu"
    assert pick_device(Torch()) == "mps"
    assert inference_dtype("cpu", Torch()) == "float32"
    assert inference_dtype("mps", Torch()) == "float16"


def test_local_model_hosts_stay_on_localhost():
    OllamaBackend("demo", host="http://127.0.0.1:9")
    LlamaCppBackend("demo", base_url="http://localhost:9/v1")
    try:
        OllamaBackend("demo", host="http://example.com")
    except ValueError:
        pass
    else:
        raise AssertionError("ollama accepted a remote host")


def test_inference_requirements_skip_bitsandbytes():
    text = Path("requirements-infer.txt").read_text(encoding="utf-8").lower()
    assert "bitsandbytes==" not in text
    assert "torch==" in text
    for relative in ("src/orbit_agents/backends.py", "src/orbit_agents/serve.py", "src/orbit_agents/doctor.py", "src/orbit_agents/engine.py"):
        assert "import bitsandbytes" not in Path(relative).read_text(encoding="utf-8")


def test_doctor_scripted_check(capsys):
    doctor_main([])
    report = json.loads(capsys.readouterr().out)
    assert report["environment"]["python_ok"] is True
    assert report["environment"]["device"] in {"cpu", "mps", "cuda"}
    assert report["scripted"]["ok"] is True
    assert report["scripted"]["kind"] == "scripted_harness"
    assert report["live"] is None


def test_live_scripted_task_streams_a_real_trace():
    server = bind(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/run",
            data=json.dumps({"backend": "scripted", "task_id": "dev-001"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            rows = [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/mission/", timeout=10)
        html = page.read().decode()
        free = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/run",
            data=json.dumps({"backend": "scripted", "question": "What is the weather?"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(free, timeout=10) as response:
            refused = [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
    finally:
        server.shutdown()
    assert any(row.get("kind") == "result" and row.get("agent") == "tool" for row in rows)
    mission = rows[-1]
    assert mission["kind"] == "mission"
    assert mission["status"] == "answer"
    assert mission["source"] == "orbit_agents.engine"
    assert "Play" in html
    assert refused == [{"kind": "error", "error": "the scripted harness only runs a bundled task id"}]
