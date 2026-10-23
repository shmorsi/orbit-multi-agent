import json
import subprocess
import sys
from pathlib import Path

import ast

from leakage import BUILT_FROM, check, overlap_error
from train import map_training_kwargs


def test_leakage_passes():
    report = check()
    assert report["ok"] is True
    assert report["checks"] == ["ids", "questions", "near_duplicate_text", "answer_strings", "tool_snippets"]
    manifest = json.loads(Path("finetune/data/manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_file_read"] is False
    assert manifest["built_from"] == BUILT_FROM
    assert manifest["n_train_dialogs"] == report["n_train_dialogs"]
    assert 280 <= manifest["n_train_dialogs"] <= 340
    assert manifest["n_tune_dialogs"] >= 40
    assert "critic" in manifest["roles"]
    train = Path("finetune/data/train.jsonl").read_text(encoding="utf-8")
    tune = Path("finetune/data/tune.jsonl").read_text(encoding="utf-8")
    assert "dev-005" not in train
    assert "dev-017" not in train
    assert "dev-017" not in tune
    assert '"role": "critic"' in train
    assert '"role": "worker"' in train
    assert '"role": "planner"' in train


def test_leakage_rejects_near_duplicate_text_and_answer_strings():
    test_rows = [
        {
            "id": "test-x",
            "question": "How many printer-b jobs have outcome fail?",
            "answer": "fail_count is 9.",
            "oracle_calls": [{"tool": "python", "arguments": {"code": "# test-x step 0\nprint(9)"}}],
        }
    ]
    near = [
        {
            "id": "gen-1",
            "role": "worker",
            "messages": [
                {"role": "user", "content": "Task: How many printer-a jobs have outcome fail?"},
                {"role": "assistant", "content": "other"},
            ],
        }
    ]
    assert "near-duplicate" in overlap_error(near, [], test_rows)
    copied_answer = [
        {
            "id": "gen-2",
            "role": "worker",
            "messages": [
                {"role": "user", "content": "Task: From the jobs table, count a different slice of rows."},
                {"role": "assistant", "content": "fail_count is 9."},
            ],
        }
    ]
    assert "test answer leaked" in overlap_error(copied_answer, [], test_rows)
    copied_id = [
        {
            "id": "gen-3",
            "role": "planner",
            "messages": [{"role": "user", "content": "see test-x"}],
        }
    ]
    assert "test id leaked" in overlap_error(copied_id, [], test_rows)


def test_dry_run_does_not_claim_a_score():
    raw = subprocess.check_output(
        [sys.executable, "finetune/train.py", "--config", "finetune/configs/qlora_small.json", "--dry-run"]
    )
    payload = json.loads(raw)
    assert payload["dry_run"] is True
    assert payload["leakage_ok"] is True
    assert payload["test_file_read"] is False
    assert "task_success" not in payload


def test_warmup_ratio_maps_onto_the_transformers_5_signature():
    mapped = map_training_kwargs(
        {"warmup_ratio": 0.03, "eval_strategy": "epoch", "learning_rate": 0.0002},
        {"warmup_steps", "evaluation_strategy", "learning_rate"},
    )
    assert mapped["warmup_steps"] == 0.03
    assert mapped["evaluation_strategy"] == "epoch"
    assert "warmup_ratio" not in mapped
    assert "eval_strategy" not in mapped


def test_notebook_trains_the_medium_config_once():
    notebook = json.loads(Path("finetune/colab_train.ipynb").read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        source = cell["source"]
        text = "".join(source) if isinstance(source, list) else source
        sources.append(text)
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("orbit_agents")
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("orbit_agents")
    blob = "\n".join(sources)
    assert "/content/orbit.zip" in blob
    assert "qlora_medium.json" in blob
    assert "Qwen/Qwen2.5-1.5B-Instruct" in blob
    assert "uninstall" in blob and "torchao" in blob
    assert "only-if-needed" in blob
    assert "--no-deps" in blob
    assert "-q" not in blob
    assert "/content/orbit-checkpoints" in blob
    assert "/content/outputs.zip" in blob
    assert "Do not score the frozen test in a loop" in blob
    assert "--split tune" in blob
    assert "import orbit_agents, os" in blob
    assert "qlora_small.json" not in blob


def test_distill_notebook_runs_the_existing_path_once():
    notebook = json.loads(Path("notebooks/run_distill_and_score.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    sources = []
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        sources.append(text)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            ast.parse(text)
    blob = "\n".join(sources)
    assert "finetune/distill_teacher.py" in blob
    assert "finetune/configs/qlora_distill.json" in blob
    assert "orbit_agents.eval.run" in blob and "'--split', 'test'" in blob
    assert "orbit_agents.eval.compare" in blob
    assert "orbit_agents.eval.readme" in blob
    assert "results/v3_qwen1.5b" in blob
    assert "userdata.get(KEY_NAME)" in blob
    assert "ORBIT_ANTHROPIC_API_KEY" in blob
    assert "MAX_REQUESTS = 800" in blob
    assert "TEACHER_MODEL = 'claude-haiku-5-5'" in blob
    assert "test_sha256" in blob
    assert "already measured" in blob
    assert "Do not score the frozen test in a loop" in blob
    assert "print(_key" not in blob and "sk-ant" not in blob
