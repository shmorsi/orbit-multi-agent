import json
import subprocess
import sys
from pathlib import Path

from leakage import check
from train import map_training_kwargs


def test_leakage_passes():
    report = check()
    assert report["ok"] is True
    manifest = json.loads(Path("finetune/data/manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_file_read"] is False
    assert manifest["built_from"] == "eval/dev.jsonl only"


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


def test_notebook_trains_the_small_config_once():
    notebook = json.loads(Path("finetune/colab_train.ipynb").read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        source = cell["source"]
        sources.append("".join(source) if isinstance(source, list) else source)
    blob = "\n".join(sources)
    assert "/content/orbit.zip" in blob
    assert "qlora_small.json" in blob
    assert "only-if-needed" in blob
    assert "--no-deps" in blob
    assert "/content/outputs.zip" in blob
    assert "Do not score the frozen test in a loop" in blob
    assert "qlora_medium" not in blob
