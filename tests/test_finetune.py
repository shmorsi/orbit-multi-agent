import json
import subprocess
import sys
from pathlib import Path

import ast

import pytest

from leakage import BUILT_FROM, check, overlap_error
from train import guard_seq_length, map_training_kwargs, mask_assistant_tokens, seq_length_error


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


class _WordTokenizer:
    """One token per whitespace-separated word, with character offsets."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages
        )

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        ids = []
        offsets = []
        index = 0
        while index < len(text):
            if text[index].isspace():
                index += 1
                continue
            end = index
            while end < len(text) and not text[end].isspace():
                end += 1
            ids.append(text[index:end])
            offsets.append((index, end))
            index = end
        encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded


def _dialog(system: str, user: str, assistant: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def test_seq_length_error_names_the_dialogs_that_would_lose_the_assistant_turn():
    # Distill rows measure about 951 median and about 1,400 max. 768 cuts them.
    message = seq_length_error([700, 951, 1400], 768)
    assert message is not None
    assert "2 of 3" in message
    assert "max_seq_length 768" in message
    assert "longest 1400" in message
    assert "assistant mask is empty" in message
    assert "at least 1400" in message
    assert seq_length_error([700, 951, 1400], 1536) is None
    assert seq_length_error([], 768) is None


def test_distill_config_fits_the_measured_dialogs():
    config = json.loads(Path("finetune/configs/qlora_distill.json").read_text(encoding="utf-8"))
    assert config["max_seq_length"] >= 1536
    assert seq_length_error([951, 1400], config["max_seq_length"]) is None


def test_guard_seq_length_rejects_a_cap_shorter_than_the_dialogs():
    tokenizer = _WordTokenizer()
    system = " ".join(["rule"] * 30)
    rows = [{"messages": _dialog(system, "count the jobs", "{\"action\": \"final\"}")}]
    with pytest.raises(SystemExit, match="max_seq_length 8") as caught:
        guard_seq_length(tokenizer, rows, 8)
    assert "assistant mask is empty" in str(caught.value)
    guard_seq_length(tokenizer, rows, 1536)


def test_mask_assistant_tokens_refuses_to_truncate_the_assistant_turn():
    tokenizer = _WordTokenizer()
    messages = _dialog(" ".join(["rule"] * 20), "count the jobs", "job_count equals 5.")
    with pytest.raises(RuntimeError, match="assistant mask is empty") as caught:
        mask_assistant_tokens(tokenizer, messages, 8)
    assert "max_seq_length is 8" in str(caught.value)
    masked = mask_assistant_tokens(tokenizer, messages, 1536)
    assert any(label != -100 for label in masked["labels"])
    assert masked["labels"].count(-100) < len(masked["labels"])
    assert "job_count" in masked["input_ids"]


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
