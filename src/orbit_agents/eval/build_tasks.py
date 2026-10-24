"""Turn the catalog into dev and test JSONL. Gold values come from executing the tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from orbit_agents.corpus import write_manifest
from orbit_agents.engine import claims_are_grounded, contains_number
from orbit_agents.eval.catalog import CATALOG, REFUSAL
from orbit_agents.eval.checks import evaluate, render_python
from orbit_agents.paths import ROOT
from orbit_agents.prompts import prompt_blob
from orbit_agents.tools import call_tool

DEV_PATH = ROOT / "eval" / "dev.jsonl"
TEST_PATH = ROOT / "eval" / "test.jsonl"
MANIFEST_PATH = ROOT / "eval" / "manifest.json"
STATS_PATH = ROOT / "data" / "stats.json"
TUNE_IDS = {task["id"] for task in CATALOG if task["subset"] == "tune"}


def _signature(call: dict) -> str:
    return call["tool"] + ":" + json.dumps(call["arguments"], sort_keys=True, ensure_ascii=True)


def _answer(claims: list[dict]) -> str:
    parts = []
    for claim in claims:
        if "value" in claim:
            parts.append(f"{claim['name']} is {claim['value']}.")
        else:
            parts.append(f"{claim['name']} is {claim['text']}.")
    return " ".join(parts)


def expand(task: dict, workspace: Path) -> dict:
    """Fill a task template with the values the checker and the scripted worker share."""
    split = "test" if task["subset"] == "test" else "dev"
    if task["refusal"]:
        return {
            "id": task["id"],
            "split": split,
            "subset": task["subset"],
            "question": task["question"],
            "expect_refusal": True,
            "plan": [],
            "oracle_calls": [],
            "gold_claims": [],
            "answer": REFUSAL,
            "status": "refuse",
        }
    calls = []
    claims = []
    plan = []
    values = []
    for index, step in enumerate(task["steps"]):
        kind = step["kind"]
        tool_name = {
            "python": "python",
            "retrieve": "retrieve",
            "read_file": "read_file",
            "web_search": "web_search",
            "calculator": "calculator",
            "write_file": "write_file",
        }[kind]
        plan.append({"goal": step["goal"], "tool": tool_name})
        if kind == "python":
            code = f"# {task['id']} step {index}\n" + render_python(step["spec"])
            calls.append({"tool": "python", "arguments": {"code": code}})
            values.append(evaluate(step["spec"]))
        elif kind == "retrieve":
            calls.append({"tool": "retrieve", "arguments": {"query": step["query"]}})
            values.append(evaluate({"op": "doc_contains", "file": step["file"], "text": step["text"]}))
        elif kind == "read_file":
            calls.append({"tool": "read_file", "arguments": {"path": step["path"]}})
            values.append(evaluate({"op": "doc_contains", "file": step["path"], "text": step["text"]}))
        elif kind == "web_search":
            calls.append({"tool": "web_search", "arguments": {"query": step["query"]}})
            values.append(step["text"])
        elif kind == "calculator":
            left = int(values[step["left"]])
            right = int(values[step["right"]])
            if right == 0:
                raise RuntimeError(f"{task['id']} divides by zero")
            calls.append({"tool": "calculator", "arguments": {"expression": f"{left}//{right}"}})
            values.append(left // right)
        elif kind == "write_file":
            content = step["template"].format(value=values[step["source"]])
            calls.append({"tool": "write_file", "arguments": {"path": step["path"], "content": content}})
            values.append(content)
        else:
            raise ValueError(kind)
        if step.get("claim"):
            value = values[-1]
            claim = {"name": step["claim"], "citation": f"call_{index + 1}"}
            if isinstance(value, int):
                claim["value"] = value
                claim["unit"] = "count"
            else:
                claim["text"] = value
            claims.append(claim)
    results = []
    for call in calls:
        result = call_tool(call["tool"], call["arguments"], workspace)
        if not result.get("ok", False) and call["tool"] != "web_search":
            raise RuntimeError(f"{task['id']} tool {call['tool']} failed: {result}")
        results.append({"tool": call["tool"], "arguments": call["arguments"], "result": result})
    record = {
        "id": task["id"],
        "split": split,
        "subset": task["subset"],
        "question": task["question"],
        "expect_refusal": False,
        "plan": plan,
        "oracle_calls": calls,
        "gold_claims": claims,
        "answer": _answer(claims),
        "status": "answer",
    }
    if not claims_are_grounded(record["question"], claims, results, record["answer"]):
        raise RuntimeError(f"{task['id']} gold is not grounded in the tool results")
    for claim in claims:
        if "value" in claim:
            blob = json.dumps(results[int(claim["citation"].split("_")[1]) - 1]["result"])
            if not contains_number(blob, claim["value"]):
                raise RuntimeError(f"{task['id']} missing {claim['value']} in {claim['citation']}")
    return record


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def build(workspace: Path | None = None) -> dict:
    """Rewrite the dev and test splits, the eval manifest, and the corpus manifest."""
    scratch = workspace or (ROOT / "traces" / "build")
    scratch.mkdir(parents=True, exist_ok=True)
    write_manifest()
    rows = [expand(task, scratch / task["id"]) for task in CATALOG]
    questions = [row["question"] for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate questions")
    signatures = [_signature(call) for row in rows for call in row["oracle_calls"]]
    if len(signatures) != len(set(signatures)):
        dupes = sorted({item for item in signatures if signatures.count(item) > 1})
        raise RuntimeError("duplicate tool signatures: " + " | ".join(dupes[:4]))
    dev = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    if len(dev) != 24 or len(test) != 12:
        raise RuntimeError(f"unexpected split sizes dev={len(dev)} test={len(test)}")
    _write_jsonl(DEV_PATH, dev)
    _write_jsonl(TEST_PATH, test)
    test_hash = hashlib.sha256(TEST_PATH.read_bytes()).hexdigest()
    prompt_hash = hashlib.sha256(prompt_blob().encode("utf-8")).hexdigest()
    traps = sum(1 for row in rows if row["expect_refusal"])
    stats = {
        "n_agents": 3,
        "n_tools": 6,
        "n_tasks": len(rows),
        "n_dev": len(dev),
        "n_train_tasks": sum(1 for row in dev if row["subset"] == "train"),
        "n_tune_tasks": sum(1 for row in dev if row["subset"] == "tune"),
        "n_test": len(test),
        "n_traps": traps,
        "max_turns": 8,
        "max_critic_retries": 2,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 0,
        "qwen_generation": 2.5,
        "small_parameter_billions": 0.5,
        "medium_parameter_billions": 1.5,
        "colab_gpu_t": 4,
        "python_min": 3.11,
        "demo_port": 8741,
        "m1_ram_gb": 8,
        "replay_clock_step_seconds": 4,
        "t4_vram_gb": 16,
        "colab_budget_minutes": 60,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "protocol": "team-eval-v1",
        "dev_file": "eval/dev.jsonl",
        "test_file": "eval/test.jsonl",
        "test_sha256": test_hash,
        "prompt_sha256": prompt_hash,
        "tune_ids": sorted(TUNE_IDS),
        "n_dev": len(dev),
        "n_test": len(test),
        "n_traps": traps,
        "models": {
            "qwen_generation": 2.5,
            "small_parameter_billions": 0.5,
            "medium_parameter_billions": 1.5,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Rebuild the frozen splits. Do not run this against a published checkout."""
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
