"""JSON schemas for agent turns and tool arguments."""

from __future__ import annotations

import json
import re

import jsonschema

TOOL_NAMES = ["calculator", "read_file", "write_file", "python", "retrieve", "web_search"]

TOOL_SCHEMAS: dict[str, dict] = {
    "calculator": {
        "type": "object",
        "additionalProperties": False,
        "required": ["expression"],
        "properties": {"expression": {"type": "string", "minLength": 1, "maxLength": 80}},
    },
    "read_file": {
        "type": "object",
        "additionalProperties": False,
        "required": ["path"],
        "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 200}},
    },
    "write_file": {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "content"],
        "properties": {
            "path": {"type": "string", "minLength": 1, "maxLength": 80},
            "content": {"type": "string", "maxLength": 400},
        },
    },
    "python": {
        "type": "object",
        "additionalProperties": False,
        "required": ["code"],
        "properties": {"code": {"type": "string", "minLength": 1, "maxLength": 800}},
    },
    "retrieve": {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
    },
    "web_search": {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
    },
}

PLANNER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        "steps": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["goal", "tool"],
                "properties": {
                    "goal": {"type": "string", "minLength": 1, "maxLength": 160},
                    "tool": {"type": "string", "enum": TOOL_NAMES},
                },
            },
        }
    },
}

WORKER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action"],
    "properties": {
        "action": {"type": "string", "enum": ["tool", "final"]},
        "tool": {"type": "string", "enum": TOOL_NAMES},
        "arguments": {"type": "object"},
        "status": {"type": "string", "enum": ["answer", "refuse"]},
        "answer": {"type": "string", "maxLength": 600},
        "claims": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "citation"],
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 40},
                    "value": {"type": "integer"},
                    "text": {"type": "string", "maxLength": 200},
                    "unit": {"type": "string", "maxLength": 20},
                    "citation": {"type": "string", "pattern": "^call_[1-9][0-9]*$"},
                },
            },
        },
    },
}

CRITIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "revise"]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 300},
    },
}

ROLE_SCHEMAS = {"planner": PLANNER_SCHEMA, "worker": WORKER_SCHEMA, "critic": CRITIC_SCHEMA}


def validate(schema: dict, payload: dict) -> None:
    jsonschema.validate(payload, schema)


def _known(payload: dict, allowed: set[str]) -> dict:
    return {key: value for key, value in payload.items() if key in allowed}


def coerce_int(value):
    """Whole numbers written as 5.0 or \"5\" are the integer 5. Booleans are not numbers."""
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    return value


def normalize_citation(value):
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"(?i)call[\s_-]*([1-9]\d*)", value.strip())
    if match:
        return f"call_{int(match.group(1))}"
    return value


def normalize_payload(role: str, payload: dict) -> dict:
    """Drop unknown keys and coerce claim spellings. Required fields are still validated later."""
    if not isinstance(payload, dict):
        return payload
    if role == "planner":
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            return _known(payload, {"steps"})
        steps = []
        for step in raw_steps:
            if not isinstance(step, dict):
                steps.append(step)
                continue
            cleaned = _known(step, {"goal", "tool"})
            tool = cleaned.get("tool")
            if isinstance(tool, str):
                cleaned["tool"] = tool.strip().lower()
            steps.append(cleaned)
        return {"steps": steps}
    if role == "critic":
        return _known(payload, {"verdict", "reason"})
    if role != "worker":
        return payload
    cleaned = _known(payload, {"action", "tool", "arguments", "status", "answer", "claims"})
    tool_name = cleaned.get("tool")
    if isinstance(tool_name, str):
        cleaned["tool"] = tool_name.strip().lower()
    arguments = cleaned.get("arguments")
    known_tool = cleaned.get("tool")
    if isinstance(arguments, dict) and known_tool in TOOL_SCHEMAS:
        allowed = set(TOOL_SCHEMAS[known_tool]["properties"])
        cleaned["arguments"] = _known(arguments, allowed)
    claims = []
    for claim in cleaned.get("claims") or []:
        if not isinstance(claim, dict):
            claims.append(claim)
            continue
        item = _known(claim, {"name", "value", "text", "unit", "citation"})
        if "value" in item:
            item["value"] = coerce_int(item["value"])
        if "citation" in item:
            item["citation"] = normalize_citation(item["citation"])
        claims.append(item)
    if "claims" in cleaned:
        cleaned["claims"] = claims
    return cleaned


def check_worker(payload: dict) -> str | None:
    try:
        validate(WORKER_SCHEMA, payload)
    except jsonschema.ValidationError as exc:
        return exc.message
    if payload["action"] == "tool":
        tool = payload.get("tool")
        arguments = payload.get("arguments")
        if tool not in TOOL_SCHEMAS or not isinstance(arguments, dict):
            return "tool action needs a known tool and arguments"
        try:
            validate(TOOL_SCHEMAS[tool], arguments)
        except jsonschema.ValidationError as exc:
            return exc.message
        return None
    if payload.get("status") not in {"answer", "refuse"} or "answer" not in payload or "claims" not in payload:
        return "final action needs status, answer, and claims"
    if payload["status"] == "refuse" and payload["claims"]:
        return "a refusal carries no claims"
    for claim in payload["claims"]:
        has_value = "value" in claim
        has_text = "text" in claim
        if has_value == has_text:
            return "a claim needs a value or a text, not both"
    return None


def check_role(role: str, payload: dict) -> str | None:
    if role == "worker":
        return check_worker(payload)
    try:
        validate(ROLE_SCHEMAS[role], payload)
    except jsonschema.ValidationError as exc:
        return exc.message
    return None


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
