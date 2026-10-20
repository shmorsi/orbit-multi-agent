"""JSON schemas for agent turns and tool arguments."""

from __future__ import annotations

import json

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
