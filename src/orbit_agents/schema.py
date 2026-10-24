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
    """Raise when the payload does not match the role schema."""
    jsonschema.validate(payload, schema)


def _known(payload: dict, allowed: set[str]) -> dict:
    return {key: value for key, value in payload.items() if key in allowed}


def coerce_int(value: object) -> object:
    """Whole numbers written as 5.0 or \"5\" are the integer 5. Booleans are not numbers."""
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    return value


def normalize_citation(value: object) -> object:
    """Rewrite spellings like 'call 1' to call_1. Other values pass through."""
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
    """Return an error string when a worker payload breaks a role rule, else None."""
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
    """Return an error string when the payload breaks that role's rules, else None."""
    if role == "worker":
        return check_worker(payload)
    try:
        validate(ROLE_SCHEMAS[role], payload)
    except jsonschema.ValidationError as exc:
        return exc.message
    return None


def dumps(payload: dict) -> str:
    """Serialize one payload the way the trace stores it."""
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Team v2: six roles. Each role's tool permissions are explicit and minimal.
# The v1 schemas above are unchanged.
# ---------------------------------------------------------------------------

ROLES_V2 = ["commander", "scout", "engineer", "analyst", "critic", "auditor"]

ROLE_TOOLS: dict[str, tuple[str, ...]] = {
    "commander": (),
    "scout": ("retrieve", "read_file", "web_search"),
    "engineer": ("python", "write_file"),
    "analyst": ("calculator",),
    "critic": (),
    "auditor": (),
}

# Roles the commander can assign a step to.
SPECIALISTS = ("scout", "engineer", "analyst")
# Only the analyst, in its drafting turn, writes the final answer.
ANSWERING_ROLES = ("analyst",)

COMMANDER_SCHEMA = {
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
                "required": ["goal", "role", "tool"],
                "properties": {
                    "goal": {"type": "string", "minLength": 1, "maxLength": 160},
                    "role": {"type": "string", "enum": list(SPECIALISTS)},
                    "tool": {"type": "string", "enum": TOOL_NAMES},
                },
            },
        }
    },
}


def role_schema_v2(role: str, *, may_answer: bool = False) -> dict:
    """The JSON schema for one v2 turn. A specialist step is a tool call from that role's tools only."""
    if role == "commander":
        return COMMANDER_SCHEMA
    if role == "critic":
        return CRITIC_SCHEMA
    if role not in SPECIALISTS:
        raise ValueError(f"{role} does not take a model turn")
    schema = json.loads(json.dumps(WORKER_SCHEMA))
    schema["properties"]["tool"]["enum"] = list(ROLE_TOOLS[role])
    if not (may_answer and role in ANSWERING_ROLES):
        schema["properties"]["action"]["enum"] = ["tool"]
    return schema


def normalize_payload_v2(role: str, payload: dict) -> dict:
    """Drop unknown keys and coerce claim spellings for a v2 role."""
    if not isinstance(payload, dict):
        return payload
    if role == "commander":
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            return _known(payload, {"steps"})
        steps = []
        for step in raw_steps:
            if not isinstance(step, dict):
                steps.append(step)
                continue
            cleaned = _known(step, {"goal", "role", "tool"})
            for key in ("role", "tool"):
                if isinstance(cleaned.get(key), str):
                    cleaned[key] = cleaned[key].strip().lower()
            steps.append(cleaned)
        return {"steps": steps}
    if role == "critic":
        return normalize_payload("critic", payload)
    return normalize_payload("worker", payload)


def _permission_error(role: str, payload: dict, may_answer: bool) -> str | None:
    action = payload.get("action")
    if action == "final" and not (may_answer and role in ANSWERING_ROLES):
        if role in ANSWERING_ROLES:
            return f"{role} runs a tool on an assigned step; the final answer comes in the drafting turn"
        return f"{role} cannot give the final answer; reply with one tool call"
    tool = payload.get("tool")
    if action == "tool" and isinstance(tool, str) and tool not in ROLE_TOOLS[role]:
        allowed = ", ".join(ROLE_TOOLS[role])
        return f"{role} may use only {allowed}; {tool} is not allowed"
    return None


def check_role_v2(role: str, payload: dict, *, may_answer: bool = False) -> str | None:
    """Validate a v2 turn. Permission errors are reported in words so the repair turn can fix them."""
    if role == "commander":
        try:
            validate(COMMANDER_SCHEMA, payload)
        except jsonschema.ValidationError as exc:
            return exc.message
        for step in payload["steps"]:
            if step["tool"] not in ROLE_TOOLS[step["role"]]:
                allowed = ", ".join(ROLE_TOOLS[step["role"]])
                return f"step {step['goal']!r}: {step['role']} may use only {allowed}"
        return None
    if role == "critic":
        return check_role("critic", payload)
    if role not in SPECIALISTS:
        return f"{role} does not take a model turn"
    if isinstance(payload, dict):
        denied = _permission_error(role, payload, may_answer)
        if denied:
            return denied
    try:
        validate(role_schema_v2(role, may_answer=may_answer), payload)
    except jsonschema.ValidationError as exc:
        return exc.message
    return check_worker(payload)


# ---------------------------------------------------------------------------
# Team v3 round table. The v1 and v2 schemas above are unchanged.
# Researcher may only read. Red-team and verifier have no tools.
# ---------------------------------------------------------------------------

ROLES_V3 = [
    "coordinator",
    "commander",
    "researcher",
    "scout",
    "engineer",
    "analyst",
    "critic",
    "redteam",
    "verifier",
    "auditor",
]

ROLE_TOOLS_V3: dict[str, tuple[str, ...]] = {
    **ROLE_TOOLS,
    "coordinator": (),
    "researcher": ("retrieve", "read_file", "web_search"),
    "redteam": (),
    "verifier": (),
}

SPECIALISTS_V3 = ("researcher", "scout", "engineer", "analyst")

COORDINATOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["brief"],
    "properties": {"brief": {"type": "string", "minLength": 1, "maxLength": 400}},
}

REDTEAM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claim", "why"],
    "properties": {
        "why": {"type": "string", "minLength": 1, "maxLength": 400},
        "claim": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "value", "unit", "citation"],
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 80},
                "value": {"type": "integer"},
                "unit": {"type": "string", "minLength": 1, "maxLength": 40},
                "citation": {"type": "string", "minLength": 1, "maxLength": 40},
            },
        },
    },
}

VERIFIER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"type": "string", "enum": ["reject", "uphold"]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 400},
    },
}

COMMANDER_SCHEMA_V3 = json.loads(json.dumps(COMMANDER_SCHEMA))
COMMANDER_SCHEMA_V3["properties"]["steps"]["items"]["properties"]["role"]["enum"] = list(SPECIALISTS_V3)


def role_schema_v3(role: str, *, may_answer: bool = False) -> dict:
    """JSON schema for one round-table turn. v2 roles keep their v2 schemas."""
    if role == "coordinator":
        return COORDINATOR_SCHEMA
    if role == "redteam":
        return REDTEAM_SCHEMA
    if role == "verifier":
        return VERIFIER_SCHEMA
    if role == "commander":
        return COMMANDER_SCHEMA_V3
    if role == "researcher":
        schema = json.loads(json.dumps(WORKER_SCHEMA))
        schema["properties"]["tool"]["enum"] = list(ROLE_TOOLS_V3["researcher"])
        schema["properties"]["action"]["enum"] = ["tool"]
        return schema
    return role_schema_v2(role, may_answer=may_answer)


def check_role_v3(role: str, payload: dict, *, may_answer: bool = False) -> str | None:
    """Validate a v3 turn. A tool outside the role's list is a validation error."""
    if role == "commander":
        try:
            validate(COMMANDER_SCHEMA_V3, payload)
        except jsonschema.ValidationError as exc:
            return exc.message
        for step in payload["steps"]:
            if step["tool"] not in ROLE_TOOLS_V3[step["role"]]:
                allowed = ", ".join(ROLE_TOOLS_V3[step["role"]])
                return f"step {step['goal']!r}: {step['role']} may use only {allowed}"
        return None
    if role in {"coordinator", "redteam", "verifier"}:
        try:
            validate(role_schema_v3(role), payload)
        except jsonschema.ValidationError as exc:
            return exc.message
        return None
    if role == "researcher":
        if isinstance(payload, dict):
            denied = _permission_error_v3(role, payload)
            if denied:
                return denied
        try:
            validate(role_schema_v3(role), payload)
        except jsonschema.ValidationError as exc:
            return exc.message
        return check_worker(payload)
    return check_role_v2(role, payload, may_answer=may_answer)


def _permission_error_v3(role: str, payload: dict) -> str | None:
    tool = payload.get("tool")
    if payload.get("action") == "tool" and isinstance(tool, str) and tool not in ROLE_TOOLS_V3[role]:
        allowed = ", ".join(ROLE_TOOLS_V3[role])
        return f"{role} may use only {allowed}; {tool} is not allowed"
    if payload.get("action") == "final":
        return f"{role} cannot give the final answer"
    return None
