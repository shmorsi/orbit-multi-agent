"""Provider-agnostic policy layer.

Every team turn goes through one small interface:

    LLMClient.complete(messages, schema, role=...) -> dict

`messages` is a chat list ({"role": "system"|"user"|"assistant", "content": str}).
`schema` is the JSON schema the reply must satisfy. The bundled backends do not
constrain decoding with it; the engine validates every reply against the same
schema after the fact, so a local model and an API model are judged by one
validator. `role` names the team seat (planner, commander, scout, ...). Real
models ignore it; the scripted harness uses it to follow a task's oracle.

Adding a provider is one class: subclass LLMClient, implement `complete_text`,
and add it to PROVIDERS. See `OpenAICompatiblePolicy` at the bottom for the
documented extension point. Nothing in this module reads a key from disk.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod

from orbit_agents.engine import extract_json
from orbit_agents.paths import ROOT

ANTHROPIC_KEY_ENV = "ORBIT_ANTHROPIC_API_KEY"


class LLMClient(ABC):
    """One policy for every seat on the team. One call is one model request."""

    provider = "abstract"
    model: str | None = None

    @abstractmethod
    def complete_text(self, messages: list[dict], schema: dict, *, role: str) -> str:
        """Return the raw reply text. The engine keeps it in the trace."""

    def complete(self, messages: list[dict], schema: dict, *, role: str) -> dict:
        """Return the first JSON object in the reply. Raises ValueError when there is none."""
        return extract_json(self.complete_text(messages, schema, role=role))

    def usage(self) -> dict:
        return {}


class LegacyBackendPolicy(LLMClient):
    """Wrap a v1 backend (`complete(role=, messages=) -> str`) as an LLMClient."""

    def __init__(self, backend, *, provider: str, model: str | None = None):
        self.backend = backend
        self.provider = provider
        self.model = model

    def complete_text(self, messages: list[dict], schema: dict, *, role: str) -> str:
        del schema
        return self.backend.complete(role=role, messages=messages)


class PolicyBackend:
    """Expose an LLMClient to the v1 engine, which calls `complete(role=, messages=)`."""

    def __init__(self, client: LLMClient):
        from orbit_agents.schema import ROLE_SCHEMAS

        self.client = client
        self.schemas = ROLE_SCHEMAS

    def complete(self, *, role: str, messages: list[dict]) -> str:
        return self.client.complete_text(messages, self.schemas[role], role=role)


def _distill_teacher():
    """The repo's request cap, cost stop, and Claude client live in finetune/distill_teacher.py."""
    folder = str(ROOT / "finetune")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import distill_teacher

    return distill_teacher


class AnthropicPolicy(LLMClient):
    """Claude as the team policy.

    This reuses the distillation client and its RequestBudget: a hard request cap,
    request pacing, a running USD estimate with a cost stop, and backoff on 429 and
    5xx, where each retry counts toward the cap. Unlike the teacher, this policy
    sends the role prompts and nothing else: no checker contract, no claim names,
    and no gold values. The key comes from ORBIT_ANTHROPIC_API_KEY.
    """

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        max_requests: int,
        max_cost_usd: float,
        requests_per_minute: float | None = None,
        max_output_tokens: int = 512,
        api_key: str | None = None,
        transport=None,
        clock=None,
        sleep=None,
        log=None,
    ):
        teacher = _distill_teacher()
        key = (api_key if api_key is not None else os.environ.get(ANTHROPIC_KEY_ENV, "")).strip()
        if not key:
            raise SystemExit(f"Set {ANTHROPIC_KEY_ENV} to an Anthropic API key. Nothing is read from disk.")
        self.model = model
        self.stops = (teacher.DistillStop,)
        self.budget = teacher.RequestBudget(
            max_requests,
            requests_per_minute or teacher.ANTHROPIC_REQUESTS_PER_MINUTE,
            clock=clock,
            sleep=sleep,
        )
        self.client = teacher.AnthropicClient(
            api_key=key,
            model=model,
            budget=self.budget,
            max_cost_usd=max_cost_usd,
            transport=transport,
            max_output_tokens=max_output_tokens,
            log=log or (lambda event: print(_json_line(event), file=sys.stderr, flush=True)),
        )
        self._preflight = teacher.preflight

    def preflight(self) -> None:
        """One request with no task text, so a bad key or model fails before any task."""
        self._preflight(self.client)

    def complete_text(self, messages: list[dict], schema: dict, *, role: str) -> str:
        del schema, role
        # The Messages API rejects an empty text turn, which a repair retry can carry.
        sent = [{**message, "content": message.get("content") or "(empty reply)"} for message in messages]
        return self.client.generate(sent)

    def usage(self) -> dict:
        budget = self.budget
        return {
            "requests": budget.used,
            "max_requests": budget.max_requests,
            "input_tokens": budget.prompt_tokens,
            "output_tokens": budget.candidates_tokens,
            "estimated_cost_usd": budget.estimated_cost_usd,
            "price_source": budget.price_source,
            "status_counts": budget.status_count_map(),
        }


def _json_line(event: dict) -> str:
    import json

    return json.dumps(event, ensure_ascii=True)


class OpenAICompatiblePolicy(LLMClient):
    """Extension point for a hosted OpenAI-compatible or Meta Model API backend. Not implemented.

    To add one: set `provider`, read the key from an ORBIT_* environment variable,
    build the request in `complete_text` from `messages` (and `schema`, if the API
    supports constrained output), and count every request with the same
    RequestBudget the Anthropic policy uses, so the cap and cost stop apply. Then
    register a factory in PROVIDERS. This repo does not guess any endpoint, model
    name, or pricing for such an API. The local llama.cpp server, which speaks the
    OpenAI chat format on localhost, is already available as `llamacpp`.
    """

    provider = "openai_compatible"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(self.__class__.__doc__)

    def complete_text(self, messages: list[dict], schema: dict, *, role: str) -> str:
        raise NotImplementedError


def _scripted(model, *, task, team, **_kwargs):
    if task is None:
        raise SystemExit("the scripted harness needs a bundled task")
    if team == "v2":
        from orbit_agents.scripted_v2 import ScriptedPolicyV2

        return ScriptedPolicyV2(task)
    from orbit_agents.scripted import ScriptedBackend

    return LegacyBackendPolicy(ScriptedBackend(task), provider="scripted")


def _transformers(model, *, adapter=None, max_new_tokens=256, **_kwargs):
    from orbit_agents.backends import TransformersBackend

    return LegacyBackendPolicy(
        TransformersBackend(model, adapter=adapter, max_new_tokens=max_new_tokens),
        provider="transformers",
        model=model,
    )


def _ollama(model, **_kwargs):
    from orbit_agents.backends import OllamaBackend

    return LegacyBackendPolicy(OllamaBackend(model), provider="ollama", model=model)


def _llamacpp(model, **_kwargs):
    from orbit_agents.backends import LlamaCppBackend

    return LegacyBackendPolicy(LlamaCppBackend(model), provider="llamacpp", model=model)


def _anthropic(model, *, max_requests, max_cost_usd, requests_per_minute=None, **kwargs):
    return AnthropicPolicy(
        model,
        max_requests=max_requests,
        max_cost_usd=max_cost_usd,
        requests_per_minute=requests_per_minute,
        **{key: kwargs[key] for key in ("transport", "clock", "sleep", "log", "api_key") if key in kwargs},
    )


PROVIDERS = {
    "scripted": _scripted,
    "transformers": _transformers,
    "ollama": _ollama,
    "llamacpp": _llamacpp,
    "anthropic": _anthropic,
}
# Scripted runs build one policy per task; the others share one policy, and one budget, across tasks.
PER_TASK_PROVIDERS = {"scripted"}


def make_policy(provider: str, model: str | None = None, **kwargs: object) -> LLMClient:
    """Build the policy for one provider. Unknown names are refused."""
    if provider not in PROVIDERS:
        raise SystemExit(f"unknown provider {provider}; choose one of {', '.join(PROVIDERS)}")
    return PROVIDERS[provider](model, **kwargs)
