"""Guardrail backend selection.

    TLIFE_GUARDRAIL_BACKEND=llm    generative judge, rules also in the prompt
    TLIFE_GUARDRAIL_BACKEND=jev    typed questions, rules NOT in the prompt
"""

from __future__ import annotations

import os
from contextvars import ContextVar

from .base import (
    GuardrailBackend,
    ReviewDecision,
    RuleFinding,
    ScopeDecision,
    ToolDecision,
)

BACKEND_NAME = os.environ.get("TLIFE_GUARDRAIL_BACKEND", "llm").strip().lower()

# Per-context, not a module global: the comparison server runs a jev agent and
# an llm agent concurrently in a thread pool, and a global would let whichever
# thread called set_backend() last decide the guardrails for both. A
# ContextVar is per-thread here, because ThreadPoolExecutor workers each start
# with a fresh context.
_ACTIVE: ContextVar[GuardrailBackend | None] = ContextVar("tlife_backend", default=None)

# Instances are cached and reused so HTTP connection pools stay warm.
_INSTANCES: dict[str, GuardrailBackend] = {}


def get_backend() -> GuardrailBackend:
    current = _ACTIVE.get()
    if current is not None:
        return current
    return _instance(BACKEND_NAME)


def set_backend(name: str) -> GuardrailBackend:
    """Make `name` the active backend for this context (thread/task)."""
    global BACKEND_NAME
    BACKEND_NAME = name.strip().lower()
    backend = _instance(BACKEND_NAME)
    _ACTIVE.set(backend)
    return backend


def use_backend(name: str) -> GuardrailBackend:
    """Set the backend for THIS context only, without moving the global default.

    This is what the comparison server uses inside each worker thread.
    """
    backend = _instance(name.strip().lower())
    _ACTIVE.set(backend)
    return backend


def _instance(name: str) -> GuardrailBackend:
    if name not in _INSTANCES:
        _INSTANCES[name] = _build(name)
    return _INSTANCES[name]


def _build(name: str) -> GuardrailBackend:
    if name == "jev":
        from .jev_backend import JevBackend

        return JevBackend()
    if name == "llm":
        from .llm_backend import LLMBackend

        return LLMBackend()
    raise ValueError(f"unknown guardrail backend {name!r}; use 'llm' or 'jev'")


__all__ = [
    "GuardrailBackend",
    "ScopeDecision",
    "ReviewDecision",
    "RuleFinding",
    "ToolDecision",
    "get_backend",
    "set_backend",
    "use_backend",
    "BACKEND_NAME",
]
