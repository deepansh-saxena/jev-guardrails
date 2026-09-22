"""Session plumbing shared by tools and middleware.

Tools need to know *which conversation* they are acting in so the guardrail
runtime can track auth level, rolls and audit entries.

Three resolution paths, in order:
  1. the ``ToolRuntime`` LangChain injects into any tool declaring a
     ``runtime: ToolRuntime`` parameter,
  2. ``langgraph.config.get_config()`` -- works anywhere inside a running graph,
     including middleware, which has no ToolRuntime,
  3. a contextvar, for direct calls from tests and the offline demo.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from langchain.tools import ToolRuntime

_FALLBACK_SESSION: ContextVar[str] = ContextVar("tlife_session", default="default")


def set_fallback_session(session_id_: str) -> None:
    _FALLBACK_SESSION.set(session_id_)


def _thread_id_from_config(cfg: Any) -> str | None:
    if not cfg:
        return None
    tid = (cfg.get("configurable") or {}).get("thread_id")
    return str(tid) if tid else None


def current_session_id() -> str:
    """Session id from the ambient graph config, else the contextvar."""
    try:
        from langgraph.config import get_config

        tid = _thread_id_from_config(get_config())
        if tid:
            return tid
    except Exception:
        pass
    return _FALLBACK_SESSION.get()


def session_id(runtime: ToolRuntime | None = None) -> str:
    if runtime is not None:
        tid = _thread_id_from_config(getattr(runtime, "config", None))
        if tid:
            return tid
    return current_session_id()
