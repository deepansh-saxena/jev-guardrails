"""Routine factory.

Every desk is the same shape: a ReAct agent over the shared tools plus its own,
with the guardrail block re-sampled on every single model call.

The resampling is what makes the guardrails stochastic at the *prompt* layer.
`guardrails.runtime` does the same at the *tool* layer.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    PIIMiddleware,
    after_model,
    before_model,
    dynamic_prompt,
    wrap_model_call,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .. import mock_db as db
from ..config import (
    AZURE_API_KEY,
    AZURE_ENDPOINT,
    DEFAULT_ROUTINE_MODEL,
    split_model,
)
from ..prompts import build_system_prompt, static_prompt
from ..state import CareState
from ..tools._session import current_session_id
from ..tools.common import COMMON_TOOLS
from .handoff import handoff_to_triage

_MODEL_CACHE: dict[str, BaseChatModel] = {}


def get_model(spec: str | BaseChatModel | None = None) -> BaseChatModel:
    if isinstance(spec, BaseChatModel):
        return spec
    name = spec or DEFAULT_ROUTINE_MODEL
    if name not in _MODEL_CACHE:
        # temperature stays at the provider default: the *rules* are stochastic
        # here, not the token sampling. Those are different knobs and mixing
        # them makes failures impossible to attribute.
        _MODEL_CACHE[name] = _build_model(name)
    return _MODEL_CACHE[name]


def _build_model(spec: str) -> BaseChatModel:
    provider, name = split_model(spec)
    if provider == "azure":
        # The resource's `/openai/v1` surface speaks the OpenAI wire protocol,
        # so the plain ChatOpenAI client works against it -- no api-version, no
        # /deployments/{name}/ path building. `name` is the DEPLOYMENT name.
        from langchain_openai import ChatOpenAI

        if not AZURE_ENDPOINT:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT is not set. Put it in .env or the "
                "environment, e.g. https://<resource>.openai.azure.com/openai/v1"
            )
        return ChatOpenAI(
            model=name,
            base_url=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            # gpt-5.x reasoning models reject a custom temperature; leaving it
            # unset keeps this deployment-agnostic.
        )
    return init_chat_model(spec)


@wrap_model_call
def _sequential_tool_calls(request: ModelRequest, handler):
    """Force one tool call per step on OpenAI models.

    Handoffs are tools that return `Command(goto=...)`. If the model emits a
    handoff in the same assistant turn as an ordinary tool call, the routing
    command and the sibling tool result race each other. One call per step
    removes the race, and costs a little latency on read-only lookups.

    `model_settings` is splatted into `bind_tools`, so this reaches the OpenAI
    request as `parallel_tool_calls=False`.
    """
    provider, _ = split_model(DEFAULT_ROUTINE_MODEL)
    if provider in {"openai", "azure", "azure_openai"}:
        settings = {**request.model_settings, "parallel_tool_calls": False}
        request = request.override(model_settings=settings)
    return handler(request)


def make_scope_judge(routine: str):
    """Classify each incoming customer message against the topical scope policy.

    Runs only when the newest message is from the customer, so a handoff or a
    tool result does not trigger a second judgment -- exactly one judge call per
    customer turn, at whichever desk receives it.

    A confident out-of-scope verdict ends the turn with a deflection before the
    agent loop runs. Anything below the threshold is recorded for review and the
    agent handles the turn normally.
    """

    @before_model(can_jump_to=["end"], name=f"scope_judge_{routine}")
    def _judge(state, runtime):  # noqa: ANN001 - middleware signature
        from ..guardrails import judge, recording
        from ..guardrails.backends import get_backend

        if not judge.JUDGE_ENABLED:
            return None
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], HumanMessage):
            return None

        text = getattr(messages[-1], "text", "") or str(messages[-1].content)
        backend = get_backend()
        decision = backend.classify_scope(text, routine)
        if decision is None:
            return None

        sid = current_session_id()
        if decision.safety_urgent:
            recording.record_safety(sid, routine, backend.name)

        # Per-desk boundary: in scope for the company, wrong desk for this
        # subagent. Becomes a routing note the prompt middleware injects once.
        db.get_session(sid)["routing_note"] = recording.record_boundary(
            sid, decision, routine, backend.name
        )

        if recording.record_scope(sid, decision, routine, backend.name):
            return {
                "messages": [AIMessage(content=judge.deflection_for(
                    judge.ScopeVerdict("out_of_scope", decision.category,
                                       decision.confidence, decision.reason)
                ))],
                "jump_to": "end",
            }
        return None

    return _judge


def make_output_scanner(routine: str):
    """Check every assistant turn against the deterministic content rules.

    Runs after the model, before the text can reach the customer. PII classes
    are redacted in place (the message is replaced by id, which `add_messages`
    merges); everything else is recorded as a violation and surfaced in the UI.
    """

    @after_model(name=f"scan_{routine}_output")
    def _scan(state, runtime):  # noqa: ANN001 - middleware signature
        from ..guardrails.output_scan import scan

        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None
        text = getattr(last, "text", "") or ""
        if not isinstance(text, str) or not text.strip():
            return None

        sid = current_session_id()
        cleaned, found = scan(text, sid, routine)

        # The soft rubric only judges a finished reply. A message that is still
        # making tool calls is mid-thought, and reviewing it would flag the
        # assistant for not having answered yet.
        if not last.tool_calls:
            from ..guardrails import judge, recording
            from ..guardrails.backends import get_backend

            if judge.JUDGE_ENABLED and judge.REVIEW_ENABLED:
                backend = get_backend()
                decision = backend.review_output(
                    cleaned,
                    list(messages[:-1]),
                    routine,
                    db.get_session(sid).get("turn", 0),
                )
                if decision is not None:
                    recording.record_review(sid, decision, routine, backend.name)

        if cleaned != text:
            # Replace by id so the redacted version is what persists.
            return {"messages": [AIMessage(id=last.id, content=cleaned,
                                           tool_calls=last.tool_calls or [])]}
        return None

    return _scan


def make_guardrail_middleware(routine: str):
    """Resample the guardrail block on every model call for this routine."""

    @dynamic_prompt
    def _prompt(request: ModelRequest) -> str:
        sid = current_session_id()
        sess = db.get_session(sid)
        sess["turn"] = sess.get("turn", 0) + 1
        prompt = build_system_prompt(routine, sid, sess["turn"])

        # A boundary verdict from this turn's judge, consumed once.
        note = sess.pop("routing_note", None)
        if note:
            prompt += "\n\n" + note

        from ..guardrails.sampler import sampler_for

        draw = sampler_for(sid).latest(routine)
        if draw is not None:
            sess["guardrail_trace"].append(draw.as_trace())
        return prompt

    return _prompt


def build_routine(
    routine: str,
    tools: Sequence[Any],
    *,
    model: str | BaseChatModel | None = None,
    extra_middleware: Sequence[Any] = (),
    include_handback: bool = True,
) -> Any:
    """Compile one desk as a LangGraph node."""
    all_tools = list(tools) + list(COMMON_TOOLS)
    if include_handback:
        all_tools.append(handoff_to_triage)

    middleware = [
        make_scope_judge(routine),
        _sequential_tool_calls,
        # Deterministic floor under the stochastic layer: a card number pasted
        # into chat never reaches the model at all.
        PIIMiddleware("credit_card", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_input=False, apply_to_output=True),
        make_guardrail_middleware(routine),
        make_output_scanner(routine),
        *extra_middleware,
    ]

    return create_agent(
        model=get_model(model),
        tools=all_tools,
        # A static fallback so the agent is still safe if middleware is stripped.
        system_prompt=static_prompt(routine),
        middleware=middleware,
        state_schema=CareState,
        name=routine,
    )
