"""The original generative-judge backend, behind the common interface.

Wraps `guardrails.judge`: a chat model prompted to return JSON. The agent's own
system prompt also carries the sampled clause block and the scope policy, which
is why `rules_in_prompt` is True here and False for Jev.
"""

from __future__ import annotations

import os

from .. import judge
from ..soft_rules import RULES_BY_ID
from .base import ReviewDecision, RuleFinding, ScopeDecision, ToolDecision


# Same threshold the Jev backend uses, so the comparison is on the signal, not
# on where the line was drawn.
SAFETY_P = float(os.environ.get("TLIFE_JEV_SAFETY_P", "0.40"))


class LLMBackend:
    name = "llm"
    rules_in_prompt = True

    def classify_scope(self, message: str, current_desk: str) -> ScopeDecision | None:
        verdict = judge.classify(message)
        if verdict is None:
            return None
        return ScopeDecision(
            in_scope=not verdict.out_of_scope,
            category=verdict.category,
            confidence=verdict.confidence,
            owning_desk=verdict.owning_desk,
            desk_confidence=verdict.desk_confidence,
            reason=verdict.reason,
            # Asked as its own field in the same JSON call, independent of the
            # scope verdict -- the same question Jev is asked, so the two
            # backends are answering the same thing.
            safety_p=verdict.safety_p,
            safety_urgent=verdict.safety_p >= SAFETY_P,
            provider_filtered=verdict.provider_filtered,
            usage=dict(verdict.usage or {}),
        )

    def review_output(
        self, reply: str, context, desk: str, turn: int
    ) -> ReviewDecision | None:
        rules = judge.sample_rules(_session(), turn, desk=desk)
        text = context if isinstance(context, str) else judge.build_context(context)
        found = judge.review(_session(), desk, reply, text, turn, record=False)
        return ReviewDecision(
            findings=[
                RuleFinding(
                    rule_id=v.evidence.split()[0],
                    probability=_conf(v.evidence),
                    evidence=v.evidence,
                )
                for v in found
                if v.evidence.split()[0] in RULES_BY_ID
            ],
            # The sampled slice, not the whole rubric -- that is the cost
            # tradeoff this backend is making.
            rules_evaluated=len(rules),
            usage=dict(judge.LAST_REVIEW_USAGE),
        )

    def check_tool_call(
        self, tool_name: str, args: dict, context, desk: str
    ) -> ToolDecision | None:
        # Not implemented for the generative backend: a model call per tool call
        # roughly doubles the cost of every turn. The deterministic gates in
        # `guardrails.runtime` cover this path instead.
        return None


def _session() -> str:
    from ...tools._session import current_session_id

    return current_session_id()


def _conf(evidence: str) -> float:
    for part in evidence.split():
        if part.startswith("conf="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return 0.0
    return 0.0
