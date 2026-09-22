"""The guardrail backend boundary.

Everything the guardrail layers need from a *judgment* engine sits behind this
protocol, so the same agent can run with either:

  * `llm`  -- a generative model prompted to return JSON (the original), with
              the rules also written into the agent's own system prompt, or
  * `jev`  -- TypeSafe's System One model, asked typed questions about the
              conversation state, with the rules removed from the prompt.

The distinction is not just which model answers. In the `llm` build the rules
live in the prompt and the judge is a second opinion. In the `jev` build the
rules are *not in the prompt at all* -- they are questions asked about the
state, and the prompt is left to describe the job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ScopeDecision:
    """Is this request ours, and whose desk is it?"""

    in_scope: bool
    category: str | None            # out-of-scope category when not in scope
    confidence: float               # confidence in the in/out call
    owning_desk: str | None
    desk_confidence: float
    # The FULL distribution, not just the winner. Jev returns one probability
    # per option; a generative judge returns a single self-reported number and
    # these stay empty. That difference is the point of the comparison.
    scope_probabilities: dict[str, float] = field(default_factory=dict)
    desk_probabilities: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    safety_p: float = 0.0           # P(customer is in danger)
    safety_urgent: bool = False     # overrides scope entirely
    # The provider's own safety classifier refused the message. Ambiguous
    # between adversarial and crisis, so it escalates rather than refusing.
    provider_filtered: bool = False
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class RuleFinding:
    rule_id: str
    probability: float
    evidence: str = ""


@dataclass
class ReviewDecision:
    findings: list[RuleFinding]
    rules_evaluated: int
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ToolDecision:
    """AutoMode-style gate on a proposed tool call."""

    action: str                     # allow | confirm | block
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    reason: str = ""


class GuardrailBackend(Protocol):
    name: str

    # Rules that must also be written into the agent's system prompt for this
    # backend to work. False for Jev -- that is the whole point of the variant.
    rules_in_prompt: bool

    def classify_scope(self, message: str, current_desk: str) -> ScopeDecision | None:
        """Judge an incoming customer message. None means 'could not decide'."""
        ...

    def review_output(
        self, reply: str, context, desk: str, turn: int
    ) -> ReviewDecision | None:
        """Judge an outgoing assistant reply against the soft rules."""
        ...

    def check_tool_call(
        self, tool_name: str, args: dict, context, desk: str
    ) -> ToolDecision | None:
        """Judge a proposed tool call before it executes."""
        ...
