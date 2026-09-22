"""The generalized triage agent.

Triage is deliberately *not* a classifier over a keyword list. It is a full
agent with:

  * a handoff tool per desk, described by intent rather than vocabulary,
  * the shared identity/verification tools, so it can pre-verify and the
    receiving desk starts unblocked,
  * escalation, so "no desk fits" has a real answer instead of a forced route,
  * the same stochastic guardrail block every desk gets.

That combination is what makes it general: a request it has never seen still
has a correct outcome, including "answer it myself" and "this needs a human".
"""

from __future__ import annotations

from typing import Any

from .routines.base import build_routine
from .routines.handoff import HANDOFF_TARGETS, make_handoff_tool

TRIAGE_HANDOFF_TOOLS = [
    make_handoff_tool(target, when) for target, when in HANDOFF_TARGETS.items()
]


def build_triage(model: Any = None):
    return build_routine(
        "triage",
        TRIAGE_HANDOFF_TOOLS,
        model=model,
        include_handback=False,  # triage cannot hand off to itself
    )
