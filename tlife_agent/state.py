"""Graph state shared by triage and every routine."""

from __future__ import annotations

from langchain.agents import AgentState


class CareState(AgentState):
    """`messages` comes from AgentState; the rest is routing bookkeeping.

    `active_agent` makes routing *sticky*: once a customer is transferred to
    Billing, their next message goes straight back to Billing rather than
    through triage again. That is how a real contact center behaves, and it
    stops the agent re-asking what desk they need every turn.
    """

    active_agent: str
    handoff_note: str
