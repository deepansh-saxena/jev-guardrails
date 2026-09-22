"""The care graph: triage plus every desk, wired for sticky routing.

Shape:

    START --(active_agent)--> [ triage | billing | payments | ... ] --> END

Each node is a compiled ReAct agent. Handoff tools return
`Command(goto=..., graph=Command.PARENT)`, which the parent graph applies -- so
a transfer re-enters the graph at the target desk inside the same turn, with the
whole message history intact.

`active_agent` persists in the checkpointer, so the customer's *next* message
goes straight back to the desk handling them instead of through triage again.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .routines.desks import DESK_NAMES, build_all_desks
from .state import CareState
from .triage import build_triage

NODE_NAMES = ("triage",) + DESK_NAMES


def _route_entry(state: CareState) -> str:
    """Resume with whichever desk owns this conversation."""
    active = state.get("active_agent") or "triage"
    return active if active in NODE_NAMES else "triage"


def build_care_graph(
    model: Any = None,
    *,
    checkpointer: Any | None = None,
    compile_graph: bool = True,
):
    """Build (and by default compile) the full customer service graph."""
    graph = StateGraph(CareState)

    graph.add_node("triage", build_triage(model))
    for name, desk in build_all_desks(model).items():
        graph.add_node(name, desk)

    graph.add_conditional_edges(
        START, _route_entry, {name: name for name in NODE_NAMES}
    )
    # A desk that finishes without handing off ends the turn and waits for the
    # customer. Handoffs bypass this edge via Command(goto=...).
    for name in NODE_NAMES:
        graph.add_edge(name, END)

    if not compile_graph:
        return graph
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
