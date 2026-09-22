"""A T-Life-style multi-agent customer service assistant built on LangChain/LangGraph.

Public surface:

    from tlife_agent import build_care_graph
    graph = build_care_graph()
    graph.invoke({"messages": [("user", "why is my bill higher?")]},
                 config={"configurable": {"thread_id": "session-1"}})
"""

from .graph import NODE_NAMES, build_care_graph
from .guardrails.sampler import GuardrailSampler, sampler_for
from .state import CareState

__all__ = [
    "build_care_graph",
    "NODE_NAMES",
    "CareState",
    "GuardrailSampler",
    "sampler_for",
]
