"""Handoff tools.

A handoff is a tool that returns a `Command` targeting a sibling node in the
parent graph. The receiving agent inherits the full message history, so the
customer never repeats themselves -- which is the single most common complaint
about transfers in real support.

One sharp edge worth knowing about: a `Command(graph=Command.PARENT)` bubbles
out of the desk's subgraph *immediately*, before that subgraph's own state is
flushed upward. If the update carries only the handoff `ToolMessage`, the
parent ends up with a `tool` message whose matching assistant `tool_calls`
message never arrived -- and the next provider call fails with

    messages with role 'tool' must be a response to a preceding message
    with 'tool_calls'

So each handoff replays the desk's full message list alongside the new
`ToolMessage`. The `add_messages` reducer de-duplicates by id, so replaying
messages the parent already has is a no-op.
"""

from __future__ import annotations

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .. import mock_db as db
from ..prompts import ROUTINE_LABELS
from ..tools._session import session_id


def _carry(runtime: ToolRuntime) -> list:
    """The desk's messages so far, so the parent keeps the assistant turn that
    holds this handoff's `tool_calls`. De-duplicated upstream by `add_messages`."""
    state = getattr(runtime, "state", None) or {}
    return list(state.get("messages") or [])


def make_handoff_tool(target: str, when_to_use: str):
    """Build `transfer_to_<target>` for the triage agent."""
    name = f"transfer_to_{target}"
    label = ROUTINE_LABELS[target]

    @tool(name, description=(
        f"Transfer this conversation to the {label} desk. {when_to_use} "
        "Pass a one-line `task_summary` of what the customer needs so the desk "
        "does not re-ask. The full conversation transfers with it."
    ))
    def _handoff(task_summary: str, runtime: ToolRuntime) -> Command:
        sid = session_id(runtime)
        sess = db.get_session(sid)
        sess["handoffs"].append(target)
        db.audit(sid, "handoff", to=target, summary=task_summary)
        tool_message = ToolMessage(
            content=f"Transferred to {label}. Context carried over: {task_summary}",
            name=name,
            tool_call_id=runtime.tool_call_id,
        )
        return Command(
            goto=target,
            graph=Command.PARENT,
            update={
                "messages": _carry(runtime) + [tool_message],
                "active_agent": target,
                "handoff_note": task_summary,
            },
        )

    return _handoff


@tool(
    "handoff_to_triage",
    description=(
        "Hand the conversation back to triage when the customer's need has "
        "moved outside this desk's scope. Pass `reason` describing what they "
        "now need, and `resolved` summarising what you did finish."
    ),
)
def handoff_to_triage(reason: str, resolved: str, runtime: ToolRuntime) -> Command:
    sid = session_id(runtime)
    sess = db.get_session(sid)
    sess["handoffs"].append("triage")
    db.audit(sid, "handoff", to="triage", reason=reason)
    tool_message = ToolMessage(
        content=(
            f"Returned to triage. Completed here: {resolved or 'nothing yet'}. "
            f"Outstanding: {reason}"
        ),
        name="handoff_to_triage",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="triage",
        graph=Command.PARENT,
        update={
            "messages": _carry(runtime) + [tool_message],
            "active_agent": "triage",
            "handoff_note": reason,
        },
    )


# What triage tells itself about each desk when choosing.
HANDOFF_TARGETS: dict[str, str] = {
    "billing": "Use for bill contents, charges, disputes, credits and billing policy.",
    "payments": "Use for taking payments, payment methods, autopay and arrangements.",
    "home_internet": "Use for Home Internet availability, gateway faults, speed and outages.",
    "plans_devices": "Use for rate plans, adding or cancelling lines, upgrades and orders.",
    "tech_support": "Use for mobile signal, data, calling, SMS, eSIM and device issues.",
    "account": "Use for contact details, address, authorized users and PIN resets.",
}
