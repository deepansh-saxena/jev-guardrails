"""Tools every routine shares: identity, verification, escalation, ticketing."""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool

from .. import mock_db as db
from ..guardrails import runtime as gr
from ._session import session_id


@tool
def lookup_account(identifier: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Find an account by phone number, email, customer id, or full name.

    Returns only non-sensitive identifying fields. This does NOT verify the
    caller -- call `verify_identity` before disclosing anything account-specific.
    """
    sid = session_id(runtime)
    cust = db.find_customer(identifier)
    if not cust:
        db.audit(sid, "lookup_account.miss", identifier=db.mask(identifier, 4))
        return {"found": False, "message": "No account matched that identifier."}

    sess = db.get_session(sid)
    sess["customer_id"] = cust["customer_id"]
    if sess["auth_level"] == "none":
        sess["auth_level"] = "soft"
    db.audit(sid, "lookup_account.hit", customer_id=cust["customer_id"])
    return {
        "found": True,
        "customer_id": cust["customer_id"],
        "name_on_account": cust["name"],
        "phone_last4": cust["phone"][-4:],
        "segment": cust["segment"],
        "tenure_months": cust["tenure_months"],
        "auth_level": sess["auth_level"],
        "note": "auth_level is 'soft'. Account details stay sealed until 'verified'.",
    }


@tool
def verify_identity(
    identifier: str,
    account_pin: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Verify the caller with their 4-digit account PIN.

    `identifier` is the phone/email/customer id they gave you. Never echo the
    PIN back in your reply.
    """
    sid = session_id(runtime)
    cust = db.find_customer(identifier)
    if not cust:
        return {"verified": False, "reason": "account_not_found"}

    sess = db.get_session(sid)
    sess.setdefault("verify_attempts", 0)
    ok = cust["account_pin_hash"] == f"pin:{account_pin.strip()}"
    if not ok:
        sess["verify_attempts"] += 1
        db.audit(sid, "verify_identity.fail", customer_id=cust["customer_id"],
                 attempts=sess["verify_attempts"])
        locked = sess["verify_attempts"] >= 2
        return {
            "verified": False,
            "reason": "pin_mismatch",
            "attempts": sess["verify_attempts"],
            "locked_out": locked,
            "next_step": (
                "Stop verifying. Offer the in-app identity check or a store visit."
                if locked
                else "One more attempt is allowed."
            ),
        }

    sess["customer_id"] = cust["customer_id"]
    sess["auth_level"] = "verified"
    sess["verify_attempts"] = 0
    db.audit(sid, "verify_identity.success", customer_id=cust["customer_id"])
    return {
        "verified": True,
        "auth_level": "verified",
        "customer_id": cust["customer_id"],
        "fraud_watch": cust["fraud_watch"],
        "note": (
            "Account data is now readable. Money movement and SIM/address/line "
            "changes may still require a step-up challenge."
        ),
    }


@tool
def check_verification(runtime: ToolRuntime) -> dict[str, Any]:
    """Report the current verification level for this conversation."""
    sid = session_id(runtime)
    sess = db.get_session(sid)
    return {
        "auth_level": sess["auth_level"],
        "customer_id": sess["customer_id"],
        "handoffs_so_far": len(sess["handoffs"]),
        "credits_issued_this_session_usd": sess["credits_issued_usd"],
    }


@tool
def confirm_step_up(code: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Submit the 6-digit one-time code the customer read back, to clear a step-up challenge."""
    sid = session_id(runtime)
    result = gr.confirm_step_up_code(sid, code)
    return {
        "confirmed": result.allowed,
        "reason": result.reason,
        "auth_level": db.get_session(sid)["auth_level"],
    }


@tool
def get_account_summary(runtime: ToolRuntime) -> dict[str, Any]:
    """Return a masked overview of the verified account: lines, plan, autopay, balance state."""
    sid = session_id(runtime)
    gate = gr.verification_gate(sid, "verified")
    if not gate.allowed:
        return gate.as_tool_payload()

    cid = db.get_session(sid)["customer_id"]
    cust = db.CUSTOMERS[cid]
    bills = db.BILLS.get(cid, [])
    open_bill = next((b for b in bills if b["status"] != "paid"), None)
    return {
        "customer_id": cid,
        "name": cust["name"],
        "segment": cust["segment"],
        "tenure_months": cust["tenure_months"],
        "autopay": cust["autopay"],
        "address_city_state": ", ".join(cust["address"].split(", ")[-2:]),
        "phone_last4": cust["phone"][-4:],
        "fraud_watch": cust["fraud_watch"],
        "accessibility_flags": cust["accessibility_flags"],
        "lines": [
            {
                "line_id": l["line_id"],
                "msisdn_last4": l["msisdn"][-4:],
                "owner": l["owner"],
                "plan": db.PLANS[l["plan_id"]]["name"],
                "device": l["device"],
                "status": l["status"],
            }
            for l in db.LINES.get(cid, [])
        ],
        "open_balance": None if not open_bill else {
            "bill_id": open_bill["bill_id"],
            "total": open_bill["total"],
            "due_date": open_bill["due_date"],
            "status": open_bill["status"],
        },
        "has_home_internet": cid in db.HOME_INTERNET,
    }


@tool
def create_ticket(
    subject: str,
    details: str,
    runtime: ToolRuntime,
    priority: str = "normal",
) -> dict[str, Any]:
    """Open a tracking ticket for work that cannot be completed in this conversation.

    `priority` is one of: low, normal, high, urgent.
    """
    sid = session_id(runtime)
    sess = db.get_session(sid)
    ticket = {
        "ticket_id": db.next_ticket_id(),
        "customer_id": sess["customer_id"],
        "subject": subject,
        "details": details,
        "priority": priority,
        "status": "open",
    }
    db.TICKETS.append(ticket)
    db.audit(sid, "create_ticket", ticket_id=ticket["ticket_id"], priority=priority)
    return {"ticket_id": ticket["ticket_id"], "status": "open", "priority": priority}


@tool
def escalate_to_human(
    reason: str,
    summary: str,
    runtime: ToolRuntime,
    priority: str = "high",
) -> dict[str, Any]:
    """Hand the conversation to a human specialist.

    Use for safety concerns, legal/regulatory threats, fraud, repeat contacts,
    or anything outside every routine's authority. `priority` may be
    low/normal/high/urgent.
    """
    sid = session_id(runtime)
    sess = db.get_session(sid)
    case_id = db.next_case_id()
    db.audit(sid, "escalate_to_human", case_id=case_id, reason=reason, priority=priority)
    sess["actions_taken"].append(f"escalated:{case_id}")
    queue = {
        "urgent": "Care Specialist -- immediate, under 2 minutes",
        "high": "Tier 2 Care -- under 15 minutes",
        "normal": "Care callback -- within 4 hours",
        "low": "Async case -- within 1 business day",
    }.get(priority, "Care callback -- within 4 hours")
    return {
        "case_id": case_id,
        "priority": priority,
        "queue": queue,
        "context_transferred": summary,
        "instructions": (
            "Tell the customer a specialist is taking over, give them the case "
            "id and the wait, and confirm they will not need to repeat themselves."
        ),
    }


@tool
def request_supervisor_approval(
    action: str,
    amount_usd: float,
    justification: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Request supervisor sign-off for an action above your authority (e.g. a large credit).

    Returns a pending decision -- never treat it as approved.
    """
    sid = session_id(runtime)
    case_id = db.next_case_id()
    db.audit(sid, "supervisor_approval.requested", case_id=case_id,
             action=action, amount=amount_usd)
    return {
        "case_id": case_id,
        "status": "pending_review",
        "action": action,
        "amount_usd": amount_usd,
        "justification": justification,
        "sla_hours": 2,
        "instructions": (
            "This is PENDING, not approved. Tell the customer it is submitted, "
            "give the case id and the 2-hour SLA, and do not promise the outcome."
        ),
    }


COMMON_TOOLS = [
    lookup_account,
    verify_identity,
    check_verification,
    confirm_step_up,
    get_account_summary,
    create_ticket,
    escalate_to_human,
    request_supervisor_approval,
]
