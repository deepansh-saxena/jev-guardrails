"""Payment desk tools: methods, one-time payments, arrangements, autopay.

Every money-moving tool routes through the stochastic step-up gate first. The
agent is expected to surface the challenge, not retry around it.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from langchain.tools import ToolRuntime, tool

from .. import mock_db as db
from ..guardrails import runtime as gr
from ._session import session_id


def _verified(sid: str) -> tuple[str | None, dict[str, Any] | None]:
    gate = gr.verification_gate(sid, "verified")
    if not gate.allowed:
        return None, gate.as_tool_payload()
    return db.get_session(sid)["customer_id"], None


@tool
def list_payment_methods(runtime: ToolRuntime) -> dict[str, Any]:
    """List saved payment methods, masked to the last four digits."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    return {
        "methods": [
            {
                "method_id": m["method_id"],
                "type": m["type"],
                "last4": m["last4"],
                "expires": m["exp"],
                "default": m["default"],
            }
            for m in db.PAYMENT_METHODS.get(cid, [])
        ]
    }


@tool
def get_payment_history(runtime: ToolRuntime, limit: int = 5) -> dict[str, Any]:
    """Recent payments, newest first, including failures and their reason codes."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    return {"payments": db.PAYMENT_HISTORY.get(cid, [])[:limit]}


@tool
def make_payment(
    amount_usd: float,
    method_id: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Charge a saved payment method.

    Read the exact amount and the method's last four back to the customer and
    get an explicit yes BEFORE calling this. A `step_up_required` response is a
    gate to walk through, not an error to retry.
    """
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err

    methods = {m["method_id"]: m for m in db.PAYMENT_METHODS.get(cid, [])}
    if method_id not in methods:
        return {"posted": False, "reason": "unknown_method_id",
                "available": sorted(methods)}
    if amount_usd <= 0:
        return {"posted": False, "reason": "amount must be positive"}

    risk = "critical" if amount_usd >= 400 else "high"
    gate = gr.maybe_require_step_up(sid, "make_payment", risk)
    if not gate.allowed:
        return {"posted": False, "status": "step_up_required", **gate.as_tool_payload()}

    hold = gr.maybe_hold_for_review(sid, "make_payment", amount_usd)
    if not hold.allowed:
        return {"posted": False, "status": "pending_review", **hold.as_tool_payload()}

    method = methods[method_id]
    payment = {
        "payment_id": f"PAY-{db.next_ticket_id().split('-')[1]}",
        "date": _dt.date(2026, 9, 21).isoformat(),
        "amount": round(amount_usd, 2),
        "method_id": method_id,
        "status": "posted",
    }
    db.PAYMENT_HISTORY.setdefault(cid, []).insert(0, payment)
    bills = db.BILLS.get(cid, [])
    if bills:
        bills[0]["total"] = round(bills[0]["total"] - amount_usd, 2)
        if bills[0]["total"] <= 0:
            bills[0]["status"] = "paid"
    db.audit(sid, "payment.posted", amount=amount_usd, method_last4=method["last4"])
    return {
        "posted": True,
        "payment_id": payment["payment_id"],
        "amount_usd": payment["amount"],
        "method": f"{method['type']} ending {method['last4']}",
        "remaining_balance": bills[0]["total"] if bills else 0.0,
    }


@tool
def setup_payment_arrangement(
    total_usd: float,
    installments: int,
    first_payment_date: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Split a past-due balance into up to 3 installments.

    `first_payment_date` is ISO (YYYY-MM-DD). Arrangements keep service on while
    the schedule is met.
    """
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    if not 2 <= installments <= 3:
        return {"created": False, "reason": "installments must be 2 or 3"}

    gate = gr.maybe_require_step_up(sid, "payment_arrangement", "medium")
    if not gate.allowed:
        return {"created": False, "status": "step_up_required", **gate.as_tool_payload()}

    per = round(total_usd / installments, 2)
    arrangement = {
        "arrangement_id": db.next_case_id(),
        "total_usd": round(total_usd, 2),
        "installments": installments,
        "amount_each": per,
        "first_payment_date": first_payment_date,
        "status": "active",
        "terms": (
            "Service stays on while payments are made on schedule. A missed "
            "installment cancels the arrangement and the full balance becomes due."
        ),
    }
    db.PAYMENT_ARRANGEMENTS.setdefault(cid, []).append(arrangement)
    db.audit(sid, "arrangement.created", **{k: arrangement[k] for k in ("arrangement_id", "total_usd", "installments")})
    return {"created": True, **arrangement}


@tool
def enroll_autopay(method_id: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Turn on autopay with a saved method and report the resulting monthly discount."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    methods = {m["method_id"]: m for m in db.PAYMENT_METHODS.get(cid, [])}
    if method_id not in methods:
        return {"enrolled": False, "reason": "unknown_method_id", "available": sorted(methods)}

    gate = gr.maybe_require_step_up(sid, "enroll_autopay", "medium")
    if not gate.allowed:
        return {"enrolled": False, "status": "step_up_required", **gate.as_tool_payload()}

    method = methods[method_id]
    voice_lines = sum(1 for l in db.LINES.get(cid, []) if l["owner"] != "watch")
    per_line = 5.00 if method["type"] in {"checking", "debit"} else 2.00
    db.CUSTOMERS[cid]["autopay"] = True
    db.audit(sid, "autopay.enrolled", method_last4=method["last4"])
    return {
        "enrolled": True,
        "method": f"{method['type']} ending {method['last4']}",
        "discount_per_line_usd": per_line,
        "voice_lines": voice_lines,
        "monthly_discount_usd": round(per_line * voice_lines, 2),
        "note": "Bank account or debit earns the full $5/line; credit cards earn $2/line.",
    }


@tool
def cancel_autopay(reason: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Turn off autopay. State the discount the customer will lose before calling this."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    db.CUSTOMERS[cid]["autopay"] = False
    db.audit(sid, "autopay.cancelled", reason=reason)
    return {
        "cancelled": True,
        "effective": "next bill cycle",
        "discount_lost": "Autopay discount is removed from the next bill.",
    }


@tool
def start_secure_card_capture(runtime: ToolRuntime) -> dict[str, Any]:
    """Send the customer a secure link to add or update a payment card themselves.

    Use this whenever a new card is needed. You must never ask for, receive, or
    relay a card number, CVV, or expiry in chat.
    """
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    gate = gr.maybe_require_step_up(sid, "add_payment_method", "critical")
    if not gate.allowed:
        return {"sent": False, "status": "step_up_required", **gate.as_tool_payload()}
    db.audit(sid, "secure_capture.sent")
    return {
        "sent": True,
        "channel": "SMS to the number on file",
        "expires_minutes": 15,
        "instructions": (
            "Tell the customer to enter the card on the secure page. Do not ask "
            "them to type any card details in this chat."
        ),
    }


PAYMENT_TOOLS = [
    list_payment_methods,
    get_payment_history,
    make_payment,
    setup_payment_arrangement,
    enroll_autopay,
    cancel_autopay,
    start_secure_card_capture,
]
