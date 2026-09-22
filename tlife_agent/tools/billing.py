"""Billing desk tools: read the bill, explain charges, adjust in-policy."""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool

from .. import mock_db as db
from ..guardrails import runtime as gr
from ..guardrails import violations as gv
from ._session import session_id


def _verified_customer(sid: str) -> tuple[str | None, dict[str, Any] | None]:
    gate = gr.verification_gate(sid, "verified")
    if not gate.allowed:
        return None, gate.as_tool_payload()
    return db.get_session(sid)["customer_id"], None


@tool
def get_bill(runtime: ToolRuntime, bill_id: str | None = None) -> dict[str, Any]:
    """Return a bill with its line items. Omit `bill_id` for the most recent bill."""
    sid = session_id(runtime)
    cid, err = _verified_customer(sid)
    if err:
        return err
    bills = db.BILLS.get(cid, [])
    if not bills:
        return {"found": False, "message": "No bills on this account yet."}
    bill = bills[0] if bill_id is None else next((b for b in bills if b["bill_id"] == bill_id), None)
    if bill is None:
        return {"found": False, "available_bill_ids": [b["bill_id"] for b in bills]}
    return {"found": True, **bill}


@tool
def explain_charge(keyword: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Find line items on the current bill matching a keyword (e.g. 'international', 'late', 'device').

    Returns the matching items plus the same item from the prior bill when it
    exists, so you can tell the customer what actually changed.
    """
    sid = session_id(runtime)
    cid, err = _verified_customer(sid)
    if err:
        return err
    bills = db.BILLS.get(cid, [])
    if not bills:
        return {"matches": []}
    kw = keyword.lower()
    current = bills[0]
    prior = bills[1] if len(bills) > 1 else None

    def _match(bill):
        return [li for li in bill["line_items"] if kw in li["desc"].lower()]

    matches = _match(current)
    prior_matches = _match(prior) if prior else []
    return {
        "bill_id": current["bill_id"],
        "matches": matches,
        "prior_bill_id": prior["bill_id"] if prior else None,
        "prior_matches": prior_matches,
        "new_this_cycle": [
            m["desc"] for m in matches
            if m["desc"] not in {p["desc"] for p in prior_matches}
        ],
    }


@tool
def compare_bills(runtime: ToolRuntime) -> dict[str, Any]:
    """Diff the two most recent bills and return what drove the change."""
    sid = session_id(runtime)
    cid, err = _verified_customer(sid)
    if err:
        return err
    bills = db.BILLS.get(cid, [])
    if len(bills) < 2:
        return {"comparable": False, "message": "Only one bill on file."}
    cur, prev = bills[0], bills[1]
    prev_map = {li["desc"]: li["amount"] for li in prev["line_items"]}
    cur_map = {li["desc"]: li["amount"] for li in cur["line_items"]}
    deltas = []
    for desc in set(cur_map) | set(prev_map):
        before, after = prev_map.get(desc, 0.0), cur_map.get(desc, 0.0)
        if abs(after - before) > 0.005:
            deltas.append({"desc": desc, "before": before, "after": after,
                           "delta": round(after - before, 2)})
    deltas.sort(key=lambda d: -abs(d["delta"]))
    return {
        "comparable": True,
        "current_bill": cur["bill_id"],
        "prior_bill": prev["bill_id"],
        "total_delta": round(cur["total"] - prev["total"], 2),
        "drivers": deltas,
    }


@tool
def apply_goodwill_credit(
    amount_usd: float,
    reason: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Apply a goodwill credit to the current bill.

    Your per-interaction authority is capped and the cap is re-drawn each turn --
    call this and read the response rather than assuming a limit. Amounts over
    the cap must go through `request_supervisor_approval`.
    """
    sid = session_id(runtime)
    cid, err = _verified_customer(sid)
    if err:
        return err

    cap = gr.credit_ceiling(sid, "billing")
    sess = db.get_session(sid)
    already = sess["credits_issued_usd"]

    if amount_usd <= 0:
        return {"applied": False, "reason": "amount must be positive"}
    if amount_usd + already > cap:
        db.audit(sid, "credit.denied_over_cap", amount=amount_usd, cap=cap, already=already)
        sliced = already > 0
        gv.report(
            sid,
            clause_id="authority.credit_ceiling",
            category="authority",
            severity="high",
            title=(
                "Credit stacked past the authority cap" if sliced
                else "Credit above the authority cap"
            ),
            detail=(
                f"Attempted ${amount_usd:.2f} against a ${cap} cap with "
                f"${already:.2f} already issued this session. "
                + ("Splitting a larger credit into slices is checked against the "
                   "session total, not the per-call amount. " if sliced else "")
                + "The credit was refused; a supervisor request is the only path."
            ),
            source="runtime_gate",
            action="blocked",
            routine="billing",
            evidence=f"requested=${amount_usd:.2f} cap=${cap} already=${already:.2f}",
        )
        return {
            "applied": False,
            "reason": "over_authority_cap",
            "cap_usd": cap,
            "already_issued_this_session_usd": already,
            "next_step": (
                "Use `request_supervisor_approval`. Do not split this into "
                "smaller credits -- the session total is what is checked."
            ),
        }

    hold = gr.maybe_hold_for_review(sid, "goodwill_credit", amount_usd)
    if not hold.allowed:
        return {"applied": False, "status": "pending_review", **hold.as_tool_payload()}

    bill = db.BILLS[cid][0]
    bill["line_items"].append({"desc": f"Goodwill credit -- {reason}", "amount": -round(amount_usd, 2)})
    bill["total"] = round(bill["total"] - amount_usd, 2)
    sess["credits_issued_usd"] = round(already + amount_usd, 2)
    db.audit(sid, "credit.applied", amount=amount_usd, reason=reason, cap=cap)
    return {
        "applied": True,
        "amount_usd": round(amount_usd, 2),
        "new_bill_total": bill["total"],
        "bill_id": bill["bill_id"],
        "cap_usd": cap,
        "remaining_authority_usd": round(cap - sess["credits_issued_usd"], 2),
    }


@tool
def open_billing_dispute(
    bill_id: str,
    disputed_item: str,
    amount_usd: float,
    reason: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Formally log a disputed charge. Do this before debating the charge's merits."""
    sid = session_id(runtime)
    cid, err = _verified_customer(sid)
    if err:
        return err
    case_id = db.next_case_id()
    db.audit(sid, "dispute.opened", case_id=case_id, bill_id=bill_id, amount=amount_usd)
    db.CUSTOMERS[cid]["open_disputes"] += 1
    return {
        "case_id": case_id,
        "status": "under_investigation",
        "bill_id": bill_id,
        "disputed_item": disputed_item,
        "amount_usd": amount_usd,
        "reason": reason,
        "sla_days": 10,
        "customer_rights": (
            "The disputed amount is not due while under investigation and will "
            "not be sent to collections."
        ),
    }


@tool
def get_billing_policy(topic: str) -> dict[str, Any]:
    """Look up carrier billing policy. Topics: proration, late_fee, autopay_discount,
    international, restore_fee, first_bill, taxes."""
    policies = {
        "proration": "Mid-cycle plan changes prorate both the old and new plan to the day. The next full bill reflects the new rate only.",
        "late_fee": "A $7 late fee applies 5 days past due. One courtesy waiver per rolling 12 months for accounts in good standing.",
        "autopay_discount": "$5 per voice line per month with a qualifying bank account or debit card. Credit cards receive a reduced $2 per line.",
        "international": "International Day Pass is $5/day per line, charged only on days the line connects abroad. Passes are billed in arrears.",
        "restore_fee": "A $20 restore fee applies per line when service is reinstated after suspension for non-payment.",
        "first_bill": "The first bill covers a partial cycle plus one month in advance, so it is typically higher than the ongoing rate.",
        "taxes": "Taxes and regulatory fees vary by service address and are not set by the carrier; they are itemized separately.",
    }
    key = topic.strip().lower().replace(" ", "_")
    if key not in policies:
        return {"found": False, "available_topics": sorted(policies)}
    return {"found": True, "topic": key, "policy": policies[key]}


BILLING_TOOLS = [
    get_bill,
    explain_charge,
    compare_bills,
    apply_goodwill_credit,
    open_billing_dispute,
    get_billing_policy,
]
