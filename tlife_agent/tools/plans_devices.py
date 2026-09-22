"""Plans, devices and orders desk."""

from __future__ import annotations

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
def list_plans() -> dict[str, Any]:
    """List sellable plans with prices and perks. No verification needed."""
    return {"plans": list(db.PLANS.values())}


@tool
def get_lines(runtime: ToolRuntime) -> dict[str, Any]:
    """List the lines on the account with plan, device, status and device balance."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    return {
        "lines": [
            {
                "line_id": l["line_id"],
                "msisdn_last4": l["msisdn"][-4:],
                "owner": l["owner"],
                "plan": db.PLANS[l["plan_id"]]["name"],
                "plan_id": l["plan_id"],
                "device": l["device"],
                "device_balance_usd": l["device_balance"],
                "device_months_left": l["device_months_left"],
                "status": l["status"],
                "insurance": l["insurance"],
                "data_used_gb": l["data_used_gb"],
            }
            for l in db.LINES.get(cid, [])
        ]
    }


@tool
def price_plan_change(line_id: str, new_plan_id: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Quote what a plan change would cost per month, without changing anything.

    Always quote before changing. Report the monthly delta, not just the new total.
    """
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    line = next((l for l in db.LINES.get(cid, []) if l["line_id"] == line_id), None)
    if not line:
        return {"quoted": False, "reason": "unknown_line_id"}
    if new_plan_id not in db.PLANS:
        return {"quoted": False, "valid_plan_ids": sorted(db.PLANS)}

    old, new = db.PLANS[line["plan_id"]], db.PLANS[new_plan_id]
    lost = [p for p in old["perks"] if p not in new["perks"]]
    return {
        "quoted": True,
        "line_id": line_id,
        "from_plan": old["name"],
        "to_plan": new["name"],
        "monthly_delta_usd": round(new["price_per_line"] - old["price_per_line"], 2),
        "new_monthly_usd": new["price_per_line"],
        "perks_lost": lost,
        "perks_gained": [p for p in new["perks"] if p not in old["perks"]],
        "eligibility_note": new.get("eligibility"),
        "proration": "Both plans prorate to the day of the change.",
    }


@tool
def change_plan(line_id: str, new_plan_id: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Change a line's plan. Quote it with `price_plan_change` and get an explicit yes first."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    line = next((l for l in db.LINES.get(cid, []) if l["line_id"] == line_id), None)
    if not line:
        return {"changed": False, "reason": "unknown_line_id"}
    if new_plan_id not in db.PLANS:
        return {"changed": False, "valid_plan_ids": sorted(db.PLANS)}

    gate = gr.maybe_require_step_up(sid, "change_plan", "high")
    if not gate.allowed:
        return {"changed": False, "status": "step_up_required", **gate.as_tool_payload()}

    old_id = line["plan_id"]
    line["plan_id"] = new_plan_id
    db.audit(sid, "plan.changed", line_id=line_id, old=old_id, new=new_plan_id)
    return {
        "changed": True,
        "line_id": line_id,
        "from_plan": db.PLANS[old_id]["name"],
        "to_plan": db.PLANS[new_plan_id]["name"],
        "monthly_delta_usd": round(
            db.PLANS[new_plan_id]["price_per_line"] - db.PLANS[old_id]["price_per_line"], 2
        ),
        "effective": "immediately, prorated",
    }


@tool
def check_upgrade_eligibility(line_id: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Check whether a line can upgrade its device now, and what it would take."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    line = next((l for l in db.LINES.get(cid, []) if l["line_id"] == line_id), None)
    if not line:
        return {"eligible": False, "reason": "unknown_line_id"}
    plan = db.PLANS[line["plan_id"]]
    cadence = plan.get("upgrade_cadence_years")
    paid_pct = 0.0 if line["device_balance"] == 0 else max(0.0, 1 - line["device_balance"] / 1200)
    eligible = cadence is not None and (line["device_balance"] == 0 or paid_pct >= 0.5)
    return {
        "line_id": line_id,
        "eligible": eligible,
        "plan": plan["name"],
        "upgrade_cadence_years": cadence,
        "device_balance_usd": line["device_balance"],
        "months_left": line["device_months_left"],
        "path_if_ineligible": (
            None if eligible else
            "Pay down the device balance to 50% or trade in the device to clear it. "
            "The balance cannot be waived."
        ),
    }


@tool
def get_order_status(order_id: str) -> dict[str, Any]:
    """Look up a device order by id. Known demo orders: ORD-55120, ORD-55121."""
    orders = {
        "ORD-55120": {
            "order_id": "ORD-55120",
            "item": "iPhone 17 Pro 256GB Deep Blue",
            "status": "in_transit",
            "carrier": "UPS",
            "tracking_last4": "9921",
            "eta": "2026-09-23",
        },
        "ORD-55121": {
            "order_id": "ORD-55121",
            "item": "5G Gateway G5AR",
            "status": "backordered",
            "carrier": None,
            "tracking_last4": None,
            "eta": None,
            "note": "No ETA available from the supplier. Do not invent one.",
        },
    }
    order = orders.get(order_id.strip().upper())
    if not order:
        return {"found": False, "message": "No order with that id."}
    return {"found": True, **order}


@tool
def add_line(plan_id: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Add a new voice line on the given plan. Quote the monthly cost and get a yes first."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    if plan_id not in db.PLANS:
        return {"added": False, "valid_plan_ids": sorted(db.PLANS)}

    gate = gr.maybe_require_step_up(sid, "add_line", "high")
    if not gate.allowed:
        return {"added": False, "status": "step_up_required", **gate.as_tool_payload()}

    plan = db.PLANS[plan_id]
    line_id = f"LN-{9000 + len(db.LINES.get(cid, []))}"
    db.LINES.setdefault(cid, []).append({
        "msisdn": "+1-000-555-0000", "line_id": line_id, "owner": "new line",
        "plan_id": plan_id, "device": "BYOD", "device_balance": 0.0,
        "device_months_left": 0, "status": "pending_activation",
        "insurance": None, "data_used_gb": 0.0,
    })
    db.audit(sid, "line.added", line_id=line_id, plan_id=plan_id)
    return {
        "added": True,
        "line_id": line_id,
        "plan": plan["name"],
        "monthly_usd": plan["price_per_line"],
        "activation_fee_usd": 35.00,
        "status": "pending_activation",
    }


@tool
def cancel_line(line_id: str, reason: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Cancel a line. Disclose any remaining device balance BEFORE calling this.

    Answer the customer's cancellation question directly first; a retention
    offer may be mentioned once, after that.
    """
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    line = next((l for l in db.LINES.get(cid, []) if l["line_id"] == line_id), None)
    if not line:
        return {"cancelled": False, "reason": "unknown_line_id"}

    gate = gr.maybe_require_step_up(sid, "cancel_line", "critical")
    if not gate.allowed:
        return {"cancelled": False, "status": "step_up_required", **gate.as_tool_payload()}

    balance = line["device_balance"]
    line["status"] = "pending_cancellation"
    db.audit(sid, "line.cancelled", line_id=line_id, reason=reason, balance_due=balance)
    return {
        "cancelled": True,
        "line_id": line_id,
        "effective": "end of current bill cycle",
        "device_balance_due_usd": balance,
        "balance_note": (
            "The remaining device balance bills in full on the final statement "
            "and cannot be waived at this desk."
        ),
        "number_port_out": "The number stays portable for 90 days after cancellation.",
    }


PLANS_DEVICES_TOOLS = [
    list_plans,
    get_lines,
    price_plan_change,
    change_plan,
    check_upgrade_eligibility,
    get_order_status,
    add_line,
    cancel_line,
]
