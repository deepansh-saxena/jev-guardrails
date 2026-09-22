"""Home Internet desk: availability, gateway diagnostics, outages, truck rolls."""

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
def check_home_internet_availability(zip_code: str) -> dict[str, Any]:
    """Check whether Home Internet can be sold at a ZIP code. No verification needed."""
    info = db.HOME_INTERNET_AVAILABILITY.get(zip_code.strip())
    if info is None:
        return {
            "zip": zip_code,
            "known": False,
            "message": "No serving data for that ZIP. Offer to check again later or take a callback.",
        }
    return {"zip": zip_code, "known": True, **info}


@tool
def get_home_internet_status(runtime: ToolRuntime) -> dict[str, Any]:
    """Current Home Internet service state: plan, gateway, signal, throughput, outage flag."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    svc = db.HOME_INTERNET.get(cid)
    if not svc:
        return {"has_service": False, "message": "No Home Internet on this account."}
    return {
        "has_service": True,
        "service_id": svc["service_id"],
        "plan": svc["plan"],
        "monthly_usd": svc["monthly"],
        "gateway": svc["gateway"],
        "gateway_serial_last4": svc["gateway_serial"][-4:],
        "status": svc["status"],
        "signal_bars": svc["signal_bars"],
        "band": svc["band"],
        "downlink_mbps": svc["downlink_mbps"],
        "uplink_mbps": svc["uplink_mbps"],
        "last_reboot": svc["last_reboot"],
        "active_outage": svc["outage"],
    }


@tool
def run_gateway_diagnostics(runtime: ToolRuntime) -> dict[str, Any]:
    """Run a live diagnostic against the gateway and return findings plus ranked fixes."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    svc = db.HOME_INTERNET.get(cid)
    if not svc:
        return {"ran": False, "message": "No Home Internet on this account."}

    findings, fixes = [], []
    if svc["outage"]:
        findings.append(f"Area outage {svc['outage']['outage_id']}: {svc['outage']['cause']}")
        fixes.append("No customer-side fix. Quote the outage ETA exactly as returned.")
    if svc["signal_bars"] <= 2:
        findings.append(f"Weak signal: {svc['signal_bars']}/5 bars on {svc['band']}")
        fixes.append("Relocate the gateway to an upper floor near an exterior window.")
    if "firmware" in " ".join(svc["known_issues"]):
        findings.append("Gateway firmware is behind by 2 releases")
        fixes.append("Push a firmware update via reboot_gateway (applies on restart).")
    if svc["downlink_mbps"] < 100 and not svc["outage"]:
        findings.append(f"Downlink {svc['downlink_mbps']} Mbps is below the expected range for {svc['plan']}")
    if not findings:
        findings.append("No faults detected; service is performing within spec.")

    db.audit(sid, "home_internet.diagnostics", service_id=svc["service_id"])
    return {
        "ran": True,
        "service_id": svc["service_id"],
        "findings": findings,
        "recommended_fixes": fixes,
        "expected_range_for_plan": "Rely 35-200 Mbps / Amplified 100-400 Mbps",
    }


@tool
def reboot_gateway(runtime: ToolRuntime) -> dict[str, Any]:
    """Remotely restart the gateway. Warn the customer it drops service for ~3 minutes first."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    svc = db.HOME_INTERNET.get(cid)
    if not svc:
        return {"rebooted": False, "message": "No Home Internet on this account."}
    if svc["outage"]:
        return {
            "rebooted": False,
            "reason": "active_area_outage",
            "outage": svc["outage"],
            "note": "A reboot will not help during an area outage. Do not suggest it.",
        }
    svc["last_reboot"] = "2026-09-21"
    svc["signal_bars"] = min(5, svc["signal_bars"] + 1)
    svc["downlink_mbps"] = round(svc["downlink_mbps"] * 1.15, 1)
    db.audit(sid, "home_internet.reboot", service_id=svc["service_id"])
    return {
        "rebooted": True,
        "downtime_minutes": 3,
        "post_reboot_signal_bars": svc["signal_bars"],
        "post_reboot_downlink_mbps": svc["downlink_mbps"],
    }


@tool
def check_area_outage(zip_code: str) -> dict[str, Any]:
    """Check for a known network or Home Internet outage at a ZIP code."""
    status = db.NETWORK_STATUS.get(zip_code.strip())
    if status is None:
        return {"zip": zip_code, "known": False}
    outage = next(
        (svc["outage"] for svc in db.HOME_INTERNET.values()
         if svc["outage"] and zip_code.strip() in svc["outage"]["area"]),
        None,
    )
    return {"zip": zip_code, "known": True, **status, "outage": outage}


@tool
def schedule_technician(
    reason: str,
    preferred_window: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Book a technician visit. `preferred_window` e.g. '2026-09-24 morning'.

    Only book after diagnostics have ruled out an outage and a reboot.
    """
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    hold = gr.maybe_hold_for_review(sid, "schedule_technician")
    if not hold.allowed:
        return {"scheduled": False, "status": "pending_review", **hold.as_tool_payload()}
    ticket = db.next_ticket_id()
    db.TICKETS.append({"ticket_id": ticket, "customer_id": cid, "subject": "Technician visit",
                       "details": reason, "priority": "normal", "status": "scheduled"})
    db.audit(sid, "home_internet.tech_scheduled", ticket_id=ticket, window=preferred_window)
    return {
        "scheduled": True,
        "ticket_id": ticket,
        "window": preferred_window,
        "dispatch_fee_usd": 0.0,
        "note": "No dispatch fee when the fault is on the network side.",
    }


@tool
def swap_home_internet_plan(new_plan: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Change the Home Internet plan. Valid: 'Rely Home Internet', 'Amplified Home Internet', 'All-In Home Internet'."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    svc = db.HOME_INTERNET.get(cid)
    if not svc:
        return {"changed": False, "message": "No Home Internet on this account."}
    prices = {"Rely Home Internet": 50.00, "Amplified Home Internet": 60.00, "All-In Home Internet": 70.00}
    if new_plan not in prices:
        return {"changed": False, "valid_plans": sorted(prices)}

    gate = gr.maybe_require_step_up(sid, "swap_home_internet_plan", "medium")
    if not gate.allowed:
        return {"changed": False, "status": "step_up_required", **gate.as_tool_payload()}

    old_price, old_plan = svc["monthly"], svc["plan"]
    svc["plan"], svc["monthly"] = new_plan, prices[new_plan]
    db.audit(sid, "home_internet.plan_changed", old=old_plan, new=new_plan)
    return {
        "changed": True,
        "from_plan": old_plan,
        "to_plan": new_plan,
        "monthly_delta_usd": round(prices[new_plan] - old_price, 2),
        "effective": "next bill cycle, prorated to the day",
    }


HOME_INTERNET_TOOLS = [
    check_home_internet_availability,
    get_home_internet_status,
    run_gateway_diagnostics,
    reboot_gateway,
    check_area_outage,
    schedule_technician,
    swap_home_internet_plan,
]
