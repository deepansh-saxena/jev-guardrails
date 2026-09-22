"""Network and device technical support desk."""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool

from .. import mock_db as db
from ..guardrails import runtime as gr
from ..guardrails import violations as gv
from ._session import session_id


def _verified(sid: str) -> tuple[str | None, dict[str, Any] | None]:
    gate = gr.verification_gate(sid, "verified")
    if not gate.allowed:
        return None, gate.as_tool_payload()
    return db.get_session(sid)["customer_id"], None


@tool
def check_network_status(zip_code: str) -> dict[str, Any]:
    """Network health at a ZIP code: normal, degraded, or outage. No verification needed."""
    status = db.NETWORK_STATUS.get(zip_code.strip())
    if status is None:
        return {"zip": zip_code, "known": False,
                "message": "No status data for that ZIP. Do not assume it is normal."}
    return {"zip": zip_code, "known": True, **status}


@tool
def run_line_diagnostics(line_id: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Run provisioning and registration checks against a line."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    line = next((l for l in db.LINES.get(cid, []) if l["line_id"] == line_id), None)
    if not line:
        return {"ran": False, "reason": "unknown_line_id"}

    issues, fixes = [], []
    if line["status"] == "suspended":
        issues.append("Line is suspended; no service will register.")
        fixes.append("Resolve the suspension reason (billing or customer request) first.")
    if line["data_used_gb"] > 50 and db.PLANS[line["plan_id"]]["hotspot_gb"] == 0:
        issues.append("Heavy data use on a plan without premium data; deprioritization is likely during congestion.")
        fixes.append("A plan with premium data would remove deprioritization.")
    if not issues:
        issues.append("Provisioning, VoLTE and IMS registration all look correct.")
        fixes.append("If the customer still has trouble, it is device- or location-side.")

    db.audit(sid, "line.diagnostics", line_id=line_id)
    return {
        "ran": True,
        "line_id": line_id,
        "status": line["status"],
        "volte_provisioned": line["status"] == "active",
        "device": line["device"],
        "issues": issues,
        "recommended_fixes": fixes,
    }


@tool
def refresh_line_provisioning(line_id: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Push a provisioning refresh to a line. Tell the customer to restart the device after."""
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    line = next((l for l in db.LINES.get(cid, []) if l["line_id"] == line_id), None)
    if not line:
        return {"sent": False, "reason": "unknown_line_id"}
    db.audit(sid, "line.provisioning_refresh", line_id=line_id)
    return {
        "sent": True,
        "line_id": line_id,
        "apply_time_minutes": 5,
        "customer_action": "Restart the device once, then retest.",
    }


@tool
def get_troubleshooting_steps(symptom: str) -> dict[str, Any]:
    """Standard troubleshooting script for a symptom.

    Symptoms: no_service, slow_data, dropped_calls, no_sms, wifi_calling, esim_activation, voicemail.
    """
    scripts = {
        "no_service": [
            "Toggle airplane mode on for 10 seconds, then off.",
            "Confirm the line is active and not suspended.",
            "Restart the device.",
            "Reseat the physical SIM or re-download the eSIM profile.",
        ],
        "slow_data": [
            "Check for an area outage or maintenance window first.",
            "Confirm the plan's data tier and whether deprioritization applies.",
            "Test in a second location to separate device from network.",
            "Reset network settings on the device.",
        ],
        "dropped_calls": [
            "Enable Wi-Fi Calling if the customer is indoors.",
            "Check for degraded towers in the area.",
            "Update carrier settings on the device.",
        ],
        "no_sms": [
            "Verify the messaging app is set as default.",
            "Confirm no number is blocked.",
            "Push a provisioning refresh, then restart.",
        ],
        "wifi_calling": [
            "Confirm an E911 address is registered -- required before Wi-Fi Calling can be enabled.",
            "Enable Wi-Fi Calling in device settings.",
            "Restart the device and test a call.",
        ],
        "esim_activation": [
            "Confirm the device is carrier-unlocked and eSIM capable.",
            "Connect to Wi-Fi before starting the transfer.",
            "Use the eSIM transfer flow in the app; do not remove the old SIM until the new profile is active.",
        ],
        "voicemail": [
            "Dial the voicemail number and check for a setup prompt.",
            "Push a provisioning refresh if the mailbox is not provisioned.",
            "Reset the voicemail password through the app, never in chat.",
        ],
    }
    key = symptom.strip().lower().replace(" ", "_")
    if key not in scripts:
        return {"found": False, "available_symptoms": sorted(scripts)}
    return {"found": True, "symptom": key, "steps": scripts[key]}


@tool
def start_esim_transfer(line_id: str, runtime: ToolRuntime) -> dict[str, Any]:
    """Begin an eSIM transfer to a new device.

    This is a SIM-change operation: it is fraud-sensitive, always step-up gated,
    and blocked outright on fraud-flagged accounts.
    """
    sid = session_id(runtime)
    cid, err = _verified(sid)
    if err:
        return err
    if db.CUSTOMERS[cid]["fraud_watch"]:
        db.audit(sid, "esim.blocked_fraud_watch", line_id=line_id)
        gv.report(
            sid,
            clause_id="escalation.fraud_watch",
            category="escalation",
            severity="critical",
            title="SIM transfer attempted on a fraud-flagged account",
            detail=(
                "This account carries an active fraud watch, which makes it "
                "read-only for SIM/eSIM operations. The transfer was refused "
                "outright; the only correct next step is the fraud desk."
            ),
            source="runtime_gate",
            action="blocked",
            routine="tech_support",
            evidence=f"line_id={line_id} fraud_watch=True",
        )
        return {
            "started": False,
            "reason": "fraud_watch_active",
            "next_step": "Escalate to the fraud desk. Do not attempt a workaround.",
        }
    gate = gr.maybe_require_step_up(sid, "sim_swap", "critical")
    if not gate.allowed:
        return {"started": False, "status": "step_up_required", **gate.as_tool_payload()}

    db.audit(sid, "esim.transfer_started", line_id=line_id)
    return {
        "started": True,
        "line_id": line_id,
        "qr_delivery": "Emailed to the address on file",
        "expires_hours": 24,
        "warning": "The old device loses service once the new profile activates.",
    }


TECH_SUPPORT_TOOLS = [
    check_network_status,
    run_line_diagnostics,
    refresh_line_provisioning,
    get_troubleshooting_steps,
    start_esim_transfer,
]
