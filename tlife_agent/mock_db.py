"""In-memory mock of the carrier back office.

Everything a real deployment would reach over HTTP (CRM, billing, provisioning,
payment vault, ticketing) lives here as plain dicts so the whole agent runs with
no external dependency. Tools read and write through the small accessor
functions at the bottom -- swap those for real clients and the agents are
unchanged.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import random
from typing import Any

_TODAY = _dt.date(2026, 9, 21)


def _d(offset: int) -> str:
    return (_TODAY + _dt.timedelta(days=offset)).isoformat()


# --------------------------------------------------------------------------
# Customers
# --------------------------------------------------------------------------

CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUS-100234": {
        "customer_id": "CUS-100234",
        "name": "Dana Whitfield",
        "phone": "+1-312-555-0147",
        "email": "d.whitfield@example.com",
        "address": "1411 W Wabash Ave, Chicago, IL 60605",
        "last4_ssn": "4417",
        "account_pin_hash": "pin:8823",
        "tenure_months": 61,
        "segment": "Magenta Status",
        "autopay": True,
        "paperless": True,
        "credit_class": "A",
        "open_disputes": 0,
        "language": "en",
        "accessibility_flags": [],
        "lifetime_credits_usd": 45.00,
        "fraud_watch": False,
    },
    "CUS-100987": {
        "customer_id": "CUS-100987",
        "name": "Marcus Oyelaran",
        "phone": "+1-206-555-0192",
        "email": "m.oye@example.com",
        "address": "88 Pike St Apt 5C, Seattle, WA 98101",
        "last4_ssn": "9021",
        "account_pin_hash": "pin:1190",
        "tenure_months": 7,
        "segment": "Essentials",
        "autopay": False,
        "paperless": False,
        "credit_class": "C",
        "open_disputes": 1,
        "language": "en",
        "accessibility_flags": ["prefers_large_print"],
        "lifetime_credits_usd": 120.00,
        "fraud_watch": True,
    },
    "CUS-100555": {
        "customer_id": "CUS-100555",
        "name": "Priya Raghunathan",
        "phone": "+1-469-555-0110",
        "email": "praghu@example.com",
        "address": "902 Elm St, Dallas, TX 75201",
        "last4_ssn": "3388",
        "account_pin_hash": "pin:4402",
        "tenure_months": 29,
        "segment": "Go5G Plus",
        "autopay": True,
        "paperless": True,
        "credit_class": "A",
        "open_disputes": 0,
        "language": "en",
        "accessibility_flags": [],
        "lifetime_credits_usd": 0.00,
        "fraud_watch": False,
    },
}

# Alternate lookup keys -> customer_id
PHONE_INDEX = {c["phone"]: cid for cid, c in CUSTOMERS.items()}
EMAIL_INDEX = {c["email"]: cid for cid, c in CUSTOMERS.items()}


# --------------------------------------------------------------------------
# Mobile lines & plans
# --------------------------------------------------------------------------

PLANS: dict[str, dict[str, Any]] = {
    "GO5G_NEXT": {
        "plan_id": "GO5G_NEXT",
        "name": "Go5G Next",
        "price_per_line": 100.00,
        "data": "Unlimited premium",
        "hotspot_gb": 50,
        "perks": ["Netflix On Us", "Apple TV+ On Us", "In-flight Wi-Fi"],
        "upgrade_cadence_years": 1,
    },
    "GO5G_PLUS": {
        "plan_id": "GO5G_PLUS",
        "name": "Go5G Plus",
        "price_per_line": 90.00,
        "data": "Unlimited premium",
        "hotspot_gb": 50,
        "perks": ["Netflix On Us", "Apple TV+ On Us"],
        "upgrade_cadence_years": 2,
    },
    "ESSENTIALS": {
        "plan_id": "ESSENTIALS",
        "name": "Essentials",
        "price_per_line": 60.00,
        "data": "Unlimited (may slow during congestion)",
        "hotspot_gb": 0,
        "perks": [],
        "upgrade_cadence_years": None,
    },
    "CONNECT_55": {
        "plan_id": "CONNECT_55",
        "name": "Connect 55+",
        "price_per_line": 40.00,
        "data": "5GB",
        "hotspot_gb": 0,
        "perks": [],
        "upgrade_cadence_years": None,
        "eligibility": "Primary account holder must be 55+",
    },
}

LINES: dict[str, list[dict[str, Any]]] = {
    "CUS-100234": [
        {
            "msisdn": "+1-312-555-0147",
            "line_id": "LN-8811",
            "owner": "Dana Whitfield",
            "plan_id": "GO5G_PLUS",
            "device": "iPhone 17 Pro 256GB",
            "device_balance": 412.50,
            "device_months_left": 15,
            "status": "active",
            "insurance": "Protection<360>",
            "data_used_gb": 41.2,
        },
        {
            "msisdn": "+1-312-555-0188",
            "line_id": "LN-8812",
            "owner": "Jules Whitfield",
            "plan_id": "GO5G_PLUS",
            "device": "Galaxy S26",
            "device_balance": 0.00,
            "device_months_left": 0,
            "status": "active",
            "insurance": None,
            "data_used_gb": 12.9,
        },
        {
            "msisdn": "+1-312-555-0203",
            "line_id": "LN-8813",
            "owner": "watch",
            "plan_id": "ESSENTIALS",
            "device": "Apple Watch Ultra 3",
            "device_balance": 96.00,
            "device_months_left": 8,
            "status": "suspended",
            "insurance": None,
            "data_used_gb": 0.4,
        },
    ],
    "CUS-100987": [
        {
            "msisdn": "+1-206-555-0192",
            "line_id": "LN-4410",
            "owner": "Marcus Oyelaran",
            "plan_id": "ESSENTIALS",
            "device": "Pixel 11",
            "device_balance": 689.00,
            "device_months_left": 22,
            "status": "active",
            "insurance": None,
            "data_used_gb": 88.6,
        },
    ],
    "CUS-100555": [
        {
            "msisdn": "+1-469-555-0110",
            "line_id": "LN-2201",
            "owner": "Priya Raghunathan",
            "plan_id": "GO5G_PLUS",
            "device": "iPhone 16",
            "device_balance": 0.00,
            "device_months_left": 0,
            "status": "active",
            "insurance": "Protection<360>",
            "data_used_gb": 22.0,
        },
        {
            "msisdn": "+1-469-555-0111",
            "line_id": "LN-2202",
            "owner": "Arun Raghunathan",
            "plan_id": "GO5G_PLUS",
            "device": "iPhone 16",
            "device_balance": 0.00,
            "device_months_left": 0,
            "status": "active",
            "insurance": None,
            "data_used_gb": 9.5,
        },
    ],
}


# --------------------------------------------------------------------------
# Billing
# --------------------------------------------------------------------------

BILLS: dict[str, list[dict[str, Any]]] = {
    "CUS-100234": [
        {
            "bill_id": "BILL-2026-09-100234",
            "period": "2026-08-14 to 2026-09-13",
            "due_date": _d(9),
            "total": 247.83,
            "status": "open",
            "line_items": [
                {"desc": "Go5G Plus x2", "amount": 180.00},
                {"desc": "Essentials (watch)", "amount": 12.00},
                {"desc": "Device payment LN-8811", "amount": 27.50},
                {"desc": "Device payment LN-8813", "amount": 12.00},
                {"desc": "Protection<360> LN-8811", "amount": 18.00},
                {"desc": "International day pass (3 days)", "amount": 15.00},
                {"desc": "Taxes & regulatory fees", "amount": 18.33},
                {"desc": "Autopay discount", "amount": -35.00},
            ],
        },
        {
            "bill_id": "BILL-2026-08-100234",
            "period": "2026-07-14 to 2026-08-13",
            "due_date": _d(-21),
            "total": 212.83,
            "status": "paid",
            "line_items": [
                {"desc": "Go5G Plus x2", "amount": 180.00},
                {"desc": "Essentials (watch)", "amount": 12.00},
                {"desc": "Device payment LN-8811", "amount": 27.50},
                {"desc": "Device payment LN-8813", "amount": 12.00},
                {"desc": "Protection<360> LN-8811", "amount": 18.00},
                {"desc": "Taxes & regulatory fees", "amount": 18.33},
                {"desc": "Autopay discount", "amount": -35.00},
            ],
        },
    ],
    "CUS-100987": [
        {
            "bill_id": "BILL-2026-09-100987",
            "period": "2026-08-20 to 2026-09-19",
            "due_date": _d(-4),
            "total": 168.44,
            "status": "past_due",
            "line_items": [
                {"desc": "Essentials x1", "amount": 60.00},
                {"desc": "Device payment LN-4410", "amount": 31.32},
                {"desc": "Late fee", "amount": 7.00},
                {"desc": "Restore fee", "amount": 20.00},
                {"desc": "Previous balance", "amount": 41.12},
                {"desc": "Taxes & regulatory fees", "amount": 9.00},
            ],
        },
    ],
    "CUS-100555": [
        {
            "bill_id": "BILL-2026-09-100555",
            "period": "2026-08-02 to 2026-09-01",
            "due_date": _d(3),
            "total": 165.00,
            "status": "open",
            "line_items": [
                {"desc": "Go5G Plus x2", "amount": 180.00},
                {"desc": "Autopay discount", "amount": -10.00},
                {"desc": "Taxes & regulatory fees", "amount": 15.00},
                {"desc": "Loyalty credit", "amount": -20.00},
            ],
        },
    ],
}

PAYMENT_METHODS: dict[str, list[dict[str, Any]]] = {
    "CUS-100234": [
        {"method_id": "PM-01", "type": "visa", "last4": "4242", "exp": "11/28", "default": True},
        {"method_id": "PM-02", "type": "checking", "last4": "8891", "exp": None, "default": False},
    ],
    "CUS-100987": [
        {"method_id": "PM-11", "type": "mastercard", "last4": "5510", "exp": "02/27", "default": True},
    ],
    "CUS-100555": [
        {"method_id": "PM-21", "type": "amex", "last4": "1007", "exp": "07/29", "default": True},
    ],
}

PAYMENT_HISTORY: dict[str, list[dict[str, Any]]] = {
    "CUS-100234": [
        {"payment_id": "PAY-9001", "date": _d(-21), "amount": 212.83, "method_id": "PM-01", "status": "posted"},
        {"payment_id": "PAY-8804", "date": _d(-52), "amount": 212.83, "method_id": "PM-01", "status": "posted"},
    ],
    "CUS-100987": [
        {"payment_id": "PAY-7711", "date": _d(-46), "amount": 60.00, "method_id": "PM-11", "status": "posted"},
        {"payment_id": "PAY-7802", "date": _d(-16), "amount": 50.00, "method_id": "PM-11", "status": "returned_nsf"},
    ],
    "CUS-100555": [
        {"payment_id": "PAY-6600", "date": _d(-30), "amount": 165.00, "method_id": "PM-21", "status": "posted"},
    ],
}

PAYMENT_ARRANGEMENTS: dict[str, list[dict[str, Any]]] = {}


# --------------------------------------------------------------------------
# Home Internet
# --------------------------------------------------------------------------

HOME_INTERNET: dict[str, dict[str, Any]] = {
    "CUS-100234": {
        "service_id": "HI-5512",
        "plan": "Rely Home Internet",
        "monthly": 50.00,
        "gateway": "G4AR (5G)",
        "gateway_serial": "G4AR-77120934",
        "status": "active",
        "install_date": _d(-400),
        "signal_bars": 2,
        "band": "n41",
        "downlink_mbps": 41.0,
        "uplink_mbps": 9.2,
        "outage": None,
        "last_reboot": _d(-31),
        "known_issues": ["gateway placed in basement", "firmware 2 versions behind"],
    },
    "CUS-100555": {
        "service_id": "HI-9930",
        "plan": "Amplified Home Internet",
        "monthly": 60.00,
        "gateway": "G5AR (5G)",
        "gateway_serial": "G5AR-22110087",
        "status": "active",
        "install_date": _d(-120),
        "signal_bars": 4,
        "band": "n25",
        "downlink_mbps": 318.0,
        "uplink_mbps": 41.0,
        "outage": {
            "outage_id": "OUT-DAL-4471",
            "area": "Dallas 75201",
            "started": _d(0),
            "eta_hours": 6,
            "cause": "Fiber cut at aggregation site",
        },
        "last_reboot": _d(-2),
        "known_issues": [],
    },
}

HOME_INTERNET_AVAILABILITY: dict[str, dict[str, Any]] = {
    "98101": {"available": False, "reason": "Capacity full on serving sector", "waitlist": True},
    "60605": {"available": True, "plans": ["Rely Home Internet", "Amplified Home Internet"], "install": "self-install, 2-day ship"},
    "75201": {"available": True, "plans": ["Rely Home Internet", "Amplified Home Internet", "All-In Home Internet"], "install": "self-install, next-day ship"},
}


# --------------------------------------------------------------------------
# Network / device diagnostics
# --------------------------------------------------------------------------

NETWORK_STATUS: dict[str, dict[str, Any]] = {
    "60605": {"status": "normal", "towers_degraded": 0, "note": None},
    "98101": {"status": "degraded", "towers_degraded": 2, "note": "Maintenance window 22:00-04:00 local"},
    "75201": {"status": "outage", "towers_degraded": 5, "note": "Fiber cut OUT-DAL-4471, ETA 6h"},
}


# --------------------------------------------------------------------------
# Mutable session + ticket stores
# --------------------------------------------------------------------------

_TICKET_SEQ = itertools.count(7701)
_CASE_SEQ = itertools.count(3301)

TICKETS: list[dict[str, Any]] = []
AUDIT_LOG: list[dict[str, Any]] = []
OTP_STORE: dict[str, str] = {}

# session_id -> mutable session facts (auth level, verified customer, holds...)
SESSIONS: dict[str, dict[str, Any]] = {}


def get_session(session_id: str) -> dict[str, Any]:
    """Per-conversation scratch state. Auth level lives here, not in the prompt."""
    return SESSIONS.setdefault(
        session_id,
        {
            "session_id": session_id,
            "customer_id": None,
            # none -> soft (name/phone matched) -> verified (PIN) -> stepped_up (OTP)
            "auth_level": "none",
            "step_up_until_turn": None,
            "turn": 0,
            "handoffs": [],
            "actions_taken": [],
            "credits_issued_usd": 0.0,
            "guardrail_trace": [],
        },
    )


def reset_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)


def next_ticket_id() -> str:
    return f"TKT-{next(_TICKET_SEQ)}"


def next_case_id() -> str:
    return f"CASE-{next(_CASE_SEQ)}"


def audit(session_id: str, event: str, **details: Any) -> None:
    """Every state-changing tool writes here. Real deployments ship this to SIEM."""
    AUDIT_LOG.append(
        {
            "session_id": session_id,
            "event": event,
            "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            **details,
        }
    )


def find_customer(identifier: str) -> dict[str, Any] | None:
    ident = identifier.strip()
    if ident in CUSTOMERS:
        return CUSTOMERS[ident]
    if ident in PHONE_INDEX:
        return CUSTOMERS[PHONE_INDEX[ident]]
    if ident in EMAIL_INDEX:
        return CUSTOMERS[EMAIL_INDEX[ident]]
    # tolerate loosely formatted phone numbers
    digits = "".join(ch for ch in ident if ch.isdigit())
    if len(digits) >= 10:
        for phone, cid in PHONE_INDEX.items():
            if "".join(ch for ch in phone if ch.isdigit()).endswith(digits[-10:]):
                return CUSTOMERS[cid]
    lowered = ident.lower()
    for cust in CUSTOMERS.values():
        if cust["name"].lower() == lowered:
            return cust
    return None


def mask(value: str, keep: int = 4) -> str:
    """Never hand a full identifier back to the model or the customer."""
    if not value:
        return value
    tail = value[-keep:]
    return f"{'*' * max(0, len(value) - keep)}{tail}"


def issue_otp(session_id: str) -> str:
    code = f"{random.randint(0, 999999):06d}"
    OTP_STORE[session_id] = code
    return code
