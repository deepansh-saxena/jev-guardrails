"""Concrete desks. Each is `build_routine` plus its own tool set."""

from __future__ import annotations

from typing import Any

from ..tools.account import ACCOUNT_TOOLS
from ..tools.billing import BILLING_TOOLS
from ..tools.home_internet import HOME_INTERNET_TOOLS
from ..tools.payments import PAYMENT_TOOLS
from ..tools.plans_devices import PLANS_DEVICES_TOOLS
from ..tools.tech_support import TECH_SUPPORT_TOOLS
from .base import build_routine

DESK_TOOLS: dict[str, list[Any]] = {
    "billing": BILLING_TOOLS,
    "payments": PAYMENT_TOOLS,
    "home_internet": HOME_INTERNET_TOOLS,
    "plans_devices": PLANS_DEVICES_TOOLS,
    "tech_support": TECH_SUPPORT_TOOLS,
    "account": ACCOUNT_TOOLS,
}

DESK_NAMES = tuple(DESK_TOOLS)


def build_desk(name: str, model: Any = None):
    if name not in DESK_TOOLS:
        raise KeyError(f"unknown desk {name!r}; known: {DESK_NAMES}")
    return build_routine(name, DESK_TOOLS[name], model=model)


def build_all_desks(model: Any = None) -> dict[str, Any]:
    return {name: build_desk(name, model=model) for name in DESK_NAMES}
