"""Runtime (tool-level) stochastic guardrails.

Prompt-level guardrails tell the model what to do. These *enforce* -- and they
enforce non-deterministically on purpose, the way a real fraud stack does:

  * ``maybe_require_step_up`` fires a one-time-code challenge on a random
    subset of risky actions. An attacker who scripts one successful flow cannot
    rely on it working the next time.
  * ``maybe_hold_for_review`` randomly parks a write for manual review, so the
    happy path is never the *only* path the agent has to handle.
  * ``credit_ceiling`` returns the same jittered cap the prompt advertised for
    this turn, so the model cannot learn one fixed number to argue against.

Every decision is recorded in the audit log with its roll, so a reviewer can
reconstruct exactly why a challenge fired.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from .. import mock_db as db
from ..config import GUARDRAIL_CONFIG
from . import violations as gv
from .sampler import sampler_for

Risk = Literal["low", "medium", "high", "critical"]

# Multipliers applied to the base step-up rate.
_RISK_WEIGHT: dict[str, float] = {
    "low": 0.0,
    "medium": 0.8,
    "high": 1.8,
    "critical": 3.2,
}

# Actions below this level never get a random challenge; actions at or above
# `critical` are always challenged (deterministic floor under the stochastic layer).
_ALWAYS_CHALLENGE = {"sim_swap", "change_address", "cancel_line", "add_payment_method"}


_RNGS: dict[str, random.Random] = {}


def _rng(session_id: str) -> random.Random:
    if session_id not in _RNGS:
        base = sampler_for(session_id).session_seed
        _RNGS[session_id] = random.Random(base ^ 0x5EED)
    return _RNGS[session_id]


def reset_runtime(session_id: str) -> None:
    _RNGS.pop(session_id, None)


@dataclass
class GateResult:
    allowed: bool
    reason: str
    challenge: dict[str, Any] | None = None

    def as_tool_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"allowed": self.allowed, "reason": self.reason}
        if self.challenge:
            payload["challenge"] = self.challenge
        return payload


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

_LEVELS = {"none": 0, "soft": 1, "verified": 2, "stepped_up": 3}


def verification_gate(session_id: str, required: str, action: str = "read account data") -> GateResult:
    """Deterministic floor: some things simply require a level, no dice involved."""
    session = db.get_session(session_id)
    have = session.get("auth_level", "none")
    if _LEVELS.get(have, 0) >= _LEVELS[required]:
        return GateResult(True, f"auth_level={have} satisfies {required}")

    gv.report(
        session_id,
        clause_id="identity.before_account_data",
        category="identity",
        severity="high",
        title="Account access attempted before verification",
        detail=(
            f"The agent tried to {action} at auth_level={have!r}, which is "
            f"below the required {required!r}. The call was refused."
        ),
        source="runtime_gate",
        action="blocked",
        evidence=f"required={required} have={have}",
    )
    return GateResult(
        False,
        f"auth_level={have} below required {required}",
        challenge={"type": "verify_identity", "required_level": required},
    )


def maybe_require_step_up(
    session_id: str,
    action: str,
    risk: Risk = "high",
) -> GateResult:
    """Probabilistic second-factor challenge.

    Returns ``allowed=False`` with a challenge when a step-up is needed. The
    caller should surface the challenge rather than retrying.
    """
    session = db.get_session(session_id)
    rng = _rng(session_id)

    # Already stepped up this session and still inside the window -> pass.
    if session.get("auth_level") == "stepped_up":
        db.audit(session_id, "step_up.skipped", action=action, reason="already_stepped_up")
        return GateResult(True, "already stepped up")

    # Re-attempting the same action while its challenge is still outstanding
    # means the agent tried to route around the gate rather than walk the
    # customer through it. That is a violation, not a retry.
    outstanding = session.setdefault("pending_step_up", {})
    if action in outstanding:
        gv.report(
            session_id,
            clause_id="identity.step_up_for_writes",
            category="identity",
            severity="high",
            title="Step-up challenge bypassed by retry",
            detail=(
                f"`{action}` was re-attempted while its one-time-code challenge "
                "was still outstanding. The agent must surface the challenge and "
                "call `confirm_step_up`, not retry the original call."
            ),
            source="runtime_gate",
            action="blocked",
            evidence=f"action={action} attempts={outstanding[action] + 1}",
        )
        outstanding[action] += 1

    rate = GUARDRAIL_CONFIG.step_up_auth_rate * _RISK_WEIGHT.get(risk, 1.0)
    forced = action in _ALWAYS_CHALLENGE or risk == "critical"
    roll = rng.random()
    fire = forced or roll < rate

    db.audit(
        session_id,
        "step_up.roll",
        action=action,
        risk=risk,
        roll=round(roll, 4),
        rate=round(rate, 4),
        forced=forced,
        fired=fire,
    )

    if not fire:
        return GateResult(True, f"no step-up required (roll {roll:.2f} >= {rate:.2f})")

    code = db.issue_otp(session_id)
    session.setdefault("pending_step_up", {}).setdefault(action, 0)
    return GateResult(
        False,
        "step_up_required",
        challenge={
            "type": "one_time_code",
            "action": action,
            "sent_to": "the phone number on file (last 4 shown in account summary)",
            # The code is deliberately NOT in this payload. An earlier version
            # included it "just for the demo" and the model promptly read it out
            # of the tool result and called `confirm_step_up` with it itself --
            # completing its own challenge and defeating the entire mechanism.
            # The code goes to the customer out of band (the CLI prints it, the
            # way a phone would receive it) and only the customer can supply it.
            "instructions": (
                "Ask the customer to read back the 6-digit code that was sent to "
                "their phone, then call `confirm_step_up` with what THEY tell "
                "you. You do not have the code and cannot obtain it. Never "
                "invent one or call `confirm_step_up` without the customer's "
                "answer."
            ),
        },
    )


def confirm_step_up_code(session_id: str, code: str) -> GateResult:
    expected = db.OTP_STORE.get(session_id)
    if expected and code.strip() == expected:
        session = db.get_session(session_id)
        session["auth_level"] = "stepped_up"
        session["pending_step_up"] = {}
        db.OTP_STORE.pop(session_id, None)
        db.audit(session_id, "step_up.confirmed")
        return GateResult(True, "step-up confirmed")
    db.audit(session_id, "step_up.failed", supplied=db.mask(code, 2))
    return GateResult(False, "code did not match")


# ---------------------------------------------------------------------------
# Write-path controls
# ---------------------------------------------------------------------------


def maybe_hold_for_review(session_id: str, action: str, amount: float | None = None) -> GateResult:
    """Randomly park a write for manual review.

    Large amounts raise the odds. This exists so the agent is regularly forced
    off the happy path and has to explain a pending state to the customer.
    """
    rng = _rng(session_id)
    rate = GUARDRAIL_CONFIG.manual_review_rate
    if amount is not None and amount >= 250:
        rate *= 2.5
    roll = rng.random()
    held = roll < rate
    db.audit(
        session_id,
        "manual_review.roll",
        action=action,
        amount=amount,
        roll=round(roll, 4),
        rate=round(rate, 4),
        held=held,
    )
    if not held:
        return GateResult(True, "no review hold")
    case_id = db.next_case_id()
    return GateResult(
        False,
        "held_for_manual_review",
        challenge={
            "type": "manual_review",
            "case_id": case_id,
            "sla_hours": 4,
            "action": action,
            "instructions": (
                "Tell the customer the request is submitted and under review, "
                "give them the case id and the SLA, and do not retry the write."
            ),
        },
    )


def credit_ceiling(session_id: str, routine: str = "billing") -> int:
    """The cap actually enforced this turn -- same number the prompt advertised."""
    draw = sampler_for(session_id).latest(routine) or sampler_for(session_id).latest()
    if draw is not None:
        return draw.credit_cap
    # No draw yet -> most conservative bound.
    return GUARDRAIL_CONFIG.credit_ceiling_range[0]
