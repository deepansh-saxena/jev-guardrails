"""Deterministic output scanning.

The sampled prompt guardrails tell the model what not to say. This checks
whether it complied, on every assistant turn, before the text reaches the
customer. Anything it catches is recorded as a violation and -- for the PII
classes -- redacted in place rather than merely logged.

These checks are intentionally *not* stochastic. A rule that decides whether a
full card number reaches a customer does not get to roll dice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .. import mock_db as db
from . import violations as v

_DIGIT_RUN = re.compile(r"(?:\d[ -]?){12,19}\d")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_OTP_CONTEXT = re.compile(
    r"(?:code|otp|one[- ]time)\D{0,24}\b(\d{6})\b|\b(\d{6})\b\D{0,24}(?:code|otp)",
    re.I,
)
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")

_PROMPT_LEAK = re.compile(
    r"\b(system prompt|my instructions (?:say|are)|operating rules|"
    r"guardrail|pinned clause|non-negotiable:|hard rule:|"
    r"resampled every turn)\b",
    re.I,
)
_INTERNAL_VOCAB = re.compile(
    r"\b(routine|handoff|hand off to the \w+ desk|tool call|"
    r"transfer_to_\w+|handoff_to_triage|step_up_required|pending_review)\b",
    re.I,
)
_SELF_AUDIT_LEAK = re.compile(
    r"^\s*(rule check|governing rule|audit mode|self-check)\b", re.I | re.M
)
_GUARANTEE = re.compile(
    r"\b(i guarantee|i promise|i can promise|you will definitely|"
    r"i'?ll make sure (?:this|it) never|rest assured (?:this|it) (?:will|won'?t)|"
    r"100% (?:refund|guaranteed))\b",
    re.I,
)
_PRESSURE = re.compile(
    r"\b(act now|limited time|only today|expires? (?:today|in \d+ (?:minutes|hours))|"
    r"last chance|don'?t miss out|before it'?s too late)\b",
    re.I,
)
_ADVICE = re.compile(
    r"\b(you should sue|for tax purposes you (?:can|should)|"
    r"this (?:will|won'?t) (?:hurt|affect) your credit score|"
    r"i'?d recommend (?:filing|disputing) with (?:the|your) (?:FCC|attorney general))\b",
    re.I,
)
_LIVE_HUMAN = re.compile(
    r"\b(i'?m a (?:real )?(?:person|human)|speaking with a human|"
    r"i am not (?:a bot|an ai))\b",
    re.I,
)


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _snippet(text: str, match: re.Match, width: int = 36) -> str:
    start = max(0, match.start() - width // 2)
    end = min(len(text), match.end() + width // 2)
    raw = text[start:end].replace("\n", " ")
    # never put the offending value itself into the log
    return re.sub(r"\d", "#", raw).strip()


@dataclass
class Check:
    clause_id: str
    category: str
    severity: str
    title: str
    detail: str
    action: str
    pattern: re.Pattern | None = None
    predicate: Callable[[str, str], re.Match | None] | None = None
    redact_with: str | None = None


def _card_predicate(text: str, _session: str) -> re.Match | None:
    for m in _DIGIT_RUN.finditer(text):
        if _luhn(m.group()):
            return m
    return None


def _otp_predicate(text: str, session_id: str) -> re.Match | None:
    # Only a violation when a one-time code is actually outstanding.
    if session_id not in db.OTP_STORE:
        return None
    return _OTP_CONTEXT.search(text)


def _unverified_money(text: str, session_id: str) -> re.Match | None:
    level = db.get_session(session_id).get("auth_level", "none")
    if level in {"verified", "stepped_up"}:
        return None
    return _MONEY.search(text)


CHECKS: list[Check] = [
    Check(
        clause_id="pii.no_full_identifiers",
        category="pii",
        severity="critical",
        title="Full payment card number in the reply",
        detail="The draft reply contained a Luhn-valid card number. Only the last four digits may ever be stated.",
        action="redacted",
        predicate=_card_predicate,
        redact_with="[card redacted]",
    ),
    Check(
        clause_id="pii.no_full_identifiers",
        category="pii",
        severity="critical",
        title="Social security number in the reply",
        detail="An SSN-shaped value reached the draft reply. SSNs are never stated, in full or in part beyond the last four.",
        action="redacted",
        pattern=_SSN,
        redact_with="[ssn redacted]",
    ),
    Check(
        clause_id="pii.no_full_identifiers",
        category="pii",
        severity="critical",
        title="One-time code read back to the customer",
        detail="A step-up code is outstanding and the reply appears to state it. The agent must never speak the code.",
        action="redacted",
        predicate=_otp_predicate,
        redact_with="[code redacted]",
    ),
    Check(
        clause_id="identity.before_account_data",
        category="identity",
        severity="high",
        title="Account amounts disclosed before verification",
        detail="A dollar amount appeared in the reply while the session is not verified. Account figures are sealed until identity is proven.",
        action="flagged",
        predicate=_unverified_money,
    ),
    Check(
        clause_id="injection.no_prompt_disclosure",
        category="injection",
        severity="high",
        title="Internal instructions disclosed",
        detail="The reply referenced the system prompt or the guardrail machinery. Policy is summarised in plain language, never quoted.",
        action="flagged",
        pattern=_PROMPT_LEAK,
    ),
    Check(
        clause_id="tone.no_internal_vocabulary",
        category="tone",
        severity="medium",
        title="Internal vocabulary leaked to the customer",
        detail="Words like 'routine', 'handoff' or a raw tool name appeared in the reply. From the customer's side there is only one assistant.",
        action="flagged",
        pattern=_INTERNAL_VOCAB,
    ),
    Check(
        clause_id="accuracy.self_audit_stays_internal",
        category="accuracy",
        severity="medium",
        title="Self-audit reasoning leaked into the reply",
        detail="The turn's self-check directive was written out to the customer instead of being applied silently.",
        action="flagged",
        pattern=_SELF_AUDIT_LEAK,
    ),
    Check(
        clause_id="authority.no_promises",
        category="authority",
        severity="high",
        title="Unbacked guarantee made to the customer",
        detail="The reply promised an outcome the agent cannot execute with a tool.",
        action="flagged",
        pattern=_GUARANTEE,
    ),
    Check(
        clause_id="tone.no_dark_patterns",
        category="tone",
        severity="high",
        title="Pressure or urgency tactic in the reply",
        detail="Scarcity or urgency language was used to push a decision.",
        action="flagged",
        pattern=_PRESSURE,
    ),
    Check(
        clause_id="compliance.no_advice",
        category="compliance",
        severity="high",
        title="Legal, tax or credit advice given",
        detail="The reply went beyond carrier policy into advice the agent is not permitted to give.",
        action="flagged",
        pattern=_ADVICE,
    ),
    Check(
        clause_id="compliance.recording_notice",
        category="compliance",
        severity="critical",
        title="Agent claimed to be human",
        detail="The reply implied a human was speaking. The assistant must always identify as automated when asked.",
        action="flagged",
        pattern=_LIVE_HUMAN,
    ),
]


def scan(text: str, session_id: str, routine: str) -> tuple[str, list[v.Violation]]:
    """Check a draft reply. Returns (possibly redacted text, violations found)."""
    if not text or not text.strip():
        return text, []

    found: list[v.Violation] = []
    cleaned = text

    for check in CHECKS:
        match = (
            check.predicate(cleaned, session_id)
            if check.predicate is not None
            else (check.pattern.search(cleaned) if check.pattern else None)
        )
        if match is None:
            continue

        found.append(
            v.report(
                session_id,
                clause_id=check.clause_id,
                category=check.category,
                severity=check.severity,  # type: ignore[arg-type]
                title=check.title,
                detail=check.detail,
                source="output_scan",
                action=check.action,  # type: ignore[arg-type]
                routine=routine,
                evidence=_snippet(cleaned, match),
            )
        )

        if check.redact_with:
            if check.predicate is _card_predicate:
                cleaned = _DIGIT_RUN.sub(
                    lambda m: check.redact_with if _luhn(m.group()) else m.group(),
                    cleaned,
                )
            elif check.pattern is not None:
                cleaned = check.pattern.sub(check.redact_with, cleaned)
            else:
                cleaned = cleaned[: match.start()] + check.redact_with + cleaned[match.end():]

    return cleaned, found
