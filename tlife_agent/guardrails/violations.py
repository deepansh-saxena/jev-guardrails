"""Guardrail violation detection and reporting.

A *violation* is any point where a guardrail actually bit: the agent tried
something a rule forbids, or its draft output tripped a content check. Each one
is recorded with the clause it maps to, what was attempted, the evidence
(redacted), and what the system did about it.

Two sources feed the registry:

  * **runtime gates** (`guardrails.runtime`, and the tool modules) -- the agent
    attempted an action above its authority, before verification, on a
    fraud-flagged account, or without clearing a step-up challenge;
  * **output scanning** (`guardrails.output_scan`, run as `after_model`
    middleware) -- the draft reply contained a full card number, an SSN, an OTP,
    a leaked system instruction, an unbacked guarantee, and so on.

The CLI drains the registry after each turn and prints a banner, so a violation
is visible in the conversation rather than buried in a log.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .. import mock_db as db

Severity = Literal["critical", "high", "medium", "low"]
Source = Literal["runtime_gate", "output_scan", "tool_policy", "scope"]
Action = Literal["blocked", "redacted", "flagged", "challenged", "held"]

_SEQ = itertools.count(1)


@dataclass
class Violation:
    clause_id: str
    category: str
    severity: Severity
    title: str
    detail: str
    source: Source
    action: Action
    routine: str = "-"
    evidence: str = ""
    session_id: str = "-"
    violation_id: str = field(default_factory=lambda: f"GV-{next(_SEQ):04d}")
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# session_id -> violations recorded this session
_REGISTRY: dict[str, list[Violation]] = {}
# session_id -> index of the first violation not yet shown to the user
_CURSOR: dict[str, int] = {}


def record(session_id: str, violation: Violation) -> Violation:
    violation.session_id = session_id
    _REGISTRY.setdefault(session_id, []).append(violation)
    db.audit(
        session_id,
        "guardrail.violation",
        violation_id=violation.violation_id,
        clause=violation.clause_id,
        severity=violation.severity,
        action=violation.action,
        title=violation.title,
    )
    return violation


def report(
    session_id: str,
    *,
    clause_id: str,
    category: str,
    severity: Severity,
    title: str,
    detail: str,
    source: Source,
    action: Action,
    routine: str = "-",
    evidence: str = "",
) -> Violation:
    """Convenience wrapper used by the gates and scanners."""
    return record(
        session_id,
        Violation(
            clause_id=clause_id,
            category=category,
            severity=severity,
            title=title,
            detail=detail,
            source=source,
            action=action,
            routine=routine,
            evidence=evidence,
        ),
    )


def all_for(session_id: str) -> list[Violation]:
    return list(_REGISTRY.get(session_id, []))


def drain(session_id: str) -> list[Violation]:
    """Violations recorded since the last drain. Used to render one banner
    per turn instead of repeating the whole history."""
    rows = _REGISTRY.get(session_id, [])
    start = _CURSOR.get(session_id, 0)
    _CURSOR[session_id] = len(rows)
    return rows[start:]


def summary(session_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in _REGISTRY.get(session_id, []):
        counts[v.severity] = counts.get(v.severity, 0) + 1
    return counts


def reset(session_id: str) -> None:
    _REGISTRY.pop(session_id, None)
    _CURSOR.pop(session_id, None)
