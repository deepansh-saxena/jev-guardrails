"""Turn backend decisions into violation records.

Both backends answer the same questions in the same shapes, so the reporting,
thresholds and deflection text live here once rather than twice. The only thing
that differs is who answered and how trustworthy the number is.
"""

from __future__ import annotations

import os

from . import violations as gv
from .backends import ReviewDecision, ScopeDecision
from .desk_policy import CHARTERS, severity_for
from .soft_rules import RULES_BY_ID
from .topic_policy import TOPICS_BY_KEY

SCOPE_BLOCK_THRESHOLD = float(os.environ.get("TLIFE_JUDGE_THRESHOLD", "0.75"))
BOUNDARY_THRESHOLD = float(os.environ.get("TLIFE_BOUNDARY_THRESHOLD", "0.75"))


def record_scope(
    session_id: str, decision: ScopeDecision, desk: str, backend: str
) -> bool:
    """Record an out-of-scope decision. Returns True when the turn should block."""
    if decision.provider_filtered:
        gv.report(
            session_id,
            clause_id="escalation.vulnerability",
            category="escalation",
            severity="critical",
            title="Provider safety filter fired -- route to a human",
            detail=(
                f"{decision.reason} Treating this as a topic refusal risks "
                "brushing off someone in crisis, so the turn escalates instead."
            ),
            source="runtime_gate",
            action="flagged",
            routine=desk,
            evidence=f"backend={backend} provider_filtered=true",
        )
        return False

    if decision.in_scope:
        return False
    # Safety always outranks scope, whatever the classifier said.
    if decision.safety_urgent:
        return False

    topic = TOPICS_BY_KEY.get(decision.category or "")
    blocking = decision.confidence >= SCOPE_BLOCK_THRESHOLD
    gv.report(
        session_id,
        clause_id=f"scope.off_topic.{decision.category or 'unclassified'}",
        category="scope",
        severity=(topic.severity if topic else "medium"),  # type: ignore[arg-type]
        title=(f"Off-topic request: {topic.label}" if topic else "Off-topic request"),
        detail=(
            f"{decision.reason} "
            + (
                "The turn was declined before the agent's model/tool loop ran."
                if blocking
                else f"Probability {decision.confidence:.2f} is below the "
                     f"{SCOPE_BLOCK_THRESHOLD:.2f} block threshold, so the agent "
                     "handled it and this is recorded for review."
            )
        ),
        source="llm_judge",
        action="blocked" if blocking else "flagged",
        routine=desk,
        evidence=f"backend={backend} category={decision.category} "
                 f"p={decision.confidence:.2f}",
    )
    return blocking


def record_boundary(
    session_id: str, decision: ScopeDecision, desk: str, backend: str
) -> str | None:
    """Record a wrong-desk request and return a routing note for the prompt."""
    if desk == "triage" or not decision.in_scope:
        return None
    owner = decision.owning_desk
    if not owner or owner == desk or owner not in CHARTERS:
        return None
    if decision.desk_confidence < BOUNDARY_THRESHOLD:
        return None

    here = CHARTERS.get(desk)
    gv.report(
        session_id,
        clause_id="scope.desk_boundary",
        category="scope",
        severity="high",
        title=f"Request belongs to the {CHARTERS[owner].label} desk",
        detail=(
            f"This is the {here.label if here else desk} desk, which does not own "
            "it and does not have the tools for it. The correct move is a "
            "handoff, not an answer."
        ),
        source="llm_judge",
        action="flagged",
        routine=desk,
        evidence=f"backend={backend} owner={owner} at={desk} "
                 f"p={decision.desk_confidence:.2f}",
    )
    return (
        f"ROUTING NOTE for this turn: this request belongs to the {owner} desk, "
        "not yours. Do not answer it from general knowledge -- you do not have "
        "that desk's tools. Transfer it, with one short sentence telling the "
        "customer you are doing so."
    )


def record_safety(session_id: str, desk: str, backend: str) -> None:
    gv.report(
        session_id,
        clause_id="topic.safety_overrides_scope",
        category="escalation",
        severity="critical",
        title="Safety situation detected -- escalate before anything else",
        detail=(
            "The customer may be in danger or unable to reach emergency "
            "services. Scope and workflow rules do not apply; escalate."
        ),
        source="llm_judge",
        action="flagged",
        routine=desk,
        evidence=f"backend={backend} safety=true",
    )


def record_review(
    session_id: str, decision: ReviewDecision, desk: str, backend: str
) -> int:
    """Record soft-rule findings. Returns how many were recorded."""
    n = 0
    for finding in decision.findings:
        rule = RULES_BY_ID.get(finding.rule_id)
        if rule is None:
            continue
        gv.report(
            session_id,
            clause_id=rule.clause_id or rule.id,
            category=rule.dimension,
            severity=severity_for(desk, rule),  # type: ignore[arg-type]
            title=f"{rule.dimension.title()}: {rule.rule}",
            detail=(
                f"P(violated)={finding.probability:.2f}. Reviewed against "
                f"{decision.rules_evaluated} of the {len(RULES_BY_ID)} soft rules."
            ),
            source="llm_judge",
            action="flagged",
            routine=desk,
            evidence=f"backend={backend} {finding.evidence or finding.rule_id}",
        )
        n += 1
    return n
