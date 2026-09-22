"""LLM judge for the soft guardrails.

`output_scan.py` handles rules a regex can settle. This handles the ones it
cannot -- topical scope, persona breaks, tone drift -- by asking a model to
classify the turn against `topic_policy`.

Two properties follow, both deliberate:

  * The verdict is **non-deterministic**. The same borderline request can be
    called differently twice, so every verdict carries a confidence and only
    acts above a threshold.
  * It is **blocking on the way in, evidential below the threshold**. A
    confident out-of-scope verdict short-circuits the turn with a deflection
    before the agent loop runs. A lower-confidence one is recorded as a
    violation for review but the agent still handles the turn -- a false refusal
    on a real support question is a worse failure than answering a marginal one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from . import violations as gv
from .desk_policy import CHARTERS, routing_rubric, severity_for
from .topic_policy import TOPICS_BY_KEY, judge_rubric

JUDGE_ENABLED = os.environ.get("TLIFE_JUDGE", "1") not in {"0", "false", "off"}

# Below this, the verdict is logged but the turn proceeds normally.
BLOCK_THRESHOLD = float(os.environ.get("TLIFE_JUDGE_THRESHOLD", "0.75"))

_JUDGE_SYSTEM = f"""You classify whether a message to a mobile carrier's support
assistant is within that assistant's scope.

The assistant handles ONLY: bills and charges, payments and autopay, rate plans
and lines, device upgrades and orders, Home Internet, mobile service and device
troubleshooting, and the account record (contact details, address, authorized
users, PIN resets).

Out-of-scope categories:
{judge_rubric()}

Rules for your judgment:
- Judge the CUSTOMER'S REQUEST, not the assistant's reply.
- A request about a phone, an app on a phone, a gateway, or connectivity is
  device/service support and is IN SCOPE. Only genuine software development is
  the `coding` category.
- Anyone in danger, in distress, or without emergency calling ability is IN
  SCOPE and urgent. Never classify that as out of scope.
- "Are you a human?" is IN SCOPE and must be answered honestly.
- Wanting to cancel, port out, or complain is IN SCOPE.
- When genuinely unsure, prefer in_scope with a low confidence. A wrongly
  refused support question costs more than a marginally answered one.

If the request IS in scope, also say which desk owns it:
{routing_rubric()}

Answer the safety question INDEPENDENTLY of scope. A message can be
out-of-scope by topic and still be someone in danger; judge it on its own.
`safety` is the probability that the customer is in danger, in distress, or
unable to reach emergency services -- self-harm, abuse or domestic violence, a
medical emergency, or this line being their only way to call 911. Ordinary
frustration, annoyance or a routine complaint is NOT this.

Reply with JSON only:
{{"verdict": "in_scope" | "out_of_scope",
  "category": "<one key from the out-of-scope list, or null>",
  "owning_desk": "<one desk name, or null when out of scope or purely general>",
  "desk_confidence": <0.0-1.0>,
  "confidence": <0.0-1.0>,
  "safety": <0.0-1.0>,
  "reason": "<one short sentence>"}}"""


@dataclass
class ScopeVerdict:
    verdict: str
    category: str | None
    confidence: float
    reason: str
    raw: str = ""
    # Which desk owns this request, when it is in scope for the company.
    owning_desk: str | None = None
    desk_confidence: float = 0.0
    # The provider's own safety classifier refused the message. That fires on
    # jailbreak attempts AND on genuine crisis disclosures, and the error does
    # not say which -- so this can never be treated as a topic refusal.
    provider_filtered: bool = False
    # Asked independently of scope, so both backends answer the same questions.
    safety_p: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def out_of_scope(self) -> bool:
        return self.verdict == "out_of_scope"

    @property
    def should_block(self) -> bool:
        return self.out_of_scope and self.confidence >= BLOCK_THRESHOLD


_JUDGE_MODEL: Any = None


def _model() -> Any:
    """Runs on the same deployment as the desks unless TLIFE_JUDGE_MODEL says
    otherwise. This is one extra call per customer turn, so point it at the
    cheapest deployment you have."""
    global _JUDGE_MODEL
    if _JUDGE_MODEL is None:
        from ..routines.base import get_model

        _JUDGE_MODEL = get_model(os.environ.get("TLIFE_JUDGE_MODEL") or None)
    return _JUDGE_MODEL


_FILTER_MARKERS = ("content_filter", "content management policy", "jailbreak",
                   "responsible ai", "content filtering")


def _is_content_filter(exc: Exception) -> bool:
    """Did the provider itself refuse to process the message?

    Azure returns a 400 with `code: content_filter` when its jailbreak /
    policy classifier rejects the *prompt*. That is not an outage -- it is a
    second opinion, and a strong one.
    """
    text = str(exc).lower()
    return any(marker in text for marker in _FILTER_MARKERS)


def classify(message: str) -> ScopeVerdict | None:
    """Judge one customer message.

    Returns None only when the judge could not reach a verdict for an
    operational reason -- the turn then proceeds normally, because a judge
    outage must not take customer support down with it. A provider-side content
    filter is NOT an outage and does not fail open.
    """
    if not JUDGE_ENABLED or not message.strip():
        return None
    try:
        reply = _model().invoke(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": message[:4000]},
            ]
        )
        text = (getattr(reply, "text", None) or str(reply.content)).strip()
        body = text[text.find("{") : text.rfind("}") + 1] or "{}"
        data = json.loads(body)
        owning = data.get("owning_desk") or None
        um = getattr(reply, "usage_metadata", None) or {}
        return ScopeVerdict(
            verdict=str(data.get("verdict", "in_scope")),
            category=data.get("category") or None,
            confidence=float(data.get("confidence", 0.0)),
            reason=str(data.get("reason", "")),
            raw=text,
            owning_desk=owning if owning in CHARTERS else None,
            desk_confidence=float(data.get("desk_confidence", 0.0)),
            safety_p=float(data.get("safety", 0.0)),
            usage={"input_tokens": um.get("input_tokens", 0) or 0,
                   "output_tokens": um.get("output_tokens", 0) or 0},
        )
    except Exception as exc:  # noqa: BLE001
        if _is_content_filter(exc):
            # The provider's classifier rejected the message. It fires on
            # jailbreaks and on self-harm or abuse disclosures alike, and the
            # error does not distinguish them. Refusing it as "off-topic" would
            # turn a crisis disclosure into a brush-off, so this routes to a
            # human instead -- the only response that is correct either way.
            return ScopeVerdict(
                verdict="in_scope",
                category=None,
                confidence=0.0,
                reason=(
                    "The model provider's safety classifier refused to process "
                    "this message. That covers both adversarial prompts and "
                    "genuine crisis disclosures, so it is routed to a human "
                    "rather than refused as off-topic."
                ),
                raw="provider_content_filter",
                provider_filtered=True,
            )
        # Anything else is an outage. Fail open: support keeps working.
        return None


def record(session_id: str, verdict: ScopeVerdict, routine: str) -> gv.Violation:
    topic = TOPICS_BY_KEY.get(verdict.category or "")
    blocked = verdict.should_block
    return gv.report(
        session_id,
        clause_id=f"scope.off_topic.{verdict.category or 'unclassified'}",
        category="scope",
        severity=(topic.severity if topic else "medium"),  # type: ignore[arg-type]
        title=(f"Off-topic request: {topic.label}" if topic else "Off-topic request"),
        detail=(
            f"{verdict.reason} "
            + (
                "The turn was declined before the agent ran."
                if blocked
                else f"Confidence {verdict.confidence:.2f} is below the "
                     f"{BLOCK_THRESHOLD:.2f} block threshold, so the agent handled "
                     "it normally and this is recorded for review."
            )
        ),
        source="llm_judge",
        action="blocked" if blocked else "flagged",
        routine=routine,
        evidence=f"category={verdict.category} confidence={verdict.confidence:.2f}",
    )


BOUNDARY_THRESHOLD = float(os.environ.get("TLIFE_BOUNDARY_THRESHOLD", "0.75"))


def check_desk_boundary(
    session_id: str, verdict: ScopeVerdict, current_desk: str
) -> str | None:
    """Did an in-scope request land at the wrong desk?

    Triage is exempt -- routing is its job. For a desk, answering another
    desk's question means answering without that desk's tools, which is the
    most common way wrong information reaches a customer. Returns a routing
    note for the prompt, or None.
    """
    if current_desk == "triage" or verdict.out_of_scope:
        return None
    owner = verdict.owning_desk
    if not owner or owner == current_desk:
        return None
    if verdict.desk_confidence < BOUNDARY_THRESHOLD:
        return None

    charter = CHARTERS.get(current_desk)
    gv.report(
        session_id,
        clause_id="scope.desk_boundary",
        category="scope",
        severity="high",
        title=f"Request belongs to the {CHARTERS[owner].label} desk",
        detail=(
            f"{verdict.reason} This is the {charter.label if charter else current_desk} "
            f"desk, which does not own it and does not have the tools for it. "
            "The correct move is a handoff, not an answer."
        ),
        source="llm_judge",
        action="flagged",
        routine=current_desk,
        evidence=f"owner={owner} at={current_desk} conf={verdict.desk_confidence:.2f}",
    )
    return (
        f"ROUTING NOTE for this turn: this request belongs to the {owner} desk, "
        f"not yours. Do not answer it from general knowledge -- you do not have "
        f"that desk's tools. Transfer it, with one short sentence telling the "
        f"customer you are doing so."
    )


def _bare_finding(rule, confidence: float, item: dict, routine: str):
    """A finding shaped like a Violation but never written to the registry."""
    from .violations import Violation

    return Violation(
        clause_id=rule.clause_id or rule.id,
        category=rule.dimension,
        severity=rule.severity,
        title=rule.rule,
        detail=str(item.get("why", "")).strip(),
        source="llm_judge",
        action="flagged",
        routine=routine,
        evidence=f"{rule.id} conf={confidence:.2f} :: "
                 f"{str(item.get('evidence', ''))[:120]}",
    )


def deflection_for(verdict: ScopeVerdict) -> str:
    """What the customer sees when a turn is short-circuited."""
    topic = TOPICS_BY_KEY.get(verdict.category or "")
    base = "I can only help with your account and service here"
    key = topic.key if topic else ""
    if key == "coding":
        base = "I can't help with coding or software questions"
    elif key == "professional_advice":
        base = "I'm not able to give medical, legal or financial advice"
    elif key == "controversial":
        base = "I'm not the right place for that conversation"
    elif key == "creative":
        base = "I can't take on writing or roleplay requests"
    elif key == "academic":
        base = "I can't help with schoolwork"
    return (
        f"{base} -- bills, payments, plans, Home Internet, or a service problem. "
        "Is there something on your account I can look at?"
    )


# ===========================================================================
# Output review: the sampled soft-rule rubric
# ===========================================================================

REVIEW_THRESHOLD = float(os.environ.get("TLIFE_REVIEW_THRESHOLD", "0.70"))

# Token usage from the most recent review call, read by the backend wrapper.
LAST_REVIEW_USAGE: dict[str, int] = {}
REVIEW_ENABLED = os.environ.get("TLIFE_REVIEW", "1") not in {"0", "false", "off"}


def sample_rules(session_id: str, turn: int, desk: str = "") -> list:
    """Draw this turn's rubric: every `always` rule plus a weighted sample.

    Seeded off the session's guardrail seed, so a run with
    TLIFE_GUARDRAIL_SEED set reviews the same rules every time -- which is what
    you want when you are debugging a false positive.
    """
    import random

    from ..config import GUARDRAIL_CONFIG
    from .sampler import sampler_for, weighted_sample
    from .soft_rules import ALWAYS_RULES, SAMPLED_RULES

    from .desk_policy import rubric_weight

    rng = random.Random(sampler_for(session_id).session_seed ^ (0xA11CE + turn))
    lo, hi = GUARDRAIL_CONFIG.soft_review_min, GUARDRAIL_CONFIG.soft_review_max
    k = rng.randint(lo, hi)

    # Re-weight the pool for this desk: consent matters more at payments,
    # repeat-scripting more at tech support. A flat rubric under-samples
    # exactly the rule a given desk is most likely to break.
    pool = [
        _Weighted(rule, rubric_weight(desk, rule)) if desk else _Weighted(rule, rule.weight)
        for rule in SAMPLED_RULES
    ]
    drawn = weighted_sample(rng, pool, k)
    return list(ALWAYS_RULES) + [w.rule for w in drawn]


class _Weighted:
    """Adapter so `weighted_sample` can use a per-desk weight."""

    __slots__ = ("rule", "weight")

    def __init__(self, rule, weight: float):
        self.rule, self.weight = rule, weight


def _rubric_text(rules: list) -> str:
    blocks = []
    for r in rules:
        blocks.append(
            f"### {r.id}\n"
            f"RULE: {r.rule}\n"
            f"VIOLATES: {r.violates}\n"
            f"DOES NOT VIOLATE: {r.does_not_violate}"
        )
    return "\n\n".join(blocks)


_REVIEW_SYSTEM = """You are a quality reviewer for a mobile carrier's support
assistant. You are given the recent conversation, the assistant's latest reply,
and a rubric of rules to check that reply against.

For each rule, decide whether the LATEST ASSISTANT REPLY breaks it.

How to judge:
- Read the DOES NOT VIOLATE line as carefully as the VIOLATES line. It exists
  because these rules are easy to over-apply.
- Judge only the latest reply, in the context of the conversation. Do not flag
  the assistant for something the customer said.
- Only report a rule whose VIOLATES description clearly matches what the reply
  actually did. Do not report a neighbouring rule because it feels related --
  if the specific rule does not fit, report nothing.
- GROUNDING: the conversation includes TOOL RESULT lines. A figure is grounded
  if it appears in, or follows arithmetically from, any tool result shown. Only
  flag a grounding rule when you can point to a specific figure that contradicts
  or is absent from the tool results. If no tool results are shown at all, you
  cannot assess grounding -- do not flag it.
- Brevity, firmness and refusal are not faults. A short correct answer is a
  good answer. Declining an out-of-scope request is correct behaviour, not a
  dead end, as long as it offers what the assistant does handle.
- Confidence must reflect real certainty. Use below 0.7 when it is arguable.

You must return a verdict for EVERY rule in the rubric -- one entry each, in
order, no omissions and no additions.

- "pass" means the reply does not break that rule. This is the normal outcome;
  most replies pass most rules.
- "fail" means the reply clearly breaks it. If your own explanation would
  contain "though", "may not be", "arguably", or would end up describing the
  reply as acceptable, the verdict is "pass". Do not record a fail you are
  about to argue yourself out of.
- `why` must state what the reply did wrong. For a pass, leave it empty.

Reply with JSON only:
{"verdicts": [{"rule_id": "<exact id from the rubric>",
               "verdict": "pass" | "fail",
               "confidence": <0.0-1.0>,
               "evidence": "<short quote from the reply, only when failing>",
               "why": "<one sentence, only when failing>"}]}"""


_HEDGES = (
    "though ", "may not", "might not", "arguably", "could be argued",
    "not necessarily", "but this is", "which is acceptable", "is acceptable",
    "directly supported", "is supported", "no violation", "does not violate",
    "this may be fine", "probably fine", "borderline",
)


def _is_hedged(why: str) -> bool:
    """Drop fails whose own explanation concedes the reply was fine.

    Small judges will happily write 'this is directly supported by the tool
    results' inside a violation record. Cheap to catch, and catching it is the
    difference between a usable signal and noise.
    """
    low = why.lower()
    return any(h in low for h in _HEDGES)


def build_context(messages: list, limit: int = 14) -> str:
    """Render the recent turn history for the reviewer, tool results included.

    The tool results are the point: without them the reviewer cannot tell a
    grounded figure from an invented one, and defaults to flagging everything.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    lines: list[str] = []
    for msg in messages[-limit:]:
        if isinstance(msg, HumanMessage):
            lines.append(f"CUSTOMER: {getattr(msg, 'text', '') or msg.content}")
        elif isinstance(msg, ToolMessage):
            body = str(msg.content)
            lines.append(f"TOOL RESULT [{msg.name}]: {body[:700]}")
        elif isinstance(msg, AIMessage):
            text = getattr(msg, "text", "") or ""
            if text.strip():
                lines.append(f"ASSISTANT: {text}")
            for call in msg.tool_calls or []:
                lines.append(f"ASSISTANT CALLED: {call['name']}({call['args']})")
    return "\n".join(lines)


def review(
    session_id: str,
    routine: str,
    reply: str,
    context: str = "",
    turn: int = 0,
    record: bool = True,
) -> list[gv.Violation]:
    """Judge one assistant reply against a sampled slice of the soft rubric.

    `record=False` returns the findings without writing them to the violation
    registry -- used by the backend wrapper, which records through
    `recording.record_review` so both backends land in the registry by the same
    path. Recording in both places duplicated every finding.
    """
    if not (REVIEW_ENABLED and JUDGE_ENABLED) or not reply.strip():
        return []

    from .soft_rules import RULES_BY_ID

    rules = sample_rules(session_id, turn, desk=routine)
    user_block = (
        f"CONVERSATION SO FAR (most recent last, including tool results):\n"
        f"{context[-6000:] or '(no prior context)'}\n\n"
        f"LATEST ASSISTANT REPLY TO REVIEW:\n{reply[:3000]}\n\n"
        f"RUBRIC FOR THIS REVIEW:\n{_rubric_text(rules)}"
    )

    try:
        raw = _model().invoke(
            [{"role": "system", "content": _REVIEW_SYSTEM},
             {"role": "user", "content": user_block}]
        )
        text = (getattr(raw, "text", None) or str(raw.content)).strip()
        body = text[text.find("{") : text.rfind("}") + 1] or "{}"
        um = getattr(raw, "usage_metadata", None) or {}
        LAST_REVIEW_USAGE.clear()
        LAST_REVIEW_USAGE.update({"input_tokens": um.get("input_tokens", 0) or 0,
                                  "output_tokens": um.get("output_tokens", 0) or 0})
        parsed = json.loads(body)
        # Accept the older shape too, in case a model ignores the schema.
        found = parsed.get("verdicts") or parsed.get("violations") or []
    except Exception:  # noqa: BLE001 - a reviewer failure must not break the turn
        return []

    recorded: list[gv.Violation] = []
    sampled_ids = {r.id for r in rules}
    for item in found:
        rule = RULES_BY_ID.get(str(item.get("rule_id", "")))
        # Ignore anything outside the rubric we actually asked about.
        if rule is None or rule.id not in sampled_ids:
            continue
        # With the per-rule schema, most entries are passes.
        if str(item.get("verdict", "fail")).lower() != "fail":
            continue
        confidence = float(item.get("confidence", 0.0))
        if confidence < REVIEW_THRESHOLD:
            continue
        why = str(item.get("why", "")).strip()
        if _is_hedged(why):
            # The reviewer talked itself out of this one while writing it down.
            continue
        if not record:
            recorded.append(
                _bare_finding(rule, confidence, item, routine)
            )
            continue
        recorded.append(
            gv.report(
                session_id,
                clause_id=rule.clause_id or rule.id,
                category=rule.dimension,
                severity=severity_for(routine, rule),  # type: ignore[arg-type]
                title=f"{rule.dimension.title()}: {rule.rule}",
                detail=(
                    f"{item.get('why', '').strip()} "
                    f"(reviewed against {len(rules)} of the {len(RULES_BY_ID)} soft "
                    f"rules sampled for this turn)"
                ),
                source="llm_judge",
                action="flagged",
                routine=routine,
                evidence=f"{rule.id} conf={confidence:.2f} :: "
                         f"{str(item.get('evidence', ''))[:120]}",
            )
        )
    return recorded
