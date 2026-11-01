"""Deterministic escalation rules (knowledge_base/internal/escalation_matrix.md).

The Log Agent's LLM *suggests* escalation; this function makes the final,
auditable decision so a ticket can never slip through because a model forgot a rule.
"""
from __future__ import annotations

import re

from core.tracing import traceable

HIGH_VALUE = 9900
_AMOUNT = re.compile(r"(?:₹|rs\.?|inr)?\s?(\d{1,3}(?:,\d{3})+|\d{4,6})", re.I)
_HIGH_VALUE_PACKS = re.compile(r"power combo|ultra|max performance|4\s?\+\s?1|8\s?\+\s?2|12\s?\+\s?3", re.I)

FLAG_RULES = {
    "legal_threat": "E4 Legal threat",
    "payment_dispute": "E5 Payment dispute",
    "fraud": "E5/E11 Fraud or unofficial seller",
    "side_effect": "E7 Side effect reported",
    "medical_question": "E8 Personal medical question",
    "self_harm": "Safety: self-harm language",
    "prompt_injection": "Guardrail: prompt injection attempt",
    "media_or_bulk": "E12 Bulk / distributor / media enquiry",
}


def is_high_value(complaint: str) -> bool:
    if _HIGH_VALUE_PACKS.search(complaint):
        return True
    for raw in _AMOUNT.findall(complaint):
        if int(raw.replace(",", "")) >= HIGH_VALUE:
            return True
    return False


@traceable(name="escalation_rules")
def decide(*, complaint: str, urgency: str, sentiment: str, kb_found: bool, confidence: str,
           input_flags: list[str], output_flags: list[str], output_blocked: bool,
           previous_tickets_for_order: int, llm_suggests: bool, llm_reason: str = "") -> tuple[str, list[str]]:
    reasons: list[str] = []
    if urgency.lower() == "high":
        reasons.append("E1 High urgency")
    if sentiment.lower() in ("very angry", "angry") and ("abusive" in input_flags or sentiment.lower() == "very angry"):
        reasons.append("E2 Very angry customer")
    if not kb_found:
        reasons.append("E3 No matching document")
    for flag, reason in FLAG_RULES.items():
        if flag in input_flags:
            reasons.append(reason)
    if confidence.lower() == "low":
        reasons.append("E6 Low confidence")
    if "refund" in complaint.lower() and re.search(r"\b(?<!un)(opened|used)\b|didn'?t work|not work|no result", complaint, re.I):
        reasons.append("E9 Refund for opened product / no results")
    if reasons and is_high_value(complaint):
        reasons.append("E10 High-value order")
    if previous_tickets_for_order >= 1:
        reasons.append("E13 Repeat complaint for this order")
    if output_blocked:
        reasons.append("Guardrail: reply blocked")
    elif any(f.startswith("unverified_price") for f in output_flags):
        reasons.append("Guardrail: price not in rate card")
    # The Log Agent may add a rule the keyword checks missed, but only by citing its rule code.
    # E10 (high value) only adds priority to another rule, so it can't escalate alone.
    code = re.match(r"\s*(E\d{1,2})\b", llm_reason or "")
    if llm_suggests and code and (code.group(1) != "E10" or reasons):
        reasons.append(f"Log Agent: {llm_reason.strip()}")
    reasons = list(dict.fromkeys(reasons))
    return ("Escalated" if reasons else "Resolved"), reasons
