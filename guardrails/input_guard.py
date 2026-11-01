"""Input guardrails: run on the raw complaint *before* any LLM sees it.

1. Validation      – empty / too long
2. PII masking     – card numbers, CVV, Aadhaar, OTPs and passwords never reach the LLM
3. Prompt injection – "ignore previous instructions", "reveal your system prompt" ...
4. Risk flags      – legal, fraud/payment dispute, medical, side effect, abuse, self-harm
5. Language        – detected so the reply can match the customer's language
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langdetect import DetectorFactory, LangDetectException, detect

from core.tracing import traceable

DetectorFactory.seed = 0
MAX_CHARS = 2000

_PII_PATTERNS = [
    ("CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b")),
    ("CVV", re.compile(r"\b(cvv|cvc)\s*(is|:)?\s*\d{3,4}\b", re.I)),
    ("OTP", re.compile(r"\b(otp|one time password)\s*(is|:)?\s*\d{4,8}\b", re.I)),
    ("PASSWORD", re.compile(r"\b(password|passcode|pin)\s*(is|:)\s*\S+", re.I)),
]

_INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules|prompts?)"
    r"|disregard (your|the|all) (instructions|rules)"
    r"|(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)"
    r"|you are now|act as (an? )?(admin|developer|dan)|developer mode|jailbreak"
    r"|pretend (to be|you are)|new instructions?:|system prompt",
    re.I,
)

RISK_KEYWORDS: dict[str, list[str]] = {
    "legal_threat": ["lawyer", "advocate", "legal notice", "legal action", "court", "consumer forum", "sue ",
                     "police", "fir ", "cyber crime", "consumer complaint"],
    "payment_dispute": ["charged twice", "charged two times", "three times", "charged 3", "double charge",
                        "deducted twice", "chargeback", "refund my money", "money deducted", "paid but",
                        "extra money", "charged my card", "charged my upi"],
    "fraud": ["fraud", "scam", "cheat", "cheated", "fake website", "fake product", "duplicate product",
              "counterfeit", "another website", "other website"],
    "medical_question": ["pregnan", "breastfeed", "diabet", "sugar patient", "blood pressure", " bp ",
                         "heart", "kidney", "liver", "thyroid", "surgery", "tablet", "medicine", "medication",
                         "doctor", "insulin"],
    "side_effect": ["side effect", "rash", "itching", "allerg", "vomit", "stomach pain", "loose motion",
                    "diarrh", "dizz", "headache", "chest pain", "swelling", "burning", "bleeding", "fainted"],
    "abusive": ["idiot", "stupid", "useless", "bloody", "nonsense", "worst company", "shame on you",
                "f***", "fuck", "bastard"],
    "self_harm": ["suicide", "kill myself", "end my life"],
    "media_or_bulk": ["distributor", "dealership", "wholesale", "bulk order", "journalist", "media",
                      "employees", "corporate"],
}


@dataclass
class InputCheck:
    original: str
    masked: str
    flags: list[str] = field(default_factory=list)
    pii_found: list[str] = field(default_factory=list)
    language: str = "en"
    blocked: bool = False
    block_reason: str = ""


def detect_language(text: str) -> str:
    if re.search(r"[஀-௿]", text):  # Tamil script
        return "ta"
    try:
        return detect(text) if len(text) > 15 else "en"
    except LangDetectException:
        return "en"


def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, pattern in _PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[{label} REDACTED]", text)
    return text, found


def risk_flags(text: str) -> list[str]:
    q = f" {text.lower()} "
    return [flag for flag, words in RISK_KEYWORDS.items() if any(w in q for w in words)]


@traceable(name="input_guardrail")
def check_input(text: str) -> InputCheck:
    text = (text or "").strip()
    if not text:
        return InputCheck(original=text, masked=text, blocked=True, block_reason="Empty complaint")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    masked, pii = mask_pii(text)
    flags = risk_flags(text)
    if pii:
        flags.append("pii_redacted")
    if _INJECTION.search(text):
        flags.append("prompt_injection")
    if text.isupper() and len(text) > 20:
        flags.append("abusive")  # all caps = shouting
    return InputCheck(original=text, masked=masked, flags=list(dict.fromkeys(flags)),
                      pii_found=pii, language=detect_language(text))
