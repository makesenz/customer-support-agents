"""Output guardrails: run on the Resolution Agent's reply *before* it is logged or shown.

1. Medical-claim filter  – removes sentences with "cure", "guaranteed", "Viagra" ...
2. Brand filter          – only "Maa Wellness" / www.maawellness.in may be named
3. Payment-detail check  – any UPI ID, account number or IFSC that isn't official → reply blocked
4. Price check           – every ₹ amount must exist in the official rate card
5. Contact check         – phone numbers must be the official support number
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import config
from core.tracing import traceable
from rag.graph import load_seed

_BANNED_CLAIMS = re.compile(
    r"\bcure[sd]?\b|\bguarantee[sd]?\b|\b100\s?% (result|cure|effective|success|safe)"
    r"|permanent (solution|cure|result)|\bviagra\b|no side[- ]effects?|side[- ]effect[- ]free"
    r"|clinically (proven|tested)|\bmiracle\b",
    re.I,
)
_OTHER_BRANDS = re.compile(r"maa\s?naturals?(?:\.com|\.co\.in)?|maanaturals\S*", re.I)
_UPI = re.compile(r"\b[\w.\-]{3,}@[a-z]{2,}\b", re.I)
_ACCOUNT = re.compile(r"\b\d{11,18}\b")
_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
_PHONE = re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b")
_RUPEES = re.compile(r"(?:₹|rs\.?|inr)\s?([\d,]+)", re.I)
# Split on sentence ends, but not after "Mr." / "Ms." / "Dr." / "Rs." or a list number like "2.".
_SENTENCES = re.compile(r"((?<!Mr\.)(?<!Ms\.)(?<!Dr\.)(?<!Rs\.)(?<!\n\d\.)(?<!^\d\.)(?<=[.!?])[ \t]+|\n+)")

FALLBACK_REPLY = (
    "Thank you for reaching out to Maa Wellness. A member of our support team will personally review your "
    f"request and contact you shortly. For urgent help, please call or WhatsApp us at {config.SUPPORT_PHONE}.\n\n"
    "Warm regards,\nTeam Maa Wellness"
)


@dataclass
class OutputCheck:
    reply: str
    flags: list[str] = field(default_factory=list)
    blocked: bool = False


@lru_cache(maxsize=1)
def allowed_amounts() -> set[int]:
    seed = load_seed()
    amounts = {300}
    for pk in seed["packages"]:
        amounts |= {pk["price"], pk["mrp"], pk["mrp"] - pk["price"], pk["price"] + 300}
    return amounts


def _official_digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _drop_sentences(text: str, pattern: re.Pattern, keep_if: re.Pattern | None = None) -> tuple[str, bool]:
    """Remove sentences matching `pattern`, keeping the original spacing and line breaks."""
    parts = _SENTENCES.split(text)  # [sentence, separator, sentence, separator, ...]
    out, removed = [], False
    for i in range(0, len(parts), 2):
        sentence, sep = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
        if pattern.search(sentence) and not (keep_if and keep_if.search(sentence)):
            removed = True
            sep = sep if "\n" in sep else ""
            sentence = ""
        out.append(sentence + sep)
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip(), removed


@traceable(name="output_guardrail")
def check_output(reply: str) -> OutputCheck:
    reply = (reply or "").strip()
    flags: list[str] = []
    if len(reply) < 20:
        return OutputCheck(reply=FALLBACK_REPLY, flags=["empty_reply"], blocked=True)

    # 3. Payment details: wrong account info is dangerous → block the whole reply.
    for upi in _UPI.findall(reply):
        if upi.lower() != config.OFFICIAL_UPI and "@gmail" not in upi.lower():
            flags.append("unofficial_upi")
    for acc in _ACCOUNT.findall(reply):
        if acc != config.OFFICIAL_ACCOUNT_NUMBER:
            flags.append("unofficial_account_number")
    for ifsc in _IFSC.findall(reply):
        if ifsc != config.OFFICIAL_IFSC:
            flags.append("unofficial_ifsc")
    official_phone = _official_digits(config.SUPPORT_PHONE)[-10:]
    for phone in _PHONE.findall(reply):
        if _official_digits(phone)[-10:] != official_phone:
            flags.append("unofficial_phone")
    if flags:
        return OutputCheck(reply=FALLBACK_REPLY, flags=flags, blocked=True)

    # 1. Medical claims: strip offending sentences.
    reply, removed = _drop_sentences(reply, _BANNED_CLAIMS)
    if removed:
        flags.append("medical_claim_removed")

    # 2. Other brand names (allowed only next to the bank account name).
    keep = re.compile(rf"account name|hdfc|{re.escape(config.SUPPORT_EMAIL)}", re.I)
    reply, removed = _drop_sentences(reply, _OTHER_BRANDS, keep_if=keep)
    if removed:
        flags.append("other_brand_removed")

    # 4. Prices must come from the rate card.
    for amount in _RUPEES.findall(reply):
        value = int(amount.replace(",", "") or 0)
        if value and value not in allowed_amounts():
            flags.append(f"unverified_price_{value}")

    if len(reply.strip()) < 20:
        return OutputCheck(reply=FALLBACK_REPLY, flags=flags + ["empty_after_filter"], blocked=True)
    return OutputCheck(reply=reply, flags=flags)
