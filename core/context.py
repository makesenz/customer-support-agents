"""Per-ticket state shared by the tools of one team run.

Agents pass short text to each other, but the tools need the full facts
(original complaint, guardrail flags, retrieved sources). We keep them in a
context variable so log_ticket can record them without the LLM copying them.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TicketContext:
    ticket_id: str
    customer_name: str
    order_id: str
    complaint: str  # original text shown to staff
    masked_complaint: str  # PII-masked text sent to the LLM
    language: str = "en"
    input_flags: list[str] = field(default_factory=list)
    # Query Agent
    triage: dict[str, str] = field(default_factory=dict)
    # Knowledge Agent
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    graph_facts: list[str] = field(default_factory=list)
    retrieval_timings: dict[str, float] = field(default_factory=dict)
    kb_found: bool = False
    # Resolution Agent
    reply: str = ""
    confidence: str = "Low"
    used_sources: list[str] = field(default_factory=list)
    # Log Agent
    result: dict[str, Any] | None = None


_current: contextvars.ContextVar[TicketContext | None] = contextvars.ContextVar("ticket_ctx", default=None)


def set_ticket(ctx: TicketContext) -> None:
    _current.set(ctx)


def get_ticket() -> TicketContext:
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError("No active ticket. Call set_ticket() before running the team.")
    return ctx
