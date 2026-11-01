"""Log Agent: records every ticket and decides escalation (rules enforced in code)."""

from autogen_agentchat.agents import AssistantAgent

import config
from agents import strict_tool
from core import ticket_log
from core.context import get_ticket
from core.tracing import traced_tool
from guardrails.escalation import decide
from guardrails.output_guard import check_output


@traced_tool("tool:log_ticket")
async def log_ticket(escalate_suggested: bool, escalation_note: str) -> str:
    """Log the ticket to the ticket file and apply the escalation rules.

    escalate_suggested: True only if a rule from the escalation matrix applies.
    escalation_note: the rule code and a short reason, e.g. "E9 customer opened the tin" (empty if not escalating).
    """
    ctx = get_ticket()
    triage = ctx.triage
    out = check_output(ctx.reply)

    log_df = ticket_log.read_log()
    previous = 0
    if ctx.order_id:
        previous = int((log_df["order_id"].str.upper() == ctx.order_id.upper()).sum())

    status, reasons = decide(
        complaint=ctx.complaint,
        urgency=triage.get("urgency", "Medium"),
        sentiment=triage.get("sentiment", "Neutral"),
        kb_found=ctx.kb_found,
        confidence=ctx.confidence,
        input_flags=ctx.input_flags,
        output_flags=out.flags,
        output_blocked=out.blocked,
        previous_tickets_for_order=previous,
        llm_suggests=escalate_suggested,
        llm_reason=escalation_note,
        category=triage.get("category", ""),
    )
    sources = ctx.used_sources or [r["source"] for r in ctx.retrieval]
    if ctx.graph_facts and "knowledge_graph" not in sources:
        sources = list(sources) + ["knowledge_graph"]
    row = ticket_log.append_row({
        "ticket_id": ctx.ticket_id,
        "customer_name": ctx.customer_name,
        "order_id": ctx.order_id,
        "complaint": ctx.masked_complaint,  # never store raw card numbers / OTPs
        "category": triage.get("category", "Other"),
        "urgency": triage.get("urgency", ""),
        "sentiment": triage.get("sentiment", ""),
        "summary": triage.get("summary", ""),
        "language": ctx.language,
        "kb_sources": "; ".join(dict.fromkeys(sources)) or "none",
        "response": out.reply,
        "confidence": ctx.confidence,
        "status": status,
        "escalation_reason": " + ".join(reasons),
        "guardrail_flags": "; ".join(ctx.input_flags + out.flags),
    })
    ctx.result = row
    return f"LOGGED {ctx.ticket_id} | status: {status}" + (f" | reason: {' + '.join(reasons)}" if reasons else "")


def build(model_client) -> AssistantAgent:
    matrix = (config.INTERNAL_KB_DIR / "escalation_matrix.md").read_text(encoding="utf-8")
    return AssistantAgent(
        name="log_agent",
        description="Records every ticket to the log file and decides whether to escalate it to human support.",
        model_client=model_client,
        tools=[strict_tool(log_ticket)],
        reflect_on_tool_use=False,
        system_message=f"""You are the Log Agent for {config.BRAND_NAME}.
Your ONLY job: review the TRIAGE, KB_RESULT and REPLY messages, decide whether a human should take over,
and call `log_ticket` exactly once. The tool already enforces urgency, sentiment, confidence, missing documents
and the guardrail flags in code, so you only add what code cannot see.

- Set escalate_suggested=true ONLY when a specific rule (E1–E13) below applies, and start escalation_note with
  that rule code, e.g. "E9 customer says the tin is opened". A suggestion without a rule code is ignored.
- Do NOT escalate normal cases that the REPLY answered from the documents, for example: a late order still within
  the delivery window, returning SEALED/unopened products within 10 days, password reset, a damaged item reported
  with a replacement offered, pricing, COD or packaging questions.
- Do NOT escalate Off-topic messages (sports, news, general knowledge) just because no document matched; the
  polite redirect is the correct answer. Only escalate them if another rule (legal threat, abuse...) applies.

{matrix}""",
    )
