"""Resolution Agent: writes the customer reply using only the retrieved facts."""

from typing import Literal

from autogen_agentchat.agents import AssistantAgent

import config
from agents import strict_tool
from core.context import get_ticket
from core.tracing import traced_tool

Confidence = Literal["High", "Medium", "Low"]


@traced_tool("tool:submit_reply")
async def submit_reply(reply: str, confidence: Confidence, used_sources: list[str]) -> str:
    """Submit the customer-ready reply.

    confidence: High = fully answered from the passages; Medium = partly; Low = not answered.
    used_sources: file names of the passages actually used (e.g. ["refund_policy.md"]).
    """
    ctx = get_ticket()
    ctx.reply = reply.strip()
    ctx.confidence = confidence
    ctx.used_sources = used_sources
    return f"REPLY (confidence: {confidence}; sources: {', '.join(used_sources) or 'none'})\n\n{reply.strip()}"


def build(model_client) -> AssistantAgent:
    voice = (config.INTERNAL_KB_DIR / "brand_voice_guidelines.md").read_text(encoding="utf-8")
    return AssistantAgent(
        name="resolution_agent",
        description="Writes a clear, polite reply that solves the problem using only the facts the Knowledge Agent found.",
        model_client=model_client,
        tools=[strict_tool(submit_reply)],
        reflect_on_tool_use=False,
        system_message=f"""You are the Resolution Agent for {config.BRAND_NAME}.
Your ONLY job: write the reply to the customer and call `submit_reply` exactly once.

Grounding rules (most important):
- Use ONLY facts from the KB_RESULT message (passages and graph facts). Never invent prices, timelines,
  policies, phone numbers, UPI IDs or account numbers.
- If the TRIAGE category is Off-topic (not about Maa Wellness at all, e.g. sports, news, weather), do not try to
  answer it and do not promise a specialist callback. Politely say you can only help with Maa Wellness orders,
  products, payments, delivery and account questions, invite them to ask one, and set confidence to High.
- Otherwise, if KB_RESULT says NO_MATCH, or the passages do not answer the question, say honestly that you don't have
  that information, tell the customer a support specialist will contact them, give {config.SUPPORT_PHONE},
  and set confidence to Low.
- If the complaint involves a side effect, a medical condition, a legal threat, fraud or a payment dispute,
  write a calm holding reply (acknowledge, say a specialist will personally follow up, share {config.SUPPORT_PHONE}),
  and do not make promises.
- Reply in the customer's language (the "Detected language" in the task). Use the customer's name if given.

Brand voice:
{voice}""",
    )
