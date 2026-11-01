"""Knowledge Base Agent: hybrid search over company documents + knowledge graph."""

import asyncio

from autogen_agentchat.agents import AssistantAgent

import config
from core.context import get_ticket
from core.tracing import traced_tool
from rag.hybrid import get_retriever


@traced_tool("tool:search_kb")
async def search_kb(query: str) -> str:
    """Search the Maa Wellness knowledge base (FAQs, policies, rate card) and knowledge graph.

    Returns matching passages with their source file names, plus structured graph facts.
    """
    ctx = get_ticket()
    result = await asyncio.to_thread(get_retriever().retrieve, query)
    ctx.retrieval = [
        {"source": p.source, "section": p.section, "score": p.rerank_score,
         "bm25_rank": p.bm25_rank, "vector_rank": p.vector_rank, "text": p.text}
        for p in result.passages
    ]
    ctx.graph_facts = result.graph_facts
    ctx.kb_found = result.found
    ctx.retrieval_timings = result.timings_ms

    if not result.found:
        return "KB_RESULT\nNO_MATCH: No relevant company document was found for this query."
    lines = ["KB_RESULT"]
    for i, p in enumerate(result.passages, 1):
        lines.append(f"[{i}] source: {p.source} | section: {p.section}\n{p.text}")
    if result.graph_facts:
        lines.append("[graph] source: knowledge_graph (Neo4j)")
        lines.extend(f"- {fact}" for fact in result.graph_facts)
    return "\n\n".join(lines)


def build(model_client) -> AssistantAgent:
    return AssistantAgent(
        name="knowledge_agent",
        description="Searches company documents and the knowledge graph and returns only the relevant passages with source file names.",
        model_client=model_client,
        tools=[search_kb],
        reflect_on_tool_use=False,
        system_message=f"""You are the Knowledge Base Agent for {config.BRAND_NAME}.
Your ONLY job: call `search_kb` exactly once using the "Search query" from the TRIAGE message
(you may improve the keywords slightly). Do not answer the customer and do not add facts yourself.""",
    )
