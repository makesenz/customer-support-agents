from unittest import mock

from autogen_core.tools import FunctionTool

from agents.knowledge_agent import search_kb
from agents.log_agent import log_ticket
from agents.query_agent import record_triage
from agents.resolution_agent import submit_reply
from core import tracing

EXPECTED_ARGS = {
    record_triage: ["category", "urgency", "sentiment", "summary", "search_query"],
    search_kb: ["query"],
    submit_reply: ["reply", "confidence", "used_sources"],
    log_ticket: ["escalate_suggested", "escalation_note"],
}


def test_tool_schemas_expose_only_real_arguments():
    for fn, args in EXPECTED_ARGS.items():
        assert list(FunctionTool(fn, description="x").schema["parameters"]["properties"]) == args


def test_traced_tool_hides_langsmith_config_param():
    """LangSmith's traceable adds `config=None`; AutoGen must not see it as a tool argument."""
    def fake_traceable(**_):
        def deco(fn):
            def wrapped(*a, config=None, **k):
                return fn(*a, **k)
            return wrapped
        return deco

    with mock.patch.object(tracing, "traceable", fake_traceable):
        @tracing.traced_tool("tool:demo")
        def demo(query: str) -> str:
            return query.upper()

    assert list(FunctionTool(demo, description="x").schema["parameters"]["properties"]) == ["query"]
    assert demo("hi") == "HI"
