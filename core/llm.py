"""Model client shared by all agents (OpenAI or any OpenAI-compatible endpoint)."""
from __future__ import annotations

from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

import config
from core.tracing import instrument_model_client


class ToolRequiredClient(OpenAIChatCompletionClient):
    """Every agent in this team must finish by calling its one tool, so never let the model skip it.

    AssistantAgent always asks for tool_choice="auto"; with reflect_on_tool_use=False there is
    no follow-up LLM call after the tool runs, so "required" cannot cause a loop.
    """

    async def create(self, messages, *, tools=(), tool_choice="auto", **kwargs):
        if tools and tool_choice == "auto":
            tool_choice = "required"
        return await super().create(messages, tools=tools, tool_choice=tool_choice, **kwargs)


def make_model_client() -> OpenAIChatCompletionClient:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    kwargs = dict(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        parallel_tool_calls=False,
    )
    if config.LLM_BASE_URL:
        # Non-OpenAI providers need model_info so AutoGen knows the model supports tools.
        kwargs["base_url"] = config.LLM_BASE_URL
        kwargs["model_info"] = {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        }
    return instrument_model_client(ToolRequiredClient(**kwargs))
