"""LangSmith tracing helpers.

`traceable` is a no-op when LangSmith is not configured, so the app runs the
same with or without a LangSmith key.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import config

if config.LANGSMITH_ENABLED:
    from langsmith import traceable as _ls_traceable
    from langsmith.wrappers import wrap_openai
else:
    _ls_traceable = None
    wrap_openai = None


def traceable(*args: Any, **kwargs: Any) -> Callable:
    if _ls_traceable is not None:
        return _ls_traceable(*args, **kwargs)
    if args and callable(args[0]):
        return args[0]
    return lambda fn: fn


def traced_tool(name: str) -> Callable:
    """Trace an agent tool while keeping its original signature.

    LangSmith's wrapper adds a `config` parameter, which AutoGen would expose to the
    LLM as a tool argument, so AutoGen gets a plain wrapper with the real signature.
    """
    def decorator(fn: Callable) -> Callable:
        traced = traceable(name=name, run_type="tool")(fn)

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return await traced(*args, **kwargs)
        else:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return traced(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(fn)
        return wrapper

    return decorator


def instrument_model_client(model_client: Any) -> Any:
    """Wrap AutoGen's internal OpenAI client so every LLM call shows in LangSmith."""
    if wrap_openai is not None and hasattr(model_client, "_client"):
        model_client._client = wrap_openai(model_client._client)
    return model_client
