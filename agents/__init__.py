import inspect
from typing import Any, Callable

from autogen_core.tools import FunctionTool


def strict_tool(fn: Callable[..., Any]) -> FunctionTool:
    """Wrap an agent tool in OpenAI strict mode, so the model must always send every argument.

    Without it the model occasionally skips arguments (e.g. `confidence`), the call fails
    validation and that agent's output is lost.
    """
    return FunctionTool(fn, description=inspect.getdoc(fn) or fn.__name__, strict=True)
