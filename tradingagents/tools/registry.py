"""Tool registry - maps analyst types to tool schemas and executors."""

import json
from typing import Callable, Dict, List, Tuple

from . import market as market_tools
from . import fundamentals as fundamentals_tools
from . import news as news_tools

# Map analyst type -> (schemas, executors)
_ANALYST_TOOLS = {
    "market": (market_tools.SCHEMAS, market_tools.EXECUTORS),
    "social": (
        [news_tools.GET_NEWS_SCHEMA],
        {"get_news": news_tools.EXECUTORS["get_news"]},
    ),
    "news": (
        [news_tools.GET_NEWS_SCHEMA, news_tools.GET_GLOBAL_NEWS_SCHEMA, news_tools.GET_INSIDER_TRANSACTIONS_SCHEMA],
        {
            "get_news": news_tools.EXECUTORS["get_news"],
            "get_global_news": news_tools.EXECUTORS["get_global_news"],
            "get_insider_transactions": news_tools.EXECUTORS["get_insider_transactions"],
        },
    ),
    "fundamentals": (fundamentals_tools.SCHEMAS, fundamentals_tools.EXECUTORS),
}


def get_tools(analyst_type: str) -> Tuple[List[dict], Callable[[str, str], str]]:
    """Get tool schemas and an executor function for a given analyst type.

    Args:
        analyst_type: One of "market", "social", "news", "fundamentals"

    Returns:
        (schemas, executor) where:
        - schemas: List of tool JSON schemas for the Responses API
        - executor: Callable(tool_name, arguments_json) -> result_string
    """
    if analyst_type not in _ANALYST_TOOLS:
        raise ValueError(f"Unknown analyst type: {analyst_type}")

    schemas, executors = _ANALYST_TOOLS[analyst_type]

    def executor(tool_name: str, arguments_json: str) -> str:
        if tool_name not in executors:
            return f"Error: Unknown tool '{tool_name}'"
        try:
            args = json.loads(arguments_json)
            result = executors[tool_name](**args)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    return schemas, executor
