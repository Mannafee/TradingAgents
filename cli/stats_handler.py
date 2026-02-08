import threading
from typing import Any, Dict, List, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        if usage_metadata:
            with self._lock:
                self.tokens_in += usage_metadata.get("input_tokens", 0)
                self.tokens_out += usage_metadata.get("output_tokens", 0)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }

    @staticmethod
    def estimate_cost(tokens_in: int, tokens_out: int, model: str, provider: str = "") -> float:
        """Estimate API cost in USD based on token usage and model.

        Prices are per 1M tokens (input, output). Updated Feb 2026.
        Returns estimated cost in USD.
        """
        # Pricing per 1M tokens: (input_price, output_price)
        PRICING = {
            # OpenAI
            "gpt-5.2": (2.50, 10.00),
            "gpt-5.1": (2.50, 10.00),
            "gpt-5": (2.00, 8.00),
            "gpt-5-mini": (0.40, 1.60),
            "gpt-5-nano": (0.10, 0.40),
            "gpt-4.1": (2.00, 8.00),
            "gpt-4.1-mini": (0.40, 1.60),
            "gpt-4.1-nano": (0.10, 0.40),
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "o3": (2.00, 8.00),
            "o3-mini": (1.10, 4.40),
            "o4-mini": (1.10, 4.40),
            # Anthropic
            "claude-sonnet-4-5": (3.00, 15.00),
            "claude-sonnet-4.5": (3.00, 15.00),
            "claude-opus-4-5": (15.00, 75.00),
            "claude-opus-4.5": (15.00, 75.00),
            "claude-opus-4-1": (15.00, 75.00),
            "claude-sonnet-4": (3.00, 15.00),
            "claude-haiku-4-5": (0.80, 4.00),
            "claude-haiku-4.5": (0.80, 4.00),
            # Google
            "gemini-2.5-flash": (0.15, 0.60),
            "gemini-2.5-pro": (1.25, 10.00),
            "gemini-3-flash-preview": (0.15, 0.60),
            "gemini-3-pro-preview": (1.25, 10.00),
            # xAI
            "grok-3": (3.00, 15.00),
            "grok-3-mini": (0.30, 0.50),
        }

        # Try exact match first, then prefix match
        prices = PRICING.get(model)
        if not prices:
            for key, val in PRICING.items():
                if model.startswith(key):
                    prices = val
                    break

        if not prices:
            # Default estimate: mid-range pricing
            prices = (2.00, 8.00)

        input_cost = (tokens_in / 1_000_000) * prices[0]
        output_cost = (tokens_out / 1_000_000) * prices[1]
        return input_cost + output_cost
