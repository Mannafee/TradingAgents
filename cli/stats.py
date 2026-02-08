"""Simple stats tracker - replaces LangChain StatsCallbackHandler."""

import threading
from typing import Any, Dict


class StatsTracker:
    """Thread-safe counter for LLM calls, tool calls, and token usage.

    CodexClient calls tracker.record_llm_call() and tracker.record_tool_call()
    directly - no LangChain callback infrastructure needed.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def record_llm_call(self, input_tokens: int = 0, output_tokens: int = 0):
        with self._lock:
            self.llm_calls += 1
            self.tokens_in += input_tokens
            self.tokens_out += output_tokens

    def record_tool_call(self):
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }

    @staticmethod
    def estimate_cost(tokens_in: int, tokens_out: int, model: str) -> float:
        """Estimate API cost in USD based on token usage and model.

        Prices are per 1M tokens (input, output).
        Returns estimated cost in USD.
        """
        PRICING = {
            "gpt-5.2": (2.50, 10.00),
            "gpt-5.2-codex": (2.50, 10.00),
            "gpt-5.3-codex": (2.50, 10.00),
            "gpt-5.1": (2.50, 10.00),
            "gpt-5.1-codex-mini": (0.40, 1.60),
            "gpt-5.1-codex-max": (2.50, 10.00),
            "gpt-5": (2.00, 8.00),
            "gpt-5-mini": (0.40, 1.60),
            "gpt-5-nano": (0.10, 0.40),
            "gpt-4.1": (2.00, 8.00),
        }

        prices = PRICING.get(model)
        if not prices:
            for key, val in PRICING.items():
                if model.startswith(key):
                    prices = val
                    break

        if not prices:
            prices = (2.00, 8.00)

        input_cost = (tokens_in / 1_000_000) * prices[0]
        output_cost = (tokens_out / 1_000_000) * prices[1]
        return input_cost + output_cost
