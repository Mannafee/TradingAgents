"""Base agent class for all pipeline agents."""

from tradingagents.client.codex import CodexClient
from tradingagents.client.types import Message
from tradingagents.pipeline.state import PipelineState


class BaseAgent:
    """Base class providing shared LLM invocation."""

    def __init__(self, client: CodexClient, model: str):
        self.client = client
        self.model = model

    async def _invoke(self, prompt: str, system: str = "") -> str:
        """Simple LLM call with system + user message, returns text."""
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))

        response = await self.client.complete(
            messages, model=self.model,
        )
        return response.text
