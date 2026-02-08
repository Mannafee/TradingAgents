"""Signal extraction - extracts BUY/SELL/HOLD from full decision text."""

from tradingagents.client.codex import CodexClient
from tradingagents.client.types import Message


SYSTEM_PROMPT = "You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: SELL, BUY, or HOLD. Provide only the extracted decision (SELL, BUY, or HOLD) as your output, without adding any additional text or information."


async def extract_signal(client: CodexClient, model: str, full_signal: str) -> str:
    """Extract a BUY/SELL/HOLD signal from a full decision text.

    Args:
        client: CodexClient instance.
        model: Model to use (quick-think).
        full_signal: Full text of the trading decision.

    Returns:
        One of "BUY", "SELL", or "HOLD".
    """
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=full_signal),
    ]
    response = await client.complete(messages, model=model)
    return response.text.strip()
