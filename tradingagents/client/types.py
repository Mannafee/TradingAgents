from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: Optional[str] = None  # For tool result messages

    def to_api_dict(self) -> dict:
        """Convert to Responses API input message format."""
        if self.role == "system":
            return {"role": "system", "content": self.content}
        elif self.role == "user":
            return {"role": "user", "content": self.content}
        elif self.role == "assistant":
            return {"role": "assistant", "content": self.content}
        elif self.role == "tool":
            return {
                "type": "function_call_output",
                "call_id": self.tool_call_id,
                "output": self.content,
            }
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: str  # JSON string


@dataclass
class LLMResponse:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Optional[Dict[str, Any]] = None


class CodexAPIError(Exception):
    """Raised when the Codex API returns an error."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Codex API error ({status_code}): {detail}")


class CodexAuthError(Exception):
    """Raised when authentication fails."""
    pass
