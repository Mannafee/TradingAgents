"""Async HTTP client for the OpenAI Responses API (Codex OAuth endpoint)."""

import json
import os
from typing import Any, Callable, Dict, List, Optional

import httpx

from .types import CodexAPIError, LLMResponse, Message, ToolCall


class CodexClient:
    """Async client for the Codex/OpenAI Responses API.

    Works with both:
    - ChatGPT backend (chatgpt.com/backend-api/codex/responses) via OAuth
    - Standard OpenAI API (api.openai.com/v1/responses) via API key
    """

    def __init__(
        self,
        access_token: str = "",
        base_url: str = "",
        extra_headers: Optional[Dict[str, str]] = None,
        stats_tracker: Any = None,
    ):
        self.access_token = access_token or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "CODEX_BASE_URL", "https://chatgpt.com/backend-api"
        )
        self.extra_headers = extra_headers or {}
        self.stats_tracker = stats_tracker

        # Pick up env-var overrides from OAuth flow
        if os.environ.get("CODEX_OAUTH_MODE"):
            account_id = os.environ.get("CHATGPT_ACCOUNT_ID", "")
            if account_id:
                self.extra_headers.setdefault("chatgpt-account-id", account_id)
            self.extra_headers.setdefault("OpenAI-Beta", "responses=experimental")
            self.extra_headers.setdefault("originator", "codex_cli_rs")

        # Build the responses endpoint
        if "chatgpt.com" in self.base_url:
            self.responses_url = f"{self.base_url.rstrip('/')}/codex/responses"
        else:
            self.responses_url = f"{self.base_url.rstrip('/')}/responses"

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                **self.extra_headers,
            }
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(300.0, connect=30.0),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[dict]] = None,
        model: str = "gpt-5.2-codex",
        reasoning_effort: Optional[str] = None,
    ) -> LLMResponse:
        """Send a completion request to the Responses API.

        Args:
            messages: List of Message objects.
            tools: Optional list of tool JSON schemas for the API.
            model: Model name to use.
            reasoning_effort: Optional reasoning effort level.

        Returns:
            LLMResponse with text and/or tool calls.
        """
        client = await self._get_client()

        # Build the input array and extract system instructions.
        # Some Codex backends require top-level "instructions".
        input_items = []
        system_parts: List[str] = []
        for m in messages:
            item = m.to_api_dict() if isinstance(m, Message) else m

            if isinstance(item, dict) and item.get("role") == "system":
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    system_parts.append(content.strip())
                continue

            input_items.append(item)

        instructions = "\n\n".join(system_parts).strip()
        if not instructions:
            instructions = "You are a helpful assistant."

        body: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "instructions": instructions,
            # ChatGPT Codex backend rejects requests unless storage is disabled.
            "store": False,
            # ChatGPT Codex backend requires streaming mode.
            "stream": True,
        }

        if tools:
            body["tools"] = tools

        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}

        # Some Codex backend responses can transiently return an empty/partial
        # body despite HTTP 200 when using streaming mode. Retry once.
        try:
            data = await self._post_and_collect_response(client, body)
        except CodexAPIError as exc:
            detail = (exc.detail or "").lower()
            is_transient_parse = (
                exc.status_code >= 500 and
                ("decode response json" in detail or "empty response body" in detail)
            )
            if not is_transient_parse:
                raise
            data = await self._post_and_collect_response(client, body)
        return self._parse_response(data)

    async def _post_and_collect_response(
        self, client: httpx.AsyncClient, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """POST to responses endpoint and normalize streamed/non-streamed payloads."""
        async with client.stream("POST", self.responses_url, json=body) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode("utf-8", errors="ignore")
                raise CodexAPIError(resp.status_code, detail[:500])

            content_type = (resp.headers.get("content-type") or "").lower()
            raw = await resp.aread()
            if not raw:
                raise CodexAPIError(
                    500, f"Empty response body (content-type={content_type or 'unknown'})"
                )

            text = raw.decode("utf-8", errors="ignore")

            # Prefer SSE parsing for declared or apparent event-stream payloads.
            looks_like_sse = "text/event-stream" in content_type or (
                text.lstrip().startswith("event:") or text.lstrip().startswith("data:")
            )
            if looks_like_sse:
                data = self._parse_sse_text(text)
            else:
                data = None

            # Fallback to JSON, then SSE again for mis-labeled responses.
            if data is None:
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = self._parse_sse_text(text)
                    if data is None:
                        raise CodexAPIError(
                            500,
                            "Failed to decode response JSON: "
                            f"non-JSON payload preview={text[:200]!r}",
                        )

        if not isinstance(data, dict):
            raise CodexAPIError(500, "Invalid response shape from Codex API")
        return data

    def _parse_sse_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse SSE text and return the final response object when available."""
        final_response: Optional[Dict[str, Any]] = None
        data_lines: List[str] = []

        def process_payload(payload: str):
            nonlocal final_response
            payload = payload.strip()
            if not payload or payload == "[DONE]":
                return
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                return
            if not isinstance(event, dict):
                return

            event_type = event.get("type")
            if isinstance(event.get("response"), dict):
                final_response = event["response"]
            elif "output" in event and isinstance(event.get("output"), list):
                final_response = event

            if event_type in {"response.completed", "response.done", "done"}:
                if isinstance(event.get("response"), dict):
                    final_response = event["response"]
                elif "output" in event and isinstance(event.get("output"), list):
                    final_response = event

        for raw_line in text.splitlines():
            line = raw_line.rstrip("\r")

            # Blank line ends one SSE event block.
            if line == "":
                if data_lines:
                    process_payload("\n".join(data_lines))
                    data_lines = []
                continue

            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line.startswith(":"):
                # SSE comment/heartbeat
                continue
            else:
                # Tolerate payload-only lines from non-compliant emitters.
                if line.startswith("{"):
                    process_payload(line)

        if data_lines:
            process_payload("\n".join(data_lines))

        return final_response

    async def complete_with_tools(
        self,
        messages: List[Message],
        tools: List[dict],
        tool_executor: Callable[[str, str], str],
        model: str = "gpt-5.2-codex",
        reasoning_effort: Optional[str] = None,
        max_iterations: int = 15,
    ) -> LLMResponse:
        """Run a tool-calling loop until the model produces a text response.

        This replaces the LangGraph analyst <-> ToolNode edges.

        Args:
            messages: Initial messages.
            tools: Tool JSON schemas.
            tool_executor: Callable(tool_name, arguments_json) -> result_string.
            model: Model name.
            reasoning_effort: Optional reasoning effort level.
            max_iterations: Safety limit on tool-calling rounds.

        Returns:
            Final LLMResponse (with text, no more tool calls).
        """
        current_messages = list(messages)

        for _ in range(max_iterations):
            response = await self.complete(
                current_messages, tools=tools, model=model,
                reasoning_effort=reasoning_effort,
            )

            if not response.tool_calls:
                # Model produced a text response - we're done
                return response

            # Add function_call items from the response, then their outputs.
            # The Responses API requires function_call items in the input
            # so it can match function_call_output items to their calls.
            for tc in response.tool_calls:
                if self.stats_tracker:
                    self.stats_tracker.record_tool_call()

                # Echo the model's function_call back into the conversation
                current_messages.append({
                    "type": "function_call",
                    "call_id": tc.call_id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                })

                # Execute tool and add result
                result_str = tool_executor(tc.name, tc.arguments)
                current_messages.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": result_str,
                })

        # If we hit max iterations, return whatever we have
        return response

    def _parse_response(self, data: dict) -> LLMResponse:
        """Parse a Responses API response into an LLMResponse."""
        text_parts = []
        tool_calls = []
        input_tokens = 0
        output_tokens = 0

        # Extract usage
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        if self.stats_tracker:
            self.stats_tracker.record_llm_call(input_tokens, output_tokens)

        # Parse output items
        for item in data.get("output", []):
            item_type = item.get("type", "")

            if item_type == "message":
                for content_part in item.get("content", []):
                    if content_part.get("type") == "output_text":
                        text_parts.append(content_part.get("text", ""))

            elif item_type == "function_call":
                tool_calls.append(ToolCall(
                    call_id=item.get("call_id", ""),
                    name=item.get("name", ""),
                    arguments=item.get("arguments", "{}"),
                ))

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=data,
        )
