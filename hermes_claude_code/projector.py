"""Projects Claude Code stream-json events into Hermes' messages list.

Sibling of codex_event_projector.py for the `claude_code` runtime: converts
the NDJSON events emitted by `claude -p --output-format stream-json
--verbose` into the standard OpenAI-shaped `{role, content, tool_calls,
tool_call_id}` entries that agent/curator.py and the sessions DB already
understand.

Event shapes (verified live against Claude Code 2.1.201):

  {"type": "system", "subtype": "init", "session_id", "apiKeySource",
   "model", ...}                          → metadata only (billing guard!)
  {"type": "system", "subtype": "thinking_tokens" | ...}  → ignored
  {"type": "assistant", "message": {"content": [
       {"type": "text", "text"} | {"type": "thinking", "thinking"} |
       {"type": "tool_use", "id", "name", "input"}], ...}}
                                          → assistant message (+tool_calls)
  {"type": "user", "message": {"content": [
       {"type": "tool_result", "tool_use_id", "content", "is_error"}]}}
                                          → tool result message(s)
  {"type": "result", "subtype": "success" | "error_*", "is_error",
   "result", "session_id", "usage", "total_cost_usd", "num_turns",
   "permission_denials", ...}             → terminal metadata + final text

The projector is stateful across one turn: it stashes thinking text onto the
next assistant message (same convention as the codex projector) and records
turn-level metadata (session_id, apiKeySource, usage, cost, error state)
that ClaudeCodeSession reads after/while consuming events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProjectionResult:
    """Output of projecting one stream-json event.

    `messages` may hold several entries (one assistant tool_call message, or
    multiple tool results from a single `user` event). `tool_iterations`
    counts completed tool results in this event — the caller adds it to the
    skill-nudge counter."""

    messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    final_text: Optional[str] = None


def _flatten_tool_result_content(content: Any) -> str:
    """Collapse a tool_result content payload (str | list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(str(block.get("text") or ""))
                elif block_type == "image":
                    parts.append("[image]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


class ClaudeCodeEventProjector:
    """Stateful projector consuming Claude Code events in arrival order."""

    def __init__(self) -> None:
        self._pending_reasoning: list[str] = []
        # Turn-level metadata captured from system:init / result events.
        self.session_id: Optional[str] = None
        self.api_key_source: Optional[str] = None
        self.model: Optional[str] = None
        self.saw_result = False
        self.result_subtype: Optional[str] = None
        self.result_is_error = False
        self.result_text: Optional[str] = None
        self.usage: Optional[dict[str, Any]] = None
        self.total_cost_usd: Optional[float] = None
        self.num_turns: Optional[int] = None
        self.permission_denials: list[dict] = []

    def project(self, event: dict) -> ProjectionResult:
        event_type = event.get("type") or ""
        if event_type == "system":
            return self._project_system(event)
        if event_type == "assistant":
            return self._project_assistant(event)
        if event_type == "user":
            return self._project_user(event)
        if event_type == "result":
            return self._project_result(event)
        # stream_event (partial deltas) and future event types: display-only.
        return ProjectionResult()

    # ---------- per-type projections ----------

    def _project_system(self, event: dict) -> ProjectionResult:
        if event.get("subtype") == "init":
            self.session_id = event.get("session_id") or self.session_id
            self.api_key_source = event.get("apiKeySource")
            self.model = event.get("model")
        # thinking_tokens / status / other system chatter never becomes a message.
        return ProjectionResult()

    def _project_assistant(self, event: dict) -> ProjectionResult:
        message = event.get("message") or {}
        content = message.get("content")
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text") or ""
                    if text:
                        text_parts.append(text)
                elif block_type == "thinking":
                    thinking = block.get("thinking") or ""
                    if thinking:
                        self._pending_reasoning.append(thinking)
                elif block_type == "tool_use":
                    args = block.get("input")
                    if not isinstance(args, dict):
                        args = {"input": args}
                    tool_calls.append(
                        {
                            # Anthropic tool_use ids ("toolu_...") are already
                            # unique and stable — reuse them directly so the
                            # projected pair correlates deterministically.
                            "id": block.get("id") or f"claude_tool_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": block.get("name") or "unknown",
                                "arguments": json.dumps(
                                    args, ensure_ascii=False, sort_keys=True
                                ),
                            },
                        }
                    )

        text = "\n".join(text_parts).strip()
        if not text and not tool_calls:
            return ProjectionResult()

        msg: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if self._pending_reasoning:
            msg["reasoning"] = "\n".join(self._pending_reasoning)
            self._pending_reasoning = []
        # Claude can emit several assistant messages per turn (text, then
        # tool calls, then closing text). The last text-only one is the
        # provisional final answer; the terminal `result` event overrides.
        final_text = text if (text and not tool_calls) else None
        return ProjectionResult(messages=[msg], final_text=final_text)

    def _project_user(self, event: dict) -> ProjectionResult:
        """Tool results come back as synthetic user turns. Project ONLY
        tool_result blocks — a plain-text user event would duplicate the
        prompt Hermes already appended to its own messages list."""
        message = event.get("message") or {}
        content = message.get("content")
        if not isinstance(content, list):
            return ProjectionResult()
        messages: list[dict] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = _flatten_tool_result_content(block.get("content"))
            if block.get("is_error"):
                text = f"[error] {text}" if text else "[error]"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or "",
                    "content": text,
                }
            )
        return ProjectionResult(messages=messages, tool_iterations=len(messages))

    def _project_result(self, event: dict) -> ProjectionResult:
        self.saw_result = True
        self.result_subtype = event.get("subtype")
        self.result_is_error = bool(event.get("is_error"))
        self.session_id = event.get("session_id") or self.session_id
        usage = event.get("usage")
        if isinstance(usage, dict):
            self.usage = usage
        cost = event.get("total_cost_usd")
        if isinstance(cost, (int, float)):
            self.total_cost_usd = float(cost)
        num_turns = event.get("num_turns")
        if isinstance(num_turns, int):
            self.num_turns = num_turns
        denials = event.get("permission_denials")
        if isinstance(denials, list):
            self.permission_denials = denials
        result_text = event.get("result")
        if isinstance(result_text, str) and result_text:
            self.result_text = result_text
        # The final assistant text was already projected from its own
        # `assistant` event — emit no message here to avoid duplication.
        final_text = self.result_text if not self.result_is_error else None
        return ProjectionResult(final_text=final_text)


__all__ = ["ClaudeCodeEventProjector", "ProjectionResult"]
