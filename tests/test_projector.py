"""Tests for the claude_code stream-json → Hermes messages projector.

Event fixtures mirror shapes recorded live from Claude Code 2.1.201
(`claude -p ... --output-format stream-json --verbose`).
"""

from hermes_claude_code.projector import ClaudeCodeEventProjector


def _init_event(api_key_source="none", session_id="sid-1"):
    return {
        "type": "system",
        "subtype": "init",
        "session_id": session_id,
        "apiKeySource": api_key_source,
        "model": "claude-haiku-4-5-20251001",
    }


def _assistant_text(text):
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _assistant_tool_use(tool_id="toolu_01", name="Bash", inp=None):
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tool_id, "name": name,
                 "input": inp or {"command": "echo hi"}},
            ],
        },
    }


def _tool_result(tool_id="toolu_01", content="hi", is_error=None):
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id,
                 "content": content, "is_error": is_error},
            ],
        },
    }


def _result_event(**overrides):
    event = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "final answer",
        "session_id": "sid-1",
        "num_turns": 2,
        "total_cost_usd": 0.0123,
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 700,
            "cache_read_input_tokens": 1700,
            "output_tokens": 57,
        },
        "permission_denials": [],
    }
    event.update(overrides)
    return event


class TestSystemInit:
    def test_captures_metadata_no_messages(self):
        projector = ClaudeCodeEventProjector()
        projection = projector.project(_init_event())
        assert projection.messages == []
        assert projector.session_id == "sid-1"
        assert projector.api_key_source == "none"

    def test_api_key_source_surfaces_billing_leak(self):
        projector = ClaudeCodeEventProjector()
        projector.project(_init_event(api_key_source="ANTHROPIC_API_KEY"))
        assert projector.api_key_source == "ANTHROPIC_API_KEY"

    def test_thinking_tokens_chatter_ignored(self):
        projector = ClaudeCodeEventProjector()
        projection = projector.project(
            {"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 5}
        )
        assert projection.messages == []


class TestAssistantProjection:
    def test_text_message(self):
        projector = ClaudeCodeEventProjector()
        projection = projector.project(_assistant_text("hello there"))
        assert projection.messages == [
            {"role": "assistant", "content": "hello there"}
        ]
        assert projection.final_text == "hello there"

    def test_tool_use_becomes_tool_call(self):
        projector = ClaudeCodeEventProjector()
        projection = projector.project(_assistant_tool_use())
        (msg,) = projection.messages
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        (call,) = msg["tool_calls"]
        assert call["id"] == "toolu_01"
        assert call["function"]["name"] == "Bash"
        assert '"command"' in call["function"]["arguments"]
        assert projection.final_text is None

    def test_thinking_stashed_on_next_message(self):
        projector = ClaudeCodeEventProjector()
        projector.project(
            {
                "type": "assistant",
                "message": {"content": [{"type": "thinking", "thinking": "hmm"}]},
            }
        )
        projection = projector.project(_assistant_text("done"))
        assert projection.messages[0]["reasoning"] == "hmm"


class TestToolResultProjection:
    def test_tool_result_becomes_tool_message(self):
        projector = ClaudeCodeEventProjector()
        projection = projector.project(_tool_result(content="hi"))
        (msg,) = projection.messages
        assert msg == {"role": "tool", "tool_call_id": "toolu_01", "content": "hi"}
        assert projection.tool_iterations == 1

    def test_error_result_prefixed(self):
        projector = ClaudeCodeEventProjector()
        projection = projector.project(
            _tool_result(content="requires approval", is_error=True)
        )
        assert projection.messages[0]["content"].startswith("[error]")

    def test_list_content_flattened(self):
        projector = ClaudeCodeEventProjector()
        projection = projector.project(
            _tool_result(content=[{"type": "text", "text": "part1"},
                                  {"type": "text", "text": "part2"}])
        )
        assert projection.messages[0]["content"] == "part1\npart2"

    def test_plain_text_user_event_not_duplicated(self):
        """A plain user prompt echo must not re-enter the messages list —
        Hermes already appended the user turn itself."""
        projector = ClaudeCodeEventProjector()
        projection = projector.project(
            {"type": "user", "message": {"role": "user", "content": "raw prompt"}}
        )
        assert projection.messages == []


class TestResultProjection:
    def test_success_captures_usage_and_final_text(self):
        projector = ClaudeCodeEventProjector()
        projection = projector.project(_result_event())
        assert projection.messages == []  # no duplicate assistant message
        assert projection.final_text == "final answer"
        assert projector.saw_result
        assert not projector.result_is_error
        assert projector.usage["output_tokens"] == 57
        assert projector.total_cost_usd == 0.0123
        assert projector.num_turns == 2

    def test_error_result(self):
        projector = ClaudeCodeEventProjector()
        projection = projector.project(
            _result_event(subtype="error_during_execution", is_error=True,
                          result="out of extra usage")
        )
        assert projector.result_is_error
        assert projection.final_text is None


class TestFullTurn:
    def test_recorded_turn_shape(self):
        """Replay of the live-recorded tool-use turn: init → text →
        tool_use → tool_result(error) → text → result."""
        projector = ClaudeCodeEventProjector()
        events = [
            _init_event(),
            _assistant_text("Let me check."),
            _assistant_tool_use(),
            _tool_result(content="requires approval", is_error=True),
            _assistant_text("I could not run it."),
            _result_event(result="I could not run it."),
        ]
        messages = []
        tool_iterations = 0
        final_text = None
        for event in events:
            projection = projector.project(event)
            messages.extend(projection.messages)
            tool_iterations += projection.tool_iterations
            if projection.final_text is not None:
                final_text = projection.final_text

        roles = [m["role"] for m in messages]
        assert roles == ["assistant", "assistant", "tool", "assistant"]
        assert tool_iterations == 1
        assert final_text == "I could not run it."
        # tool_call / tool_result correlation
        call_id = messages[1]["tool_calls"][0]["id"]
        assert messages[2]["tool_call_id"] == call_id
