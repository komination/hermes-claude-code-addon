"""Tests for ClaudeCodeSession turn orchestration with a stubbed CLI."""

from typing import Optional

import pytest

from hermes_claude_code.session import ClaudeCodeSession, TurnResult


def _init_event(api_key_source="none", session_id="sid-1"):
    return {
        "type": "system",
        "subtype": "init",
        "session_id": session_id,
        "apiKeySource": api_key_source,
    }


def _result_event(**overrides):
    event = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "hello!",
        "session_id": "sid-1",
        "usage": {"input_tokens": 5, "output_tokens": 9},
        "total_cost_usd": 0.001,
        "num_turns": 1,
    }
    event.update(overrides)
    return event


class _FakeCLI:
    """Stands in for ClaudeCodeCLI: replays canned events, records argv-ish
    kwargs, and mimics the diagnostics surface the session reads."""

    def __init__(self, events, returncode=0, raise_timeout=False):
        self._events = events
        self.last_returncode = returncode
        self.timed_out = False
        self.interrupted = False
        self.aborted = False
        self.raise_timeout = raise_timeout
        self.calls = []

    def run_print_turn(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.raise_timeout:
            self.timed_out = True
            raise TimeoutError("deadline")
        for event in self._events:
            yield event

    def stderr_tail(self, n=12):
        return []

    def abort(self):
        self.aborted = True


def _make_session(fake_cli) -> ClaudeCodeSession:
    session = ClaudeCodeSession(cwd="/tmp", model="claude-test")
    session._cli = fake_cli
    return session


@pytest.fixture(autouse=True)
def _fake_env(monkeypatch):
    """Session turns must never depend on real credentials in tests."""
    monkeypatch.setattr(
        "hermes_claude_code.session.build_claude_child_env",
        lambda **kwargs: {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"},
    )


class TestSuccessTurn:
    def test_success_sets_final_text_and_resume_chain(self):
        fake = _FakeCLI([
            _init_event(),
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "hello!"}]},
            },
            _result_event(),
        ])
        session = _make_session(fake)
        turn = session.run_turn("hi")

        assert turn.error is None
        assert turn.final_text == "hello!"
        assert turn.billing_lane == "plan"
        assert turn.claude_session_id == "sid-1"
        assert turn.usage == {"input_tokens": 5, "output_tokens": 9}
        # First turn passed a generated --session-id, not --resume.
        assert fake.calls[0]["resume"] is None
        assert fake.calls[0]["session_id"]

    def test_second_turn_resumes(self):
        fake = _FakeCLI([_init_event(), _result_event()])
        session = _make_session(fake)
        session.run_turn("first")
        session.run_turn("second")
        assert fake.calls[1]["resume"] == "sid-1"
        assert fake.calls[1]["session_id"] is None


class TestBillingGuard:
    def test_api_key_source_kills_turn(self):
        fake = _FakeCLI([
            _init_event(api_key_source="ANTHROPIC_API_KEY"),
            _result_event(),  # must never be reached
        ])
        session = _make_session(fake)
        turn = session.run_turn("hi")

        assert turn.billing_lane == "api_key"
        assert turn.should_retire
        assert turn.error and "ANTHROPIC_API_KEY" in turn.error
        assert fake.aborted

    def test_extra_usage_error_tagged(self):
        fake = _FakeCLI([
            _init_event(),
            _result_event(
                subtype="error_during_execution",
                is_error=True,
                result="API Error: out of extra usage",
            ),
        ])
        session = _make_session(fake)
        turn = session.run_turn("hi")
        assert turn.billing_lane == "extra_usage"
        assert turn.error


class TestErrorPaths:
    def test_timeout_retires(self):
        fake = _FakeCLI([], raise_timeout=True)
        session = _make_session(fake)
        turn = session.run_turn("hi", turn_timeout=1.0)
        assert turn.interrupted
        assert turn.should_retire
        assert "timed out" in turn.error

    def test_no_result_event_is_error(self):
        fake = _FakeCLI([_init_event()], returncode=1)
        session = _make_session(fake)
        turn = session.run_turn("hi")
        assert turn.error and "without a result event" in turn.error
        assert turn.should_retire

    def test_resume_miss_resets_chain(self):
        fake = _FakeCLI([_init_event(), _result_event()])
        session = _make_session(fake)
        session.run_turn("first")
        assert session._claude_session_id == "sid-1"

        session._cli = _FakeCLI([
            _result_event(
                subtype="error_during_execution",
                is_error=True,
                result="No conversation found with session ID sid-1",
            ),
        ])
        turn = session.run_turn("second")
        assert turn.error
        assert session._claude_session_id is None

    def test_env_failure_is_soft_error(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_claude_code.session.build_claude_child_env",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no token")),
        )
        session = _make_session(_FakeCLI([]))
        turn = session.run_turn("hi")
        assert turn.error and "no token" in turn.error
        assert not turn.projected_messages
