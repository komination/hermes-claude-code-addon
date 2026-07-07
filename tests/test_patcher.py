"""Seam-patcher unit tests against fake upstream modules.

The fakes reproduce the exact shapes the patcher verifies (signatures,
source anchors, module helpers), so these tests prove the wrapper logic
without importing the heavy real modules. test_integration_hermes.py
covers the real tree.
"""

import sys
import textwrap
import types
from types import SimpleNamespace

import pytest

from hermes_claude_code import patcher
from hermes_claude_code.patcher import (
    SEAM_AGENT_INIT,
    SEAM_CONVERSATION_LOOP,
    SEAM_RUNTIME_PROVIDER,
    SEAM_RUN_AGENT,
)


@pytest.fixture(autouse=True)
def _isolated_status(monkeypatch):
    """Fake-module patches must not leak seam statuses into other tests."""
    monkeypatch.setattr(
        patcher, "_status", {seam: "pending" for seam in patcher.ALL_SEAMS}
    )
    monkeypatch.setattr(patcher, "_downstream_warned", False)


# ---------------------------------------------------------------------------
# runtime_provider seam
# ---------------------------------------------------------------------------

def _fake_runtime_provider():
    rp = types.ModuleType("fake_runtime_provider")
    rp._VALID_API_MODES = {
        "chat_completions",
        "codex_responses",
        "anthropic_messages",
        "bedrock_converse",
        "codex_app_server",
    }

    def _parse_api_mode(raw):
        if isinstance(raw, str) and raw.strip().lower() in rp._VALID_API_MODES:
            return raw.strip().lower()
        return None

    def _maybe_apply_codex_app_server_runtime(*, provider, api_mode, model_cfg):
        if model_cfg and provider in {"openai", "openai-codex"}:
            if str(model_cfg.get("openai_runtime") or "").lower() == "codex_app_server":
                return "codex_app_server"
        return api_mode

    rp._parse_api_mode = _parse_api_mode
    rp._maybe_apply_codex_app_server_runtime = _maybe_apply_codex_app_server_runtime
    return rp


class TestRuntimeProviderSeam:
    def test_adds_mode_and_wraps_gate(self, monkeypatch):
        rp = _fake_runtime_provider()
        patcher._patch_runtime_provider(rp)
        assert "claude_code" in rp._VALID_API_MODES
        assert rp._parse_api_mode("claude_code") == "claude_code"
        assert patcher.seam_status()[SEAM_RUNTIME_PROVIDER] == "patched"

        monkeypatch.setattr(patcher, "downstream_ready", lambda: True)
        mode = rp._maybe_apply_codex_app_server_runtime(
            provider="anthropic",
            api_mode="anthropic_messages",
            model_cfg={"claude_code_runtime": "claude_code"},
        )
        assert mode == "claude_code"

    def test_codex_optin_still_works(self, monkeypatch):
        rp = _fake_runtime_provider()
        patcher._patch_runtime_provider(rp)
        monkeypatch.setattr(patcher, "downstream_ready", lambda: True)
        mode = rp._maybe_apply_codex_app_server_runtime(
            provider="openai",
            api_mode="chat_completions",
            model_cfg={"openai_runtime": "codex_app_server"},
        )
        assert mode == "codex_app_server"

    def test_all_or_nothing_falls_back_when_downstream_missing(self, monkeypatch):
        rp = _fake_runtime_provider()
        patcher._patch_runtime_provider(rp)
        monkeypatch.setattr(patcher, "downstream_ready", lambda: False)
        mode = rp._maybe_apply_codex_app_server_runtime(
            provider="anthropic",
            api_mode="anthropic_messages",
            model_cfg={"claude_code_runtime": "claude_code"},
        )
        assert mode == "anthropic_messages"

    def test_idempotent(self):
        rp = _fake_runtime_provider()
        patcher._patch_runtime_provider(rp)
        wrapped_once = rp._maybe_apply_codex_app_server_runtime
        patcher._patch_runtime_provider(rp)
        assert rp._maybe_apply_codex_app_server_runtime is wrapped_once

    def test_signature_drift_fails_seam(self):
        rp = _fake_runtime_provider()

        def drifted(*, provider, api_mode):  # model_cfg dropped upstream
            return api_mode

        rp._maybe_apply_codex_app_server_runtime = drifted
        with pytest.raises(patcher.SeamError):
            patcher._patch_runtime_provider(rp)


# ---------------------------------------------------------------------------
# agent_init seam
# ---------------------------------------------------------------------------

def _fake_agent_init():
    ai = types.ModuleType("fake_agent_init")

    def init_agent(agent, base_url=None, api_key=None, provider=None,
                   api_mode=None, **kwargs):
        # Mirrors the stock acceptance set: rejects unknown modes.
        if api_mode in {"chat_completions", "codex_responses",
                        "anthropic_messages", "bedrock_converse",
                        "codex_app_server"}:
            agent.api_mode = api_mode
        else:
            agent.api_mode = "anthropic_messages"

    ai.init_agent = init_agent
    return ai


class TestAgentInitSeam:
    def test_restores_requested_claude_code_mode(self):
        ai = _fake_agent_init()
        patcher._patch_agent_init(ai)
        agent = SimpleNamespace()
        ai.init_agent(agent, provider="anthropic", api_mode="claude_code")
        assert agent.api_mode == "claude_code"
        assert patcher.seam_status()[SEAM_AGENT_INIT] == "patched"

    def test_other_modes_unaffected(self):
        ai = _fake_agent_init()
        patcher._patch_agent_init(ai)
        agent = SimpleNamespace()
        ai.init_agent(agent, provider="anthropic", api_mode="anthropic_messages")
        assert agent.api_mode == "anthropic_messages"

    def test_idempotent(self):
        ai = _fake_agent_init()
        patcher._patch_agent_init(ai)
        wrapped_once = ai.init_agent
        patcher._patch_agent_init(ai)
        assert ai.init_agent is wrapped_once


# ---------------------------------------------------------------------------
# conversation_loop seam (fake module written to disk so getsource works)
# ---------------------------------------------------------------------------

_FAKE_CONVERSATION_LOOP = textwrap.dedent(
    '''
    """Miniature stand-in for agent.conversation_loop with the exact seam shape."""

    def build_turn_context(agent, user_message, system_message,
                           conversation_history, task_id, stream_callback,
                           persist_user_message, persist_user_timestamp,
                           **helpers):
        agent.helpers_seen = sorted(helpers)
        return agent._ctx

    def _restore_or_build_system_prompt(*args, **kwargs):
        pass

    def _install_safe_stdio(*args, **kwargs):
        pass

    def _sanitize_surrogates(value):
        return value

    def _summarize_user_message_for_log(*args, **kwargs):
        pass

    def set_session_context(*args, **kwargs):
        pass

    def set_current_write_origin(*args, **kwargs):
        pass

    def _ra():
        pass

    def run_conversation(agent, user_message, system_message=None,
                         conversation_history=None, task_id=None,
                         stream_callback=None, persist_user_message=None,
                         persist_user_timestamp=None, moa_config=None):
        _ctx = build_turn_context(
            agent, user_message, system_message, conversation_history,
            task_id, stream_callback, persist_user_message,
            persist_user_timestamp,
            restore_or_build_system_prompt=_restore_or_build_system_prompt,
            install_safe_stdio=_install_safe_stdio,
            sanitize_surrogates=_sanitize_surrogates,
            summarize_user_message_for_log=_summarize_user_message_for_log,
            set_session_context=set_session_context,
            set_current_write_origin=set_current_write_origin,
            ra=_ra,
        )
        agent._auth_pool_refresh_counts = {}
        if agent.api_mode == "codex_app_server":
            return {"final_response": "codex"}
        return {"final_response": "stock", "messages": _ctx.messages}
    '''
)


@pytest.fixture()
def fake_conversation_loop(tmp_path, monkeypatch):
    name = "hcc_fake_conversation_loop"
    (tmp_path / f"{name}.py").write_text(_FAKE_CONVERSATION_LOOP)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    module = __import__(name)
    yield module
    sys.modules.pop(name, None)


def _agent_stub(api_mode="claude_code"):
    return SimpleNamespace(
        api_mode=api_mode,
        _ctx=SimpleNamespace(
            user_message="sanitized",
            original_user_message="original",
            messages=[{"role": "user", "content": "sanitized"}],
            effective_task_id="task-1",
            should_review_memory=True,
        ),
    )


class TestConversationLoopSeam:
    def test_patch_applies_on_expected_shape(self, fake_conversation_loop):
        patcher._patch_conversation_loop(fake_conversation_loop)
        assert patcher.seam_status()[SEAM_CONVERSATION_LOOP] == "patched"
        assert getattr(
            fake_conversation_loop.run_conversation,
            "__hermes_claude_code_addon__",
        )

    def test_claude_code_agent_forks_to_runtime(
        self, fake_conversation_loop, monkeypatch
    ):
        patcher._patch_conversation_loop(fake_conversation_loop)
        calls = {}

        def fake_runtime(agent, **kwargs):
            calls.update(kwargs)
            return {"final_response": "from-claude", "messages": kwargs["messages"]}

        import hermes_claude_code.runtime as runtime_mod

        monkeypatch.setattr(runtime_mod, "run_claude_code_turn", fake_runtime)

        agent = _agent_stub("claude_code")
        result = fake_conversation_loop.run_conversation(agent, "hello")

        assert result["final_response"] == "from-claude"
        assert calls["user_message"] == "sanitized"
        assert calls["original_user_message"] == "original"
        assert calls["effective_task_id"] == "task-1"
        assert calls["should_review_memory"] is True
        assert agent._auth_pool_refresh_counts == {}
        # the prologue replica passed all seven upstream helpers through
        assert agent.helpers_seen == sorted(
            [
                "restore_or_build_system_prompt",
                "install_safe_stdio",
                "sanitize_surrogates",
                "summarize_user_message_for_log",
                "set_session_context",
                "set_current_write_origin",
                "ra",
            ]
        )

    def test_other_agents_pass_through_unchanged(self, fake_conversation_loop):
        patcher._patch_conversation_loop(fake_conversation_loop)
        agent = _agent_stub("anthropic_messages")
        result = fake_conversation_loop.run_conversation(agent, "hello")
        assert result["final_response"] == "stock"

    def test_missing_helper_fails_seam(self, fake_conversation_loop):
        del fake_conversation_loop._ra
        with pytest.raises(patcher.SeamError):
            patcher._patch_conversation_loop(fake_conversation_loop)

    def test_signature_drift_fails_seam(self, fake_conversation_loop):
        original = fake_conversation_loop.run_conversation

        def drifted(agent, user_message, extra_new_param=None):
            return original(agent, user_message)

        fake_conversation_loop.run_conversation = drifted
        with pytest.raises(patcher.SeamError):
            patcher._patch_conversation_loop(fake_conversation_loop)


# ---------------------------------------------------------------------------
# run_agent seam
# ---------------------------------------------------------------------------

def _fake_run_agent():
    ra = types.ModuleType("fake_run_agent")

    class AIAgent:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    ra.AIAgent = AIAgent
    return ra


class TestRunAgentSeam:
    def test_close_drops_claude_session_first(self):
        ra = _fake_run_agent()
        patcher._patch_run_agent(ra)
        agent = ra.AIAgent()

        class _Session:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        session = _Session()
        agent._claude_session = session
        agent.close()
        assert session.closed
        assert agent._claude_session is None
        assert agent.closed  # original close still ran

    def test_close_without_session_is_fine(self):
        ra = _fake_run_agent()
        patcher._patch_run_agent(ra)
        agent = ra.AIAgent()
        agent.close()
        assert agent.closed

    def test_session_close_failure_does_not_block_teardown(self):
        ra = _fake_run_agent()
        patcher._patch_run_agent(ra)
        agent = ra.AIAgent()
        agent._claude_session = SimpleNamespace(
            close=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        agent.close()  # must not raise
        assert agent.closed

    def test_idempotent(self):
        ra = _fake_run_agent()
        patcher._patch_run_agent(ra)
        wrapped_once = ra.AIAgent.close
        patcher._patch_run_agent(ra)
        assert ra.AIAgent.close is wrapped_once
