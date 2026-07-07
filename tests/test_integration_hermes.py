"""Integration: apply the patcher to the REAL Hermes tree.

Requires the Hermes checkout + its venv site-packages on sys.path (the
Makefile test target wires this). Verifies that every seam check passes
against the pinned upstream and that the gate chain works end-to-end at
the api_mode-resolution level (no network, no agent construction).
"""

import pytest

pytest.importorskip("hermes_cli.runtime_provider")

from hermes_claude_code import patcher  # noqa: E402


def _install_and_ready():
    patcher.install()
    assert patcher.downstream_ready(), patcher.seam_status()


class TestRealSeams:
    def test_all_seams_patch_clean(self):
        _install_and_ready()
        import agent.agent_init as ai
        import agent.conversation_loop as cl
        import hermes_cli.runtime_provider as rp
        import run_agent as ra

        assert "claude_code" in rp._VALID_API_MODES
        assert rp._parse_api_mode("claude_code") == "claude_code"
        for func in (
            rp._maybe_apply_codex_app_server_runtime,
            ai.init_agent,
            cl.run_conversation,
            ra.AIAgent.close,
        ):
            assert getattr(func, "__hermes_claude_code_addon__", False), func

        status = patcher.seam_status()
        assert all(state == "patched" for state in status.values()), status

    def test_gate_rewrites_via_real_codex_wrapper(self):
        _install_and_ready()
        import hermes_cli.runtime_provider as rp

        mode = rp._maybe_apply_codex_app_server_runtime(
            provider="anthropic",
            api_mode="anthropic_messages",
            model_cfg={"claude_code_runtime": "claude_code"},
        )
        assert mode == "claude_code"

    def test_gate_inert_without_optin(self):
        _install_and_ready()
        import hermes_cli.runtime_provider as rp

        mode = rp._maybe_apply_codex_app_server_runtime(
            provider="anthropic",
            api_mode="anthropic_messages",
            model_cfg={},
        )
        assert mode == "anthropic_messages"

    def test_original_run_conversation_reachable(self):
        """The wrapper must keep a handle to the untouched original so
        non-claude_code agents run byte-identical upstream code."""
        _install_and_ready()
        import agent.conversation_loop as cl

        original = cl.run_conversation.__hermes_claude_code_original__
        assert callable(original)
        assert not getattr(original, "__hermes_claude_code_addon__", False)
