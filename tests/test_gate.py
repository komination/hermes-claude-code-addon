"""Tests for the pure claude_code opt-in gate."""

from hermes_claude_code.gate import maybe_apply_claude_code_runtime


class TestClaudeCodeRuntimeGate:
    def test_opt_in_rewrites_anthropic_api_mode(self):
        api_mode = maybe_apply_claude_code_runtime(
            provider="anthropic",
            api_mode="anthropic_messages",
            model_cfg={"claude_code_runtime": "claude_code"},
        )
        assert api_mode == "claude_code"

    def test_unset_key_is_noop(self):
        for cfg in ({}, {"claude_code_runtime": ""}, {"claude_code_runtime": "auto"}, None):
            assert maybe_apply_claude_code_runtime(
                provider="anthropic",
                api_mode="anthropic_messages",
                model_cfg=cfg,
            ) == "anthropic_messages"

    def test_non_anthropic_provider_is_noop(self):
        for provider in ("openai", "openrouter", "custom", "openai-codex"):
            assert maybe_apply_claude_code_runtime(
                provider=provider,
                api_mode="chat_completions",
                model_cfg={"claude_code_runtime": "claude_code"},
            ) == "chat_completions"

    def test_case_insensitive_value(self):
        assert maybe_apply_claude_code_runtime(
            provider="anthropic",
            api_mode="anthropic_messages",
            model_cfg={"claude_code_runtime": "Claude_Code"},
        ) == "claude_code"
