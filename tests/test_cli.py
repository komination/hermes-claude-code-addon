"""Tests for the claude_code CLI transport (env guard, argv, version parse)."""

import pytest

from hermes_claude_code.cli import (
    ClaudeCodeAuthError,
    ClaudeCodeBillingLeakError,
    ClaudeCodeCLI,
    build_claude_child_env,
    parse_claude_version,
)


class TestParseClaudeVersion:
    def test_parses_real_output(self):
        assert parse_claude_version("2.1.201 (Claude Code)") == (2, 1, 201)

    def test_garbage_returns_none(self):
        assert parse_claude_version("not a version") is None
        assert parse_claude_version("") is None


class TestBuildClaudeChildEnv:
    def test_strips_api_key_and_injects_oauth_token(self, monkeypatch):
        """The $1,800 footgun guard: a parent ANTHROPIC_API_KEY must never
        reach the child, and the OAuth token must be the only credential."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-SHOULD-NOT-LEAK")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "should-not-leak")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil.example")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test-token")

        env = build_claude_child_env()

        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert "ANTHROPIC_BASE_URL" not in env
        assert "ANTHROPIC_BEDROCK_BASE_URL" not in env
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-test-token"

    def test_explicit_token_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-env")
        env = build_claude_child_env(oauth_token="sk-ant-oat01-explicit")
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-explicit"

    def test_strips_nested_claude_markers(self, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        env = build_claude_child_env()
        assert "CLAUDECODE" not in env
        assert "CLAUDE_CODE_ENTRYPOINT" not in env

    def test_strips_nested_session_markers_by_prefix(self, monkeypatch):
        """When Hermes itself runs inside a Claude Code session, the parent
        env carries session markers that must not reach the spawned child —
        an inherited CLAUDE_CODE_SESSION_ID would fight the explicit
        --session-id/--resume argv. Only the OAuth token survives."""
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent-session")
        monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "1")
        monkeypatch.setenv("CLAUDE_CODE_EXECPATH", "/some/path")
        monkeypatch.setenv("CLAUDE_EFFORT", "xhigh")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        env = build_claude_child_env()
        leaked = [k for k in env if k.startswith(("CLAUDE_CODE_", "CLAUDECODE", "CLAUDE_EFFORT"))]
        assert leaked == ["CLAUDE_CODE_OAUTH_TOKEN"], leaked
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-test"

    def test_no_token_raises_auth_error(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "hermes_claude_code.cli._oauth_token_from_credentials_file",
            lambda: None,
        )
        with pytest.raises(ClaudeCodeAuthError):
            build_claude_child_env()


class TestBuildArgv:
    def test_first_turn_uses_session_id(self):
        cli = ClaudeCodeCLI(cwd="/tmp")
        argv = cli.build_argv(
            "hello", session_id="abc-123", model="claude-sonnet-4-6"
        )
        assert argv[:3] == ["claude", "-p", "hello"]
        assert "--output-format" in argv and "stream-json" in argv
        assert "--verbose" in argv
        assert ["--session-id", "abc-123"] == argv[
            argv.index("--session-id"): argv.index("--session-id") + 2
        ]
        assert "--resume" not in argv
        assert "--dangerously-skip-permissions" in argv

    def test_resume_wins_over_session_id(self):
        cli = ClaudeCodeCLI(cwd="/tmp")
        argv = cli.build_argv("hi", session_id="new-id", resume="old-id")
        assert "--resume" in argv
        assert "--session-id" not in argv

    def test_mcp_config_is_strict(self):
        cli = ClaudeCodeCLI(cwd="/tmp")
        argv = cli.build_argv("hi", mcp_config_path="/tmp/mcp.json")
        idx = argv.index("--mcp-config")
        assert argv[idx + 1] == "/tmp/mcp.json"
        assert "--strict-mcp-config" in argv

    def test_run_refuses_api_key_in_env(self):
        cli = ClaudeCodeCLI(cwd="/tmp")
        with pytest.raises(ClaudeCodeBillingLeakError):
            list(
                cli.run_print_turn(
                    "hi", env={"ANTHROPIC_API_KEY": "sk-ant-api"}
                )
            )
