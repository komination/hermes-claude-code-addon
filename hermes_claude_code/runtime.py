"""Claude Code CLI runtime — hands whole turns to the real `claude` binary.

Sibling of ``agent/codex_runtime.py``'s app-server path, for the
``claude_code`` api_mode: instead of speaking the Anthropic API directly
(anthropic_messages, which Anthropic bills to the metered "extra usage"
lane for third-party clients), each Hermes turn is delegated to a spawned
`claude -p` subprocess authenticated with the user's Claude Code OAuth
token — the first-party path that consumes Pro/Max plan quota.

Enabled via config: ``model.provider: anthropic`` +
``model.claude_code_runtime: claude_code``. Like the codex app-server
runtime, this path bypasses the transport registry entirely; the addon's
patcher (hermes_claude_code/patcher.py) forks run_conversation() to here
when agent.api_mode == "claude_code" — the upstream Hermes checkout is
never modified on disk.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Env vars forwarded into the hermes-tools MCP grandchild. Deliberately an
# allowlist — HERMES_* also covers Tier-1 secrets (dashboard session token)
# that must not be persisted into the mcp-config JSON on disk.
_MCP_ENV_FORWARD_EXACT = ("HERMES_HOME",)
_MCP_ENV_FORWARD_PREFIXES = ("HERMES_KANBAN_", "HERMES_SESSION_")

_RUNTIME_DISABLE_HINT = (
    "Remove `model.claude_code_runtime` from config.yaml (or set it to "
    "`auto`) to fall back to the default runtime."
)


def _claude_event_to_tool_progress(event: dict) -> list[tuple[str, str, dict]]:
    """Map a stream-json `assistant` event's tool_use blocks to Hermes
    tool-progress tuples ``(tool_name, preview, args)`` so gateways show
    "running X" breadcrumbs on this route, same as every other provider."""
    if not isinstance(event, dict) or event.get("type") != "assistant":
        return []
    content = (event.get("message") or {}).get("content")
    if not isinstance(content, list):
        return []
    mapped: list[tuple[str, str, dict]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name") or "unknown"
        args = block.get("input")
        if not isinstance(args, dict):
            args = {"input": args}
        if name == "Bash":
            preview = str(args.get("command") or "")
        elif name in {"Read", "Write", "Edit"}:
            preview = str(args.get("file_path") or "")
        else:
            preview = name
        mapped.append((name, preview, args))
    return mapped


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _record_claude_code_usage(agent, turn) -> dict[str, Any]:
    """Translate Claude Code result usage into Hermes accounting.

    The terminal `result` event reports Anthropic-shaped usage:
    input_tokens (uncached), cache_read_input_tokens,
    cache_creation_input_tokens, output_tokens. Turns on this runtime are
    covered by the user's plan quota, so the billing mode is recorded as
    subscription_included and the incremental cost is zero — the
    total_cost_usd the CLI reports is the nominal API-equivalent value,
    not money spent.
    """
    agent.session_api_calls += 1

    usage = getattr(turn, "usage", None)
    if not isinstance(usage, dict) or not usage:
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=agent.model,
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "claude_code api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        return {}

    from agent.usage_pricing import CanonicalUsage

    canonical_usage = CanonicalUsage(
        input_tokens=_coerce_usage_int(usage.get("input_tokens")),
        output_tokens=_coerce_usage_int(usage.get("output_tokens")),
        cache_read_tokens=_coerce_usage_int(usage.get("cache_read_input_tokens")),
        cache_write_tokens=_coerce_usage_int(usage.get("cache_creation_input_tokens")),
        reasoning_tokens=0,
        raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": 0,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
        except Exception:
            logger.debug("claude_code usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens

    agent.session_cost_status = "included"
    agent.session_cost_source = "claude_code_cli"

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                estimated_cost_usd=0.0,
                cost_status="included",
                cost_source="claude_code_cli",
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included",
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "claude_code token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "estimated_cost_usd": 0.0,
        "cost_status": "included",
        "cost_source": "claude_code_cli",
    }


def _ensure_hermes_tools_mcp_config(agent) -> str | None:
    """Write (once per agent) the mcp-config JSON that exposes Hermes'
    stateless tool surface (hermes_tools_mcp_server) to the spawned claude
    binary via `--mcp-config <path> --strict-mcp-config`.

    Returns the config path, or None when it could not be built (non-fatal:
    the runtime still works with Claude's built-in tools only)."""
    cached = getattr(agent, "_claude_mcp_config_path", None)
    if cached and os.path.isfile(cached):
        return cached
    try:
        # The addon lives outside the Hermes checkout, so the repo root must
        # be derived from the installed `agent` package, not this file.
        import agent as _hermes_agent_pkg

        repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(_hermes_agent_pkg.__file__))
        )
        env: dict[str, str] = {}
        for key, value in os.environ.items():
            if key in _MCP_ENV_FORWARD_EXACT or key.startswith(
                _MCP_ENV_FORWARD_PREFIXES
            ):
                env[key] = value
        # The grandchild runs `python -m agent.transports...` from an
        # arbitrary cwd — PYTHONPATH must carry the Hermes checkout.
        pythonpath_parts = [repo_root] + [
            p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if p and p != repo_root
        ]
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        env["HERMES_QUIET"] = "1"
        env.setdefault("HERMES_REDACT_SECRETS", "true")

        config = {
            "mcpServers": {
                "hermes-tools": {
                    "command": sys.executable,
                    "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
                    "env": env,
                }
            }
        }
        fd, path = tempfile.mkstemp(prefix="hermes-claude-mcp-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        agent._claude_mcp_config_path = path
        return path
    except Exception:
        logger.debug("claude_code mcp-config build failed", exc_info=True)
        return None


def _assert_plan_billing(turn) -> None:
    """Guard layer B (runtime detection): a turn that authenticated via
    ANTHROPIC_API_KEY must never be treated as a success. The session
    already kills the child and errors the turn; this backstop makes the
    invariant explicit at the runtime layer too."""
    if getattr(turn, "billing_lane", None) == "api_key" and not turn.error:
        turn.error = (
            "claude_code runtime: turn billed to ANTHROPIC_API_KEY lane "
            "(plan-billing guard). " + _RUNTIME_DISABLE_HINT
        )
        turn.should_retire = True


def run_claude_code_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """claude_code runtime path. Hands the entire turn to a `claude -p`
    subprocess and projects its events back into Hermes' messages list so
    memory/skill review keep working.

    Called from run_conversation() when agent.api_mode == "claude_code".
    Returns the same dict shape as the chat_completions path.
    """
    from .session import ClaudeCodeSession

    # Lazy session: one ClaudeCodeSession per AIAgent instance. Holds the
    # resume-chain session id in memory; no persistent subprocess.
    if getattr(agent, "_claude_session", None) is None:
        from .cli import check_claude_binary

        ok, version_or_message = check_claude_binary()
        if not ok:
            return {
                "final_response": (
                    f"claude_code runtime unavailable: {version_or_message}. "
                    + _RUNTIME_DISABLE_HINT
                ),
                "messages": messages,
                "api_calls": 0,
                "completed": False,
                "partial": True,
                "error": version_or_message,
            }

        from agent.runtime_cwd import resolve_agent_cwd

        cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())

        def _on_claude_event(event: dict) -> None:
            progress_callback = getattr(agent, "tool_progress_callback", None)
            if progress_callback is None:
                return
            for tool_name, preview, args in _claude_event_to_tool_progress(event):
                try:
                    progress_callback("tool.started", tool_name, preview, args)
                except Exception:
                    logger.debug(
                        "claude_code tool-progress callback raised", exc_info=True
                    )

        agent._claude_session = ClaudeCodeSession(
            cwd=cwd,
            model=agent.model,
            mcp_config_path=_ensure_hermes_tools_mcp_config(agent),
            on_event=_on_claude_event,
        )

    # NOTE: the user message is ALREADY appended to messages by the
    # standard run_conversation() flow before the early return reaches us.
    # Do NOT append again — that would duplicate.

    try:
        turn = agent._claude_session.run_turn(
            user_input=user_message,
            interrupt_check=lambda: bool(
                getattr(agent, "_interrupt_requested", False)
            ),
        )
    except Exception as exc:
        logger.exception("claude_code turn failed")
        try:
            agent._claude_session.close()
        except Exception:
            pass
        agent._claude_session = None
        return {
            "final_response": (
                f"claude_code turn failed: {exc}. " + _RUNTIME_DISABLE_HINT
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
        }

    _assert_plan_billing(turn)

    # Retire on request: unlike codex there is no long-lived subprocess,
    # but a retire signal still means per-session state (resume id) is
    # suspect — drop the session object so the next turn starts clean.
    if getattr(turn, "should_retire", False):
        logger.warning(
            "claude_code session retired (turn error: %s)", turn.error
        )
        try:
            agent._claude_session.close()
        except Exception:
            pass
        agent._claude_session = None

    # Splice projected messages into the conversation. Same persistence
    # contract as the codex app-server early-return path: we flush the new
    # rows ourselves (idempotent via _DB_PERSISTED_MARKER) and report
    # agent_persisted=True so the gateway skips its own DB write.
    if turn.projected_messages:
        messages.extend(turn.projected_messages)
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "claude_code projected-message flush failed", exc_info=True
                )

    # Counter ticks for the agent-improvement loop. _turns_since_memory and
    # _user_turn_count are already incremented in the run_conversation()
    # pre-loop block; only _iters_since_skill needs explicit bumping here.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    usage_result = _record_claude_code_usage(agent, turn)
    api_calls = 1

    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider sync. Skipped on interrupt/error to avoid
    # feeding partial transcripts to memory.
    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    final_response = turn.final_text
    if turn.error and not final_response:
        final_response = f"{turn.error}\n\n{_RUNTIME_DISABLE_HINT}"

    return {
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_calls,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        "agent_persisted": True,
        "claude_session_id": turn.claude_session_id,
        "billing_lane": turn.billing_lane,
        **usage_result,
    }


__all__ = [
    "run_claude_code_turn",
    "_record_claude_code_usage",
    "_ensure_hermes_tools_mcp_config",
    "_assert_plan_billing",
]
