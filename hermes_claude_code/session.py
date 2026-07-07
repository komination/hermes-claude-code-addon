"""Session adapter for the claude_code runtime.

Owns the resume chain for one Hermes session: the first turn generates a
fresh UUID and passes `--session-id`, later turns pass `--resume` so the
Claude Code binary rehydrates its own transcript. Hermes' canonical
history remains SessionDB / agent._session_messages — Claude Code's
transcript is just the mechanism that gives the child context.

Structure mirrors codex_app_server_session.py, minus everything that only
exists because codex is a long-lived JSON-RPC server (approval bridging,
server-initiated requests, watchdogs): `claude -p` is a one-shot child per
turn, events flow one way, and the process is gone when the turn ends.

Billing guard (runtime detection layer): the `system:init` event reports
`apiKeySource`. If it says the binary authenticated via ANTHROPIC_API_KEY,
the turn is killed on the spot and surfaced as an error with
billing_lane="api_key" — that lane is metered, not the plan quota this
runtime exists to use. A terminal result mentioning the extra-usage bucket
is tagged billing_lane="extra_usage" the same way.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.redact import redact_sensitive_text

from .cli import (
    ClaudeCodeCLI,
    build_claude_child_env,
)
from .projector import ClaudeCodeEventProjector

logger = logging.getLogger(__name__)

_STDERR_TAIL_LINES = 12

# Substrings in claude stderr / result text that signal the OAuth token is
# no longer valid. Conservative, mirrors the codex session's classifier.
_OAUTH_FAILURE_HINTS = (
    "invalid_grant",
    "invalid grant",
    "refresh token",
    "token has expired",
    "expired token",
    "not authenticated",
    "unauthenticated",
    "unauthorized",
    "401",
    "invalid bearer token",
    "oauth token has expired",
    "please run /login",
    "run `claude login`",
    "authentication_error",
)

# Terminal result text that signals the Anthropic overage bucket — the
# billing lane this runtime exists to avoid. Matches error_classifier's
# _BILLING_PATTERNS entry for the same failure.
_EXTRA_USAGE_HINTS = (
    "out of extra usage",
    "extra usage",
)

# Result text that signals the --resume target is gone (transcript pruned,
# claude reinstalled, projects dir cleared). The session id must be dropped
# so the next turn starts a fresh Claude-side session instead of failing
# forever.
_RESUME_MISS_HINTS = (
    "no conversation found",
    "session not found",
    "could not resume",
)


@dataclass
class TurnResult:
    """Result of one user→assistant→tool turn through `claude -p`."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    claude_session_id: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    total_cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    # Which billing lane the turn actually hit: "plan" (OAuth, subscription
    # quota — the only acceptable value), "api_key" (metered key leaked in),
    # or "extra_usage" (plan overage bucket, the $-footgun this runtime
    # replaces). None when the turn died before authentication.
    billing_lane: Optional[str] = None
    # Hint to the caller that per-session state (resume id, MCP config)
    # should be dropped so the next turn starts clean.
    should_retire: bool = False


def _coerce_turn_input_text(user_input: Any) -> str:
    """Collapse Hermes/OpenAI rich content into a plain prompt string.

    Same behavior as the codex session's coercion: keep text fragments,
    replace opaque image payloads with a marker."""
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                if item is not None:
                    parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item_type in {"image", "image_url", "input_image"}:
                parts.append("[image attached]")
        text = "\n\n".join(p for p in parts if p).strip()
        return text or "What do you see in this image?"
    return "" if user_input is None else str(user_input)


def _classify_oauth_failure(*parts: str) -> Optional[str]:
    haystack = " ".join(p for p in parts if p).lower()
    if not haystack:
        return None
    for needle in _OAUTH_FAILURE_HINTS:
        if needle in haystack:
            return (
                "Claude Code authentication failed — the OAuth token looks "
                "expired or invalid. Re-inject a fresh CLAUDE_CODE_OAUTH_TOKEN "
                "(e.g. rotate it in Proton Pass and restart the gateway) or run "
                "`claude setup-token`, then retry."
            )
    return None


class ClaudeCodeSession:
    """One Claude Code resume chain per Hermes session, owned by AIAgent.

    Not thread-safe — one caller drives it at a time, same contract as
    CodexAppServerSession."""

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        claude_bin: str = "claude",
        mcp_config_path: Optional[str] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._cli = ClaudeCodeCLI(claude_bin=claude_bin, cwd=cwd)
        self._model = model
        self._mcp_config_path = mcp_config_path
        self._on_event = on_event  # Display hook (tool-progress breadcrumbs)
        self._claude_session_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        self._closed = False

    # ---------- interrupt ----------

    def request_interrupt(self) -> None:
        """Idempotent: signal the in-flight turn to kill the child and unwind."""
        self._interrupt_event.set()
        self._cli.abort()

    # ---------- lifecycle ----------

    def close(self) -> None:
        """No persistent subprocess to tear down — clear state only."""
        if self._closed:
            return
        self._closed = True
        self._cli.abort()
        self._claude_session_id = None

    def __enter__(self) -> "ClaudeCodeSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- diagnostics ----------

    def _format_error_with_stderr(self, prefix: str, exc: Any = "") -> str:
        exc_str = str(exc) if exc != "" and exc is not None else ""
        base = f"{prefix}: {exc_str}" if exc_str else prefix
        tail = self._cli.stderr_tail(_STDERR_TAIL_LINES)
        joined = "\n".join(line.rstrip() for line in tail if line).strip()
        if not joined:
            return base
        redacted = redact_sensitive_text(joined, force=True)
        return f"{base}\nclaude stderr (last {len(tail)} lines):\n{redacted}"

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: float = 600.0,
        interrupt_check: Optional[Callable[[], bool]] = None,
    ) -> TurnResult:
        result = TurnResult()

        # Env construction is the spawn-side billing hard guard — an auth or
        # leak failure surfaces as a turn error, never a spawn.
        try:
            env = build_claude_child_env()
        except Exception as exc:
            result.error = f"claude_code runtime cannot build child env: {exc}"
            return result

        prompt = _coerce_turn_input_text(user_input)
        resume_id = self._claude_session_id
        new_session_id = None if resume_id else str(uuid.uuid4())

        self._interrupt_event.clear()

        def _interrupted() -> bool:
            if self._interrupt_event.is_set():
                return True
            return bool(interrupt_check is not None and interrupt_check())

        projector = ClaudeCodeEventProjector()
        billing_killed = False
        try:
            for event in self._cli.run_print_turn(
                prompt,
                env=env,
                session_id=new_session_id,
                resume=resume_id,
                model=self._model,
                mcp_config_path=self._mcp_config_path,
                timeout=turn_timeout,
                interrupt_check=_interrupted,
            ):
                if self._on_event is not None:
                    try:
                        self._on_event(event)
                    except Exception:  # pragma: no cover - display callback
                        logger.debug("claude_code on_event raised", exc_info=True)

                projection = projector.project(event)
                if projection.messages:
                    result.projected_messages.extend(projection.messages)
                result.tool_iterations += projection.tool_iterations
                if projection.final_text is not None:
                    result.final_text = projection.final_text

                # Runtime billing-lane detection (guard layer B): if the
                # binary authenticated with an API key despite the sanitized
                # env, kill the turn immediately — every further token is
                # metered spend.
                if projector.api_key_source == "ANTHROPIC_API_KEY":
                    billing_killed = True
                    result.billing_lane = "api_key"
                    result.error = (
                        "claude_code runtime aborted: the claude binary "
                        "authenticated via ANTHROPIC_API_KEY (metered API "
                        "billing), not the plan OAuth token. Turn killed to "
                        "avoid API-key spend."
                    )
                    result.should_retire = True
                    self._cli.abort()
                    break
        except TimeoutError:
            result.interrupted = True
            result.should_retire = True
            result.error = self._format_error_with_stderr(
                f"claude_code turn timed out after {turn_timeout:.0f}s"
            )
        except FileNotFoundError as exc:
            result.error = (
                f"claude binary not found: {exc}. Install with: "
                f"npm i -g @anthropic-ai/claude-code"
            )
            result.should_retire = True
        except Exception as exc:
            logger.exception("claude_code turn failed")
            result.error = self._format_error_with_stderr(
                "claude_code turn failed", exc
            )
            result.should_retire = True

        if self._cli.interrupted:
            result.interrupted = True

        # ---- terminal bookkeeping from the projector ----
        result.claude_session_id = (
            projector.session_id or resume_id or new_session_id
        )
        result.usage = projector.usage
        result.total_cost_usd = projector.total_cost_usd
        result.num_turns = projector.num_turns

        if billing_killed or result.interrupted or result.error:
            return result

        if not projector.saw_result:
            # Child exited without a terminal result event — crash or kill.
            result.error = self._format_error_with_stderr(
                "claude exited without a result event "
                f"(exit code {self._cli.last_returncode})"
            )
            result.should_retire = True
            return result

        if projector.result_is_error:
            error_text = projector.result_text or projector.result_subtype or ""
            stderr_blob = "\n".join(self._cli.stderr_tail(40))
            lowered = f"{error_text}\n{stderr_blob}".lower()
            if any(h in lowered for h in _EXTRA_USAGE_HINTS):
                result.billing_lane = "extra_usage"
            hint = _classify_oauth_failure(error_text, stderr_blob)
            if hint is not None:
                result.error = hint
                result.should_retire = True
            else:
                result.error = self._format_error_with_stderr(
                    f"claude turn ended subtype={projector.result_subtype}",
                    redact_sensitive_text(error_text, force=True),
                )
            if resume_id and any(h in lowered for h in _RESUME_MISS_HINTS):
                # Resume target is gone — drop the chain so the next turn
                # starts a fresh Claude-side session.
                logger.warning(
                    "claude_code resume target %s missing; resetting chain",
                    resume_id[:8],
                )
                self._claude_session_id = None
            return result

        # ---- success ----
        if projector.result_text:
            result.final_text = projector.result_text
        # apiKeySource "none" = OAuth (verified live: plan-quota runs report
        # 'none' — there is no explicit "oauth" value on 2.1.x).
        result.billing_lane = result.billing_lane or "plan"
        self._claude_session_id = result.claude_session_id
        return result


__all__ = ["ClaudeCodeSession", "TurnResult"]
