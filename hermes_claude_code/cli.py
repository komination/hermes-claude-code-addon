"""Claude Code CLI print-mode client.

Drives one-shot `claude -p "<prompt>" --output-format stream-json --verbose`
turns for the `claude_code` api_mode (see hermes_claude_code/runtime.py).
The real Claude Code binary owns the agentic loop (model calls, built-in
tools, transcript persistence); Hermes reads the one-way NDJSON event
stream from stdout and projects it back into its own messages list.

Unlike the codex app-server sibling (codex_app_server.py) there is no
JSON-RPC correlation: `-p` is fire-and-forget per turn, input goes in as a
positional argument, and multi-turn context is reconstructed by Claude Code
itself via `--session-id <uuid>` (first turn) / `--resume <uuid>`
(subsequent turns).

Billing guard (the "$1,800 footgun"): `build_claude_child_env` is the ONLY
sanctioned way to build the child environment. It starts from
`hermes_subprocess_env(inherit_credentials=False)` so no provider API keys
leak in, explicitly pops every ANTHROPIC_* credential/routing variable, and
re-injects exactly one credential — the Claude Code OAuth token — so the
spawned binary bills the user's Pro/Max plan quota, never the metered API
key lane. It refuses to return an environment that still carries
ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Any, Callable, Iterator, Optional

from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

# Minimum Claude Code version we test against. `--session-id`/`--resume`
# print-mode behavior verified live on 2.1.201.
MIN_CLAUDE_VERSION = (2, 1, 0)

# Anthropic credential / routing env vars that must never reach the child:
# an inherited ANTHROPIC_API_KEY silently flips the binary onto metered
# API-key billing, and a stale ANTHROPIC_BASE_URL would reroute the call.
_ANTHROPIC_CRED_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_BEDROCK_BASE_URL",
)

# Nested-Claude markers: Hermes itself may be running inside a Claude Code
# session (dev workflows), whose env carries CLAUDECODE, CLAUDE_CODE_ENTRYPOINT,
# CLAUDE_CODE_SESSION_ID, CLAUDE_CODE_CHILD_SESSION, CLAUDE_EFFORT, etc. The
# child must start as a clean top-level session: an inherited
# CLAUDE_CODE_SESSION_ID would fight the explicit --session-id/--resume argv,
# and other markers make the binary think it is a nested tool call. Everything
# with these prefixes is popped EXCEPT the OAuth token (the one credential we
# re-inject). In production (systemd gateway) none of these are present anyway.
_NESTED_CLAUDE_KEYS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
_NESTED_CLAUDE_PREFIXES = ("CLAUDE_CODE_", "CLAUDECODE", "CLAUDE_EFFORT")
_OAUTH_TOKEN_KEY = "CLAUDE_CODE_OAUTH_TOKEN"

_STDERR_MAX_LINES = 500


class ClaudeCodeAuthError(RuntimeError):
    """No usable Claude Code OAuth token could be resolved."""


class ClaudeCodeBillingLeakError(RuntimeError):
    """The constructed child env would bill the metered API-key lane."""


def _oauth_token_from_credentials_file() -> Optional[str]:
    """Fallback token source: refreshable Claude Code credentials on disk.

    Uses the pure refresh path when the stored access token is expired so
    nothing is written back to disk (the user's Proton-Pass workflow keeps
    ~/.claude/.credentials.json absent; this fallback only matters for
    installs that do persist credentials)."""
    try:
        from agent.anthropic_adapter import (
            is_claude_code_token_valid,
            read_claude_code_credentials,
            refresh_anthropic_oauth_pure,
        )

        creds = read_claude_code_credentials()
        if not creds:
            return None
        if is_claude_code_token_valid(creds):
            return creds.get("accessToken") or None
        refresh_token = creds.get("refreshToken") or ""
        if not refresh_token:
            return None
        refreshed = refresh_anthropic_oauth_pure(refresh_token)
        return refreshed.get("access_token") or None
    except Exception:
        logger.debug("claude_code credentials-file fallback failed", exc_info=True)
        return None


def build_claude_child_env(*, oauth_token: Optional[str] = None) -> dict[str, str]:
    """Build the sanitized environment for a spawned `claude` binary.

    Token resolution order: explicit argument → CLAUDE_CODE_OAUTH_TOKEN in
    the parent env (gateway/dashboard wrappers inject it from Proton Pass)
    → refreshable ~/.claude credentials (pure refresh, no disk writes).

    Raises ClaudeCodeAuthError when no token is available and
    ClaudeCodeBillingLeakError if the result would still carry
    ANTHROPIC_API_KEY (defense in depth — the pops above make this
    unreachable unless the strip logic regresses).
    """
    # NOTE: intentionally inherit_credentials=False, unlike the codex
    # app-server spawn. The claude binary must authenticate with the OAuth
    # token ONLY — any inherited provider key is a billing-lane leak, not a
    # convenience.
    env = hermes_subprocess_env(inherit_credentials=False)
    for key in _ANTHROPIC_CRED_KEYS:
        env.pop(key, None)
    # Strip every nested-Claude marker by prefix, preserving only the OAuth
    # token we are about to (re)set.
    for key in list(env):
        if key == _OAUTH_TOKEN_KEY:
            continue
        if key.startswith(_NESTED_CLAUDE_PREFIXES):
            env.pop(key, None)

    token = (
        oauth_token
        or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
        or _oauth_token_from_credentials_file()
    )
    if not token:
        raise ClaudeCodeAuthError(
            "No Claude Code OAuth token available. Inject CLAUDE_CODE_OAUTH_TOKEN "
            "into the Hermes process env (e.g. via the gateway wrapper) or run "
            "`claude setup-token`."
        )
    env["CLAUDE_CODE_OAUTH_TOKEN"] = token

    if "ANTHROPIC_API_KEY" in env:
        raise ClaudeCodeBillingLeakError(
            "Refusing to spawn claude: ANTHROPIC_API_KEY survived env "
            "sanitization and would bill the metered API lane."
        )
    return env


def parse_claude_version(output: str) -> Optional[tuple[int, int, int]]:
    """Parse `claude --version` output ("2.1.201 (Claude Code)")."""
    import re

    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output or "")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def check_claude_binary(
    claude_bin: str = "claude",
    min_version: tuple[int, int, int] = MIN_CLAUDE_VERSION,
) -> tuple[bool, str]:
    """Verify the Claude Code CLI is installed and recent enough.

    Returns (ok, version_or_message)."""
    try:
        proc = subprocess.run(
            [claude_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, (
            f"claude CLI not found at {claude_bin!r}. Install with: "
            f"npm i -g @anthropic-ai/claude-code"
        )
    except subprocess.TimeoutExpired:
        return False, "claude --version timed out"
    if proc.returncode != 0:
        return False, f"claude --version exited {proc.returncode}: {proc.stderr.strip()}"
    version = parse_claude_version(proc.stdout)
    if version is None:
        return False, f"could not parse claude version from: {proc.stdout!r}"
    if version < min_version:
        return False, (
            f"claude {'.'.join(map(str, version))} is older than required "
            f"{'.'.join(map(str, min_version))}. Run: npm i -g @anthropic-ai/claude-code"
        )
    return True, ".".join(map(str, version))


class ClaudeCodeCLI:
    """One-way stream-json reader around `claude -p`.

    One instance per ClaudeCodeSession; `run_print_turn` spawns a fresh
    subprocess per turn (print mode is one-shot) and yields parsed NDJSON
    events. Diagnostics (stderr tail, exit code, interrupt flag) for the
    most recent turn are kept on the instance.
    """

    def __init__(self, *, claude_bin: str = "claude", cwd: Optional[str] = None) -> None:
        self._claude_bin = claude_bin
        self._cwd = cwd or os.getcwd()
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self.last_returncode: Optional[int] = None
        self.timed_out = False
        self.interrupted = False

    # ---------- diagnostics ----------

    def stderr_tail(self, n: int = 12) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    # ---------- per-turn ----------

    def build_argv(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        resume: Optional[str] = None,
        model: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
        skip_permissions: bool = True,
        extra_args: Optional[list[str]] = None,
    ) -> list[str]:
        argv = [
            self._claude_bin,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if model:
            argv += ["--model", model]
        if resume:
            argv += ["--resume", resume]
        elif session_id:
            argv += ["--session-id", session_id]
        if mcp_config_path:
            argv += ["--mcp-config", mcp_config_path, "--strict-mcp-config"]
        if skip_permissions:
            # Full-autonomy runtime: the user opted into unattended built-in
            # tool execution when enabling model.claude_code_runtime.
            argv.append("--dangerously-skip-permissions")
        if extra_args:
            argv += list(extra_args)
        return argv

    def run_print_turn(
        self,
        prompt: str,
        *,
        env: dict[str, str],
        session_id: Optional[str] = None,
        resume: Optional[str] = None,
        model: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
        skip_permissions: bool = True,
        timeout: float = 600.0,
        poll_timeout: float = 0.25,
        interrupt_check: Optional[Callable[[], bool]] = None,
        extra_args: Optional[list[str]] = None,
    ) -> Iterator[dict]:
        """Spawn one print-mode turn and yield parsed stream-json events.

        Raises TimeoutError when the turn deadline passes (child is killed
        first). When `interrupt_check` returns True the child is killed,
        `self.interrupted` is set, and iteration ends without raising —
        mirrors how the codex session treats user interrupts as a clean,
        reportable outcome rather than an exception.
        """
        self.last_returncode = None
        self.timed_out = False
        self.interrupted = False
        with self._stderr_lock:
            self._stderr_lines = []

        argv = self.build_argv(
            prompt,
            session_id=session_id,
            resume=resume,
            model=model,
            mcp_config_path=mcp_config_path,
            skip_permissions=skip_permissions,
            extra_args=extra_args,
        )

        # Belt-and-braces: never spawn with an API key in the child env,
        # whatever the caller handed us.
        if "ANTHROPIC_API_KEY" in env:
            raise ClaudeCodeBillingLeakError(
                "Refusing to spawn claude: ANTHROPIC_API_KEY present in child env."
            )

        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._cwd,
            env=env,
        )
        self._proc = proc

        lines: queue.Queue = queue.Queue()

        def _read_stdout() -> None:
            try:
                assert proc.stdout is not None
                for raw in iter(proc.stdout.readline, b""):
                    lines.put(raw)
            except Exception as exc:  # pragma: no cover - defensive
                with self._stderr_lock:
                    self._stderr_lines.append(f"<stdout reader error> {exc}")
            finally:
                lines.put(None)  # EOF sentinel

        def _read_stderr() -> None:
            try:
                assert proc.stderr is not None
                for raw in iter(proc.stderr.readline, b""):
                    with self._stderr_lock:
                        self._stderr_lines.append(
                            raw.decode("utf-8", "replace").rstrip()
                        )
                        if len(self._stderr_lines) > _STDERR_MAX_LINES:
                            self._stderr_lines = self._stderr_lines[-_STDERR_MAX_LINES:]
            except Exception:  # pragma: no cover
                pass

        threading.Thread(target=_read_stdout, daemon=True).start()
        threading.Thread(target=_read_stderr, daemon=True).start()

        deadline = time.monotonic() + timeout
        try:
            while True:
                if interrupt_check is not None and interrupt_check():
                    self.interrupted = True
                    self._kill(proc)
                    return
                if time.monotonic() > deadline:
                    self.timed_out = True
                    self._kill(proc)
                    raise TimeoutError(
                        f"claude -p turn exceeded {timeout:.0f}s deadline"
                    )
                try:
                    raw = lines.get(timeout=poll_timeout)
                except queue.Empty:
                    continue
                if raw is None:  # EOF
                    break
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    with self._stderr_lock:
                        self._stderr_lines.append(
                            f"<non-json on stdout> {raw[:200]!r}"
                        )
                    continue
                if isinstance(event, dict):
                    yield event
        finally:
            self.last_returncode = proc.poll()
            if self.last_returncode is None:
                self._kill(proc)
                self.last_returncode = proc.poll()
            self._proc = None

    def abort(self) -> None:
        """Kill the in-flight child (billing guard / interrupt path)."""
        proc = self._proc
        if proc is not None:
            self._kill(proc)

    @staticmethod
    def _kill(proc: subprocess.Popen, grace: float = 3.0) -> None:
        try:
            proc.terminate()
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:  # pragma: no cover
                pass
        except Exception:  # pragma: no cover
            pass


__all__ = [
    "ClaudeCodeCLI",
    "ClaudeCodeAuthError",
    "ClaudeCodeBillingLeakError",
    "MIN_CLAUDE_VERSION",
    "build_claude_child_env",
    "check_claude_binary",
    "parse_claude_version",
]
