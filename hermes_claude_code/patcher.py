"""Runtime seam patches that teach an unmodified Hermes the claude_code mode.

Four seams, mirroring the (never-committed) in-tree diff of attempt 1 but
applied by monkeypatch at import time instead of editing the checkout:

  1. hermes_cli.runtime_provider
       - `_VALID_API_MODES.add("claude_code")` so an explicit
         `model.api_mode: claude_code` survives `_parse_api_mode`.
       - Wrap `_maybe_apply_codex_app_server_runtime` — it is invoked as a
         module-global inside `_resolve_runtime_from_pool_entry` for every
         pool-entry resolution (verified: no early-bound from-import exists
         anywhere), which makes it the exact insertion point the in-tree
         patch used. The wrapper delegates to the original, then applies the
         claude_code gate (`model.provider: anthropic` +
         `model.claude_code_runtime: claude_code`).
  2. agent.agent_init
       - Wrap `init_agent`: the stock acceptance set rejects "claude_code"
         and falls back to anthropic_messages; the wrapper restores the
         requested mode afterwards.
  3. agent.conversation_loop
       - Replace `run_conversation` with a signature-identical wrapper.
         Non-claude_code agents pass straight through to the original. For
         api_mode == "claude_code" the wrapper reproduces the upstream
         prologue (moa decode → build_turn_context → refresh-counter reset;
         lines 550-617 at pinned commit 22c5048d9) using conversation_loop's
         own module namespace, then forks into runtime.run_claude_code_turn
         — byte-equivalent behavior to the in-tree fork at line 624.
         Every production call site funnels through the function-local
         import in AIAgent.run_conversation (run_agent.py:5723), so the
         module-attribute swap catches all of them.
  4. run_agent
       - Wrap `AIAgent.close` to drop the ClaudeCodeSession (kills any
         in-flight `claude -p` child, clears the resume chain) before the
         stock teardown.

Fail-safe wiring: each seam verifies the upstream symbols it touches and
records "patched" / "failed: <reason>". The runtime_provider gate — the
only entry point that can switch an agent onto claude_code — refuses to do
so unless ALL downstream seams report "patched", so a partial patch (e.g.
after an upstream update drifts a seam) degrades to stock Hermes behavior
with a loud log line instead of a half-wired runtime.
"""

from __future__ import annotations

import functools
import inspect
import logging
import sys
import threading
from typing import Any, Dict, List, Optional

from . import gate
from ._postimport import register

logger = logging.getLogger("hermes_claude_code")

SEAM_RUNTIME_PROVIDER = "hermes_cli.runtime_provider"
SEAM_AGENT_INIT = "agent.agent_init"
SEAM_CONVERSATION_LOOP = "agent.conversation_loop"
SEAM_RUN_AGENT = "run_agent"

ALL_SEAMS = (
    SEAM_RUNTIME_PROVIDER,
    SEAM_AGENT_INIT,
    SEAM_CONVERSATION_LOOP,
    SEAM_RUN_AGENT,
)

# Seams that must be live before the gate may switch an agent to claude_code.
_DOWNSTREAM_SEAMS = (SEAM_AGENT_INIT, SEAM_CONVERSATION_LOOP, SEAM_RUN_AGENT)

_MARKER = "__hermes_claude_code_addon__"

# conversation_loop module-level names the prologue replica reads at call
# time. Missing any of them means the upstream prologue changed shape.
_PROLOGUE_HELPERS = (
    "build_turn_context",
    "_restore_or_build_system_prompt",
    "_install_safe_stdio",
    "_sanitize_surrogates",
    "_summarize_user_message_for_log",
    "set_session_context",
    "set_current_write_origin",
    "_ra",
)

_EXPECTED_RUN_CONVERSATION_PARAMS = [
    "agent",
    "user_message",
    "system_message",
    "conversation_history",
    "task_id",
    "stream_callback",
    "persist_user_message",
    "persist_user_timestamp",
    "moa_config",
]

_TURN_CONTEXT_REQUIRED_FIELDS = {
    "user_message",
    "original_user_message",
    "messages",
    "effective_task_id",
    "should_review_memory",
}

_lock = threading.Lock()
_installed = False
_status: Dict[str, str] = {seam: "pending" for seam in ALL_SEAMS}
_downstream_warned = False


class SeamError(RuntimeError):
    """A seam's upstream shape does not match what the patch expects."""


def seam_status() -> Dict[str, str]:
    """Copy of the per-seam status map ("pending"/"patched"/"failed: ...")."""
    with _lock:
        return dict(_status)


def _set_status(seam: str, value: str) -> None:
    with _lock:
        _status[seam] = value
    if value.startswith("failed"):
        logger.warning("hermes_claude_code seam %s: %s", seam, value)
    else:
        logger.info("hermes_claude_code seam %s: %s", seam, value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SeamError(message)


def _source_of(func) -> str:
    try:
        return inspect.getsource(func)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Seam 1: hermes_cli.runtime_provider
# ---------------------------------------------------------------------------

def downstream_ready() -> bool:
    """True when every seam needed to run a claude_code turn is patched.

    Imports the downstream modules on first use (triggering their patches);
    afterwards it is a cheap dict read. Called by the gate wrapper right
    before it would rewrite api_mode, making the whole addon all-or-nothing.
    """
    import importlib

    for module_name in _DOWNSTREAM_SEAMS:
        if module_name not in sys.modules:
            try:
                importlib.import_module(module_name)
            except Exception:
                _set_status(module_name, "failed: import error")
                return False
    with _lock:
        return all(_status[s] == "patched" for s in _DOWNSTREAM_SEAMS)


def _make_codex_gate_wrapper(original):
    @functools.wraps(original)
    def _maybe_apply_codex_app_server_runtime(
        *, provider: str, api_mode: str, model_cfg: Optional[Dict[str, Any]]
    ) -> str:
        resolved = original(
            provider=provider, api_mode=api_mode, model_cfg=model_cfg
        )
        claude_mode = gate.maybe_apply_claude_code_runtime(
            provider=provider, api_mode=resolved, model_cfg=model_cfg
        )
        if claude_mode == resolved:
            return resolved
        if not downstream_ready():
            global _downstream_warned
            if not _downstream_warned:
                _downstream_warned = True
                logger.warning(
                    "hermes_claude_code: config requests the claude_code "
                    "runtime but not all seams are patched (%s) — keeping "
                    "api_mode=%r (stock Hermes behavior).",
                    seam_status(),
                    resolved,
                )
            return resolved
        logger.info(
            "hermes_claude_code: provider=%s routed to the claude_code "
            "runtime (plan-quota billing via the real claude binary)",
            provider,
        )
        return claude_mode

    setattr(_maybe_apply_codex_app_server_runtime, _MARKER, True)
    _maybe_apply_codex_app_server_runtime.__hermes_claude_code_original__ = original
    return _maybe_apply_codex_app_server_runtime


def _patch_runtime_provider(rp) -> None:
    original = getattr(rp, "_maybe_apply_codex_app_server_runtime", None)
    if getattr(original, _MARKER, False):
        _set_status(SEAM_RUNTIME_PROVIDER, "patched")
        return
    valid_modes = getattr(rp, "_VALID_API_MODES", None)
    _require(isinstance(valid_modes, set), "_VALID_API_MODES is not a set")
    _require(callable(original), "_maybe_apply_codex_app_server_runtime missing")
    params = inspect.signature(original).parameters
    _require(
        set(params) == {"provider", "api_mode", "model_cfg"},
        f"codex gate signature drifted: {sorted(params)}",
    )
    _require(
        callable(getattr(rp, "_parse_api_mode", None)),
        "_parse_api_mode missing",
    )
    valid_modes.add(gate.API_MODE)
    rp._maybe_apply_codex_app_server_runtime = _make_codex_gate_wrapper(original)
    _set_status(SEAM_RUNTIME_PROVIDER, "patched")


# ---------------------------------------------------------------------------
# Seam 2: agent.agent_init
# ---------------------------------------------------------------------------

def _make_init_agent_wrapper(original, signature):
    @functools.wraps(original)
    def init_agent(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            bound = signature.bind_partial(*args, **kwargs)
            requested = bound.arguments.get("api_mode")
            agent_obj = bound.arguments.get("agent")
            if agent_obj is None and args:
                agent_obj = args[0]
            if requested == gate.API_MODE and agent_obj is not None:
                # Stock init_agent rejected the unknown mode and fell back
                # (anthropic provider → "anthropic_messages"); restore it.
                agent_obj.api_mode = gate.API_MODE
        except Exception:
            logger.exception("hermes_claude_code init_agent post-fix failed")
        return result

    setattr(init_agent, _MARKER, True)
    init_agent.__hermes_claude_code_original__ = original
    return init_agent


def _patch_agent_init(ai) -> None:
    original = getattr(ai, "init_agent", None)
    if getattr(original, _MARKER, False):
        _set_status(SEAM_AGENT_INIT, "patched")
        return
    _require(callable(original), "init_agent missing")
    signature = inspect.signature(original)
    params = list(signature.parameters)
    _require(params and params[0] == "agent", f"first param drifted: {params[:1]}")
    _require("api_mode" in signature.parameters, "api_mode param missing")
    _require(
        "codex_app_server" in _source_of(original),
        "init_agent acceptance set not found (source drifted)",
    )
    ai.init_agent = _make_init_agent_wrapper(original, signature)
    _set_status(SEAM_AGENT_INIT, "patched")


# ---------------------------------------------------------------------------
# Seam 3: agent.conversation_loop
# ---------------------------------------------------------------------------

def _run_claude_code_fork(
    cl,
    agent,
    user_message,
    system_message,
    conversation_history,
    task_id,
    stream_callback,
    persist_user_message,
    persist_user_timestamp,
    moa_config,
) -> Dict[str, Any]:
    """Replica of run_conversation's prologue (upstream lines 550-617 at
    commit 22c5048d9) followed by the claude_code fork.

    All helpers are read from conversation_loop's module namespace at call
    time, exactly as the original does, so upstream monkeypatch seams
    (e.g. `_ra`) keep working.
    """
    if moa_config is None:
        try:
            from hermes_cli.moa_config import decode_moa_turn

            _decoded_message, _decoded_moa_config = decode_moa_turn(user_message)
            if _decoded_moa_config is not None:
                user_message = _decoded_message
                moa_config = _decoded_moa_config
                if persist_user_message is None:
                    persist_user_message = _decoded_message
        except Exception:
            pass

    ctx = cl.build_turn_context(
        agent,
        user_message,
        system_message,
        conversation_history,
        task_id,
        stream_callback,
        persist_user_message,
        persist_user_timestamp,
        restore_or_build_system_prompt=cl._restore_or_build_system_prompt,
        install_safe_stdio=cl._install_safe_stdio,
        sanitize_surrogates=cl._sanitize_surrogates,
        summarize_user_message_for_log=cl._summarize_user_message_for_log,
        set_session_context=cl.set_session_context,
        set_current_write_origin=cl.set_current_write_origin,
        ra=cl._ra,
    )

    # Same per-turn reset the upstream prologue performs (see #26080 there).
    agent._auth_pool_refresh_counts = {}

    from .runtime import run_claude_code_turn

    return run_claude_code_turn(
        agent,
        user_message=ctx.user_message,
        original_user_message=ctx.original_user_message,
        messages=ctx.messages,
        effective_task_id=ctx.effective_task_id,
        should_review_memory=ctx.should_review_memory,
    )


def _make_run_conversation_wrapper(cl, original):
    @functools.wraps(original)
    def run_conversation(
        agent,
        user_message,
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        persist_user_timestamp=None,
        moa_config=None,
    ) -> Dict[str, Any]:
        if getattr(agent, "api_mode", None) != gate.API_MODE:
            return original(
                agent,
                user_message,
                system_message,
                conversation_history,
                task_id,
                stream_callback,
                persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
                moa_config=moa_config,
            )
        return _run_claude_code_fork(
            cl,
            agent,
            user_message,
            system_message,
            conversation_history,
            task_id,
            stream_callback,
            persist_user_message,
            persist_user_timestamp,
            moa_config,
        )

    setattr(run_conversation, _MARKER, True)
    run_conversation.__hermes_claude_code_original__ = original
    return run_conversation


def _patch_conversation_loop(cl) -> None:
    original = getattr(cl, "run_conversation", None)
    if getattr(original, _MARKER, False):
        _set_status(SEAM_CONVERSATION_LOOP, "patched")
        return
    _require(callable(original), "run_conversation missing")
    params = list(inspect.signature(original).parameters)
    _require(
        params == _EXPECTED_RUN_CONVERSATION_PARAMS,
        f"run_conversation signature drifted: {params}",
    )
    for name in _PROLOGUE_HELPERS:
        _require(
            getattr(cl, name, None) is not None,
            f"conversation_loop.{name} missing",
        )
    source = _source_of(original)
    for anchor in (
        "build_turn_context(",
        "_auth_pool_refresh_counts",
        'api_mode == "codex_app_server"',
    ):
        _require(anchor in source, f"prologue anchor {anchor!r} not found")

    import dataclasses

    from agent.turn_context import TurnContext

    fields = {f.name for f in dataclasses.fields(TurnContext)}
    missing = _TURN_CONTEXT_REQUIRED_FIELDS - fields
    _require(not missing, f"TurnContext fields missing: {sorted(missing)}")

    cl.run_conversation = _make_run_conversation_wrapper(cl, original)
    _set_status(SEAM_CONVERSATION_LOOP, "patched")


# ---------------------------------------------------------------------------
# Seam 4: run_agent (AIAgent.close)
# ---------------------------------------------------------------------------

def _make_close_wrapper(original):
    @functools.wraps(original)
    def close(self) -> None:
        # 0. Drop the claude_code runtime session first (kills any in-flight
        # `claude -p` child and clears the resume chain).
        try:
            session = getattr(self, "_claude_session", None)
            if session is not None:
                session.close()
                self._claude_session = None
        except Exception:
            pass
        return original(self)

    setattr(close, _MARKER, True)
    close.__hermes_claude_code_original__ = original
    return close


def _patch_run_agent(ra) -> None:
    agent_cls = getattr(ra, "AIAgent", None)
    _require(inspect.isclass(agent_cls), "AIAgent class missing")
    original = agent_cls.__dict__.get("close")
    if getattr(original, _MARKER, False):
        _set_status(SEAM_RUN_AGENT, "patched")
        return
    _require(callable(original), "AIAgent.close missing")
    agent_cls.close = _make_close_wrapper(original)
    _set_status(SEAM_RUN_AGENT, "patched")


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

_SEAM_PATCHERS = {
    SEAM_RUNTIME_PROVIDER: _patch_runtime_provider,
    SEAM_AGENT_INIT: _patch_agent_init,
    SEAM_CONVERSATION_LOOP: _patch_conversation_loop,
    SEAM_RUN_AGENT: _patch_run_agent,
}


def _hook_for(seam: str):
    def hook(module) -> None:
        try:
            _SEAM_PATCHERS[seam](module)
        except SeamError as exc:
            _set_status(seam, f"failed: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            _set_status(seam, f"failed: unexpected {exc!r}")

    return hook


def install() -> None:
    """Register post-import hooks for every seam (idempotent).

    Modules already imported are patched immediately; the rest are patched
    the moment Hermes imports them. Costs nothing for python processes that
    never import Hermes modules.
    """
    global _installed
    with _lock:
        if _installed:
            return
        _installed = True
    for seam in ALL_SEAMS:
        register(seam, _hook_for(seam))


__all__ = [
    "ALL_SEAMS",
    "SeamError",
    "downstream_ready",
    "install",
    "seam_status",
]
