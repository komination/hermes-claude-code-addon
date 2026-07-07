"""Opt-in gate for the claude_code runtime.

Pure function, no Hermes imports — the patcher wires it into
hermes_cli.runtime_provider so that `model.provider: anthropic` +
`model.claude_code_runtime: claude_code` in config.yaml rewrites the
resolved api_mode to "claude_code". Default behavior is preserved: when
the key is unset, "auto", or empty, the gate is a no-op, and only the
"anthropic" provider is eligible.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

API_MODE = "claude_code"
CONFIG_KEY = "claude_code_runtime"
ELIGIBLE_PROVIDER = "anthropic"


def maybe_apply_claude_code_runtime(
    *,
    provider: str,
    api_mode: str,
    model_cfg: Optional[Dict[str, Any]],
) -> str:
    """Return "claude_code" when the config opts this model in, else the
    api_mode unchanged.

    Turns then run through the real Claude Code binary (`claude -p` with
    the user's OAuth token), which Anthropic bills against the Pro/Max plan
    quota — unlike Hermes' own anthropic_messages client, which lands on
    the metered extra-usage lane.
    """
    if not model_cfg:
        return api_mode
    if provider != ELIGIBLE_PROVIDER:
        return api_mode
    runtime = str(model_cfg.get(CONFIG_KEY) or "").strip().lower()
    if runtime == API_MODE:
        return API_MODE
    return api_mode


__all__ = ["maybe_apply_claude_code_runtime", "API_MODE", "CONFIG_KEY"]
