"""hermes-claude-code-addon — plan-quota Claude Code runtime for Hermes.

Out-of-tree addon that teaches an *unmodified* Hermes checkout a new
``claude_code`` api_mode: whole turns are delegated to a spawned
``claude -p`` subprocess (the real Claude Code binary, authenticated with
the user's OAuth token) so Anthropic bills the Pro/Max plan quota instead
of the metered extra-usage lane.

Integration is 100% runtime patching — no file in the Hermes checkout is
ever modified. See ``patcher.py`` for the four seams and ``activate.py``
for the ``.pth`` startup hook. Kill switch: ``HERMES_CLAUDE_CODE_ADDON=0``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
