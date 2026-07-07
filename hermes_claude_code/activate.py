"""Startup activation — imported by the .pth hook on every venv python.

Must never raise: an exception here would print a traceback at interpreter
startup for every process in the Hermes venv. Registers lazy post-import
hooks only (near-zero cost for processes that never import Hermes).

Kill switch: set HERMES_CLAUDE_CODE_ADDON=0 to disable the addon entirely
without uninstalling it.
"""

from __future__ import annotations

import os


def _activate() -> None:
    if os.environ.get("HERMES_CLAUDE_CODE_ADDON", "").strip() == "0":
        return
    from . import patcher

    patcher.install()


try:
    _activate()
except Exception:  # pragma: no cover - never break interpreter startup
    try:
        import logging

        logging.getLogger("hermes_claude_code").exception(
            "hermes_claude_code activation failed"
        )
    except Exception:
        pass
