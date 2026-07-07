"""Make the addon and the Hermes source tree importable during tests.

The tests import both `hermes_claude_code` (this package) and the real
Hermes modules it patches (`agent.*`, `tools.*`, `hermes_cli.*`,
`run_agent`), plus Hermes' own third-party deps. The Hermes tree is a
private editable install that cannot be a pip dependency, so this conftest
prepends it — and its venv site-packages — to sys.path at collection time
(the idiomatic pytest place for test-path setup, replacing PYTHONPATH
plumbing). Override the checkout location with the HERMES_REPO env var.
"""

from __future__ import annotations

import glob
import os
import sys

_ADDON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERMES_REPO = os.environ.get("HERMES_REPO") or os.path.expanduser(
    "~/.hermes/hermes-agent"
)

# The addon itself (so `import hermes_claude_code` works without an editable
# install), the Hermes checkout (agent/tools/hermes_cli/run_agent), and the
# Hermes venv's site-packages (anthropic, mcp, … that Hermes modules import).
_paths = [_ADDON_ROOT, _HERMES_REPO]
_paths += glob.glob(os.path.join(_HERMES_REPO, "venv", "lib", "python3.*", "site-packages"))

for _path in _paths:
    if _path and os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)
