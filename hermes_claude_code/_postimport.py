"""Minimal post-import hook machinery (wrapt-style, dependency-free).

`register(module_name, callback)` arranges for `callback(module)` to run
right after `module_name` finishes executing — or immediately if it is
already imported. Used by the patcher to apply seam patches lazily, so the
`.pth` activation hook adds zero import cost to python processes that never
touch Hermes (e.g. the hermes-tools MCP grandchild).

Implementation: a sys.meta_path finder that intercepts watched names,
resolves the real spec via importlib.util.find_spec (guarded against
recursion), and wraps the spec's loader so exec_module fires the callbacks
afterwards. Callbacks must never raise into the import system — they are
wrapped in try/except and failures are logged.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
from typing import Callable, Dict, List

logger = logging.getLogger("hermes_claude_code")

_lock = threading.Lock()
_hooks: Dict[str, List[Callable]] = {}
_finder_installed = False


def _fire(module_name: str, module) -> None:
    with _lock:
        callbacks = list(_hooks.get(module_name, ()))
    for callback in callbacks:
        try:
            callback(module)
        except Exception:
            logger.exception(
                "hermes_claude_code post-import hook for %s failed", module_name
            )


class _LoaderProxy:
    """Delegates to the real loader, firing hooks after exec_module."""

    def __init__(self, loader, fullname: str) -> None:
        self._loader = loader
        self._fullname = fullname

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        _fire(self._fullname, module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _PostImportFinder:
    """meta_path finder that only reacts to watched module names."""

    def __init__(self) -> None:
        self._in_progress: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        with _lock:
            watched = fullname in _hooks
        if not watched or fullname in self._in_progress:
            return None
        self._in_progress.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            return None
        finally:
            self._in_progress.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _LoaderProxy(spec.loader, fullname)
        return spec


def register(module_name: str, callback: Callable) -> None:
    """Run callback(module) after module_name is imported (or now, if it
    already is). Thread-safe; multiple callbacks per module allowed."""
    global _finder_installed
    existing = sys.modules.get(module_name)
    if existing is not None:
        callback(existing)
        return
    with _lock:
        _hooks.setdefault(module_name, []).append(callback)
        if not _finder_installed:
            sys.meta_path.insert(0, _PostImportFinder())
            _finder_installed = True


__all__ = ["register"]
