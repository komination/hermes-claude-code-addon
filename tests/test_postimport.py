"""Tests for the post-import hook machinery."""

import sys
import textwrap

from hermes_claude_code import _postimport


def test_already_imported_module_fires_immediately():
    fired = []
    _postimport.register("json", fired.append)
    assert len(fired) == 1
    assert fired[0] is sys.modules["json"]


def test_hook_fires_after_deferred_import(tmp_path, monkeypatch):
    module_name = "hcc_postimport_fixture"
    (tmp_path / f"{module_name}.py").write_text("VALUE = 41\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    fired = []
    _postimport.register(module_name, lambda mod: fired.append(mod.VALUE))
    assert fired == []  # not imported yet

    imported = __import__(module_name)
    assert fired == [41]
    assert imported.VALUE == 41
    sys.modules.pop(module_name, None)


def test_callback_exception_does_not_break_import(tmp_path, monkeypatch):
    module_name = "hcc_postimport_raiser"
    (tmp_path / f"{module_name}.py").write_text("VALUE = 7\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    def bad_hook(_mod):
        raise RuntimeError("boom")

    _postimport.register(module_name, bad_hook)
    imported = __import__(module_name)  # must not raise
    assert imported.VALUE == 7
    sys.modules.pop(module_name, None)


def test_module_with_hook_still_imports_normally(tmp_path, monkeypatch):
    """The loader proxy must preserve normal module semantics (spec, name,
    submodule imports from the module body)."""
    module_name = "hcc_postimport_semantics"
    (tmp_path / f"{module_name}.py").write_text(
        textwrap.dedent(
            """
            import json as _json
            NAME = __name__
            def roundtrip(x):
                return _json.loads(_json.dumps(x))
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    _postimport.register(module_name, lambda mod: None)
    imported = __import__(module_name)
    assert imported.NAME == module_name
    assert imported.roundtrip({"a": 1}) == {"a": 1}
    sys.modules.pop(module_name, None)
