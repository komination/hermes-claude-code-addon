"""Ops CLI: `hermes-claude-code-addon <subcommand>`.

Run with the Hermes venv python (the console script installed there does
this automatically):

  install-pth    Write the startup hook (.pth) into this interpreter's
                 site-packages so every venv python activates the addon.
  uninstall-pth  Remove the startup hook.
  status         Show seam patch status, activation state, config gate,
                 and claude binary health. --import forces the Hermes
                 modules to load so the real patch outcome is shown.
  smoke          Billing smoke test: spawn the real `claude -p` with the
                 sanitized child env and report apiKeySource / is_error /
                 cost. Never prints credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig

PTH_NAME = "zzz_hermes_claude_code_addon.pth"
PTH_LINE = "import hermes_claude_code.activate\n"


def _site_packages() -> str:
    return sysconfig.get_paths()["purelib"]


def _pth_path() -> str:
    return os.path.join(_site_packages(), PTH_NAME)


def cmd_install_pth(_args) -> int:
    path = _pth_path()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(PTH_LINE)
    print(f"wrote {path}")
    print("every python in this environment now activates the addon at startup")
    return 0


def cmd_uninstall_pth(_args) -> int:
    path = _pth_path()
    if os.path.exists(path):
        os.remove(path)
        print(f"removed {path}")
    else:
        print(f"not installed ({path} absent)")
    return 0


def cmd_status(args) -> int:
    from . import __version__, patcher

    print(f"hermes-claude-code-addon {__version__}")
    print(f"python: {sys.executable}")
    pth = _pth_path()
    print(f"startup hook: {pth} -> {'present' if os.path.exists(pth) else 'ABSENT'}")
    kill = os.environ.get("HERMES_CLAUDE_CODE_ADDON", "")
    print(f"kill switch HERMES_CLAUDE_CODE_ADDON: {kill or '(unset)'}")

    if args.force_import:
        patcher.install()
        ready = patcher.downstream_ready()  # imports downstream seams
        import importlib

        importlib.import_module("hermes_cli.runtime_provider")
        print(f"downstream_ready: {ready}")
    for seam, state in patcher.seam_status().items():
        print(f"  seam {seam}: {state}")

    try:
        from hermes_cli.config import load_config

        model_cfg = load_config().get("model") or {}
        provider = model_cfg.get("provider")
        runtime = model_cfg.get("claude_code_runtime")
        enabled = provider == "anthropic" and str(runtime or "").lower() == "claude_code"
        print(
            f"config gate: model.provider={provider!r} "
            f"model.claude_code_runtime={runtime!r} -> "
            f"{'ENABLED' if enabled else 'disabled'}"
        )
    except Exception as exc:
        print(f"config gate: unreadable ({exc})")

    try:
        from .cli import check_claude_binary

        ok, version_or_message = check_claude_binary()
        print(f"claude binary: {'ok ' + version_or_message if ok else version_or_message}")
    except Exception as exc:
        print(f"claude binary: check failed ({exc})")
    return 0


def cmd_smoke(args) -> int:
    from .cli import build_claude_child_env

    env = build_claude_child_env()
    print("child env: ANTHROPIC_API_KEY stripped, OAuth token injected")
    argv = [
        "claude",
        "-p",
        "Reply with exactly: ok",
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    if args.model:
        argv += ["--model", args.model]
    proc = subprocess.run(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )
    api_key_source = None
    verdict = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            api_key_source = event.get("apiKeySource")
            print(
                f"init: apiKeySource={api_key_source!r} model={event.get('model')!r}"
            )
        elif event.get("type") == "result":
            verdict = event
            print(
                "result: is_error=%r subtype=%r total_cost_usd=%r text=%r"
                % (
                    event.get("is_error"),
                    event.get("subtype"),
                    event.get("total_cost_usd"),
                    str(event.get("result"))[:60],
                )
            )
    if proc.returncode != 0 or verdict is None or verdict.get("is_error"):
        print(f"SMOKE FAILED (exit={proc.returncode})")
        return 1
    if api_key_source == "ANTHROPIC_API_KEY":
        print("SMOKE FAILED: billed to the metered API-key lane")
        return 1
    print("SMOKE OK: plan/OAuth lane")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-claude-code-addon")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("install-pth").set_defaults(func=cmd_install_pth)
    sub.add_parser("uninstall-pth").set_defaults(func=cmd_uninstall_pth)
    status = sub.add_parser("status")
    status.add_argument(
        "--import",
        dest="force_import",
        action="store_true",
        help="import the Hermes modules so real seam outcomes are shown",
    )
    status.set_defaults(func=cmd_status)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--model", default=None)
    smoke.set_defaults(func=cmd_smoke)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
