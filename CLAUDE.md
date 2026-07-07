# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An out-of-tree addon that teaches an **unmodified** Hermes Agent checkout (default `~/.hermes/hermes-agent`, override with `HERMES_REPO`) a new `claude_code` api_mode: whole turns are delegated to a spawned `claude -p` subprocess authenticated with the user's Claude Code OAuth token, so usage bills against the Pro/Max **plan quota** instead of the metered API lane. Integration is 100% runtime monkeypatching via a `.pth` startup hook — no file in the Hermes checkout is ever modified.

The package has **zero runtime dependencies** and is not independently installable — everything Hermes-side comes from the host venv it is installed into; everything else is stdlib. Keep it that way.

## Commands

```sh
# run all tests (pytest comes from the PEP 735 `test` dependency-group)
uv run --group test -- pytest tests/ -q

# run a single test file / test
uv run --group test -- pytest tests/test_patcher.py -q
uv run --group test -- pytest tests/test_session.py::TestRunTurn -q

# ops CLI (run with the Hermes venv python once installed there)
hermes-claude-code-addon status --import   # seam patch status + config gate + claude binary health
hermes-claude-code-addon smoke             # real `claude -p` billing smoke: asserts plan/OAuth lane
hermes-claude-code-addon install-pth / uninstall-pth
```

Tests need no PYTHONPATH plumbing: `tests/conftest.py` prepends the addon root, the Hermes checkout, and the Hermes venv's site-packages to `sys.path`. `tests/test_integration_hermes.py` runs against the **real** Hermes tree and auto-skips (`pytest.importorskip`) when it isn't importable; the other test files use fakes.

## Architecture

Two layers: **addon machinery** (how the patches get applied) and the **runtime** (what a claude_code turn does).

### Addon machinery: `.pth` → activate → post-import hooks → seams

- `tool.py` `install-pth` writes `zzz_hermes_claude_code_addon.pth` into site-packages, which imports `activate.py` at every interpreter startup. `activate.py` **must never raise** (it would traceback every python process in the venv) and honors the `HERMES_CLAUDE_CODE_ADDON=0` kill switch.
- `_postimport.py` is dependency-free wrapt-style machinery: a `sys.meta_path` finder that fires callbacks right after a watched module executes. Patching is therefore **lazy** — processes that never import Hermes pay ~nothing.
- `patcher.py` registers the four seams. Each seam patcher **self-verifies the upstream shape** (signature, source anchors, dataclass fields) before wrapping, and records `"patched"` / `"failed: <reason>"` in a status map:
  1. `hermes_cli.runtime_provider` — add `claude_code` to `_VALID_API_MODES` and wrap `_maybe_apply_codex_app_server_runtime` to apply the opt-in gate.
  2. `agent.agent_init.init_agent` — restore `api_mode="claude_code"` after stock code rejects the unknown mode.
  3. `agent.conversation_loop.run_conversation` — signature-identical wrapper; non-claude_code agents pass through untouched. For claude_code it replays the upstream prologue (reading helpers from conversation_loop's module namespace at call time, so upstream monkeypatch seams keep working) then forks into `runtime.run_claude_code_turn`. The prologue replica is pinned to upstream commit 22c5048d9.
  4. `run_agent.AIAgent.close` — drop the `ClaudeCodeSession` (kill in-flight child) before stock teardown.
- **All-or-nothing**: the gate wrapper (seam 1) calls `patcher.downstream_ready()` before rewriting api_mode; if any downstream seam failed, it logs loudly once and leaves the agent on stock Hermes behavior. Never bypass this — a half-wired runtime is the failure mode this design exists to prevent.
- `gate.py` is a pure function with no Hermes imports: `model.provider: anthropic` + `model.claude_code_runtime: claude_code` in config.yaml opts a model in.
- Wrapped functions carry the `__hermes_claude_code_addon__` marker attribute (idempotency) and `__hermes_claude_code_original__` (the original).

### Runtime: one `claude -p` child per turn

- `runtime.py` `run_claude_code_turn` is the api_mode entry point (sibling of Hermes' codex app-server path). It lazily creates one `ClaudeCodeSession` per agent (stored as `agent._claude_session`), writes an MCP config for the hermes-tools grandchild (env forwarding is a deliberate **allowlist** — `HERMES_*` includes Tier-1 secrets that must not be persisted to disk), and maps stream events to tool-progress breadcrumbs.
- `session.py` `ClaudeCodeSession` owns the resume chain: first turn passes `--session-id <fresh uuid>`, later turns `--resume`. Hermes' SessionDB stays the canonical history; Claude Code's transcript is just how the child gets context. Classifies terminal failures (OAuth expiry hints, resume-target-gone → drop the chain, extra-usage lane) and sets `TurnResult.billing_lane` (`"plan"` is the only acceptable value; note `apiKeySource == "none"` means OAuth on 2.1.x).
- `cli.py` `ClaudeCodeCLI` drives the one-shot `claude -p ... --output-format stream-json --verbose` child and yields NDJSON events.
- `projector.py` converts stream-json events into OpenAI-shaped `{role, content, tool_calls, tool_call_id}` messages Hermes already understands; stateful across one turn (thinking text stashed onto the next assistant message, terminal metadata recorded).

### The billing guard (the "$1,800 footgun")

An inherited `ANTHROPIC_API_KEY` silently flips the child onto metered API billing. Two layers, both non-negotiable:

- **Spawn side**: `cli.build_claude_child_env` is the ONLY sanctioned way to build the child env. It starts from `hermes_subprocess_env(inherit_credentials=False)`, strips all `ANTHROPIC_*` credential/routing vars and all nested `CLAUDE_CODE_*`/`CLAUDECODE` markers except the OAuth token, re-injects only `CLAUDE_CODE_OAUTH_TOKEN`, and refuses to return an env still carrying `ANTHROPIC_API_KEY`.
- **Runtime side**: if the `system:init` event reports `apiKeySource == "ANTHROPIC_API_KEY"`, the session kills the turn immediately (`billing_lane="api_key"`).

Any change touching child-env construction or event handling must preserve both layers; `hermes-claude-code-addon smoke` verifies the lane end-to-end against the real binary.

## Conventions

- Upstream-shape assumptions are enforced in `_patch_*` via `_require(...)` checks (signatures, source anchors like `"build_turn_context("`, `TurnContext` fields). If you change what the wrappers rely on, add/update the corresponding drift guard — an unguarded assumption defeats the all-or-nothing design.
- The runtime modules deliberately mirror their codex siblings in Hermes (`codex_app_server_session.py`, `codex_event_projector.py`); when in doubt about behavior, match the codex path's contract.
- Event shapes and version-specific behavior are verified live against a pinned Claude Code version (`MIN_CLAUDE_VERSION` in cli.py, currently tested on 2.1.201); note the verified version in comments when relying on observed behavior.
