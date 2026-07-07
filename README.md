# hermes-claude-code-addon

Teaches an **unmodified** [Hermes Agent](https://github.com/NousResearch/hermes-agent)
checkout a new `claude_code` runtime that delegates whole turns to the real
`claude -p` binary (authenticated with your Claude Code OAuth token), so
usage is billed against your **Pro/Max plan quota** instead of the metered
"extra usage" lane that Hermes' own `anthropic_messages` client lands on.

Integration is **100% runtime monkeypatching** applied by a `.pth` startup
hook. No file in the Hermes checkout is ever modified — its `git status`
stays clean and `git pull` is unaffected. This is the successor to an
earlier in-tree fork, which was abandoned because a permanently-dirty
checkout is painful to manage against upstream.

## How it works

A `.pth` file in the venv's `site-packages` imports `hermes_claude_code.activate`
at interpreter startup, which registers **lazy** post-import hooks (a
`sys.meta_path` finder). Nothing is patched until Hermes actually imports the
target modules, so Python processes that never touch Hermes incur
essentially no overhead.

Four seams, each a wrapper that self-verifies the upstream symbol's shape
before patching:

| Seam | Module | What the wrapper does |
|---|---|---|
| 1 | `hermes_cli.runtime_provider` | `_VALID_API_MODES.add("claude_code")` + wrap `_maybe_apply_codex_app_server_runtime` to apply the opt-in gate |
| 2 | `agent.agent_init.init_agent` | restore `api_mode="claude_code"` (the stock acceptance set rejects it) |
| 3 | `agent.conversation_loop.run_conversation` | signature-identical wrapper: replays the upstream prologue, then forks `claude_code` agents to `runtime.run_claude_code_turn` |
| 4 | `run_agent.AIAgent.close` | drop the `ClaudeCodeSession` (kill any in-flight child) before stock teardown |

**All-or-nothing:** if any seam's upstream shape has drifted (signature,
source anchor, dataclass field), that seam records `failed: …` and the gate
**refuses** to switch any agent onto `claude_code` — it degrades to stock
Hermes behavior with a loud log line instead of running half-wired.

## Billing guard (the `$1,800` footgun)

The spawned child env must never carry `ANTHROPIC_API_KEY` — its presence
silently flips the binary onto metered pay-as-you-go billing.
`build_claude_child_env` starts from `hermes_subprocess_env(inherit_credentials=False)`,
strips every `ANTHROPIC_*` credential/routing var and every nested
`CLAUDE_CODE_*` marker (except the OAuth token), re-injects only
`CLAUDE_CODE_OAUTH_TOKEN`, and refuses to return an env that still has
`ANTHROPIC_API_KEY`. At runtime, a `system:init` event reporting
`apiKeySource == "ANTHROPIC_API_KEY"` kills the turn on the spot.

## Install

Installation is two steps: `uv` installs the package into the Hermes venv,
then the addon's own CLI writes the `.pth` startup hook:

```sh
VENV=~/.hermes/hermes-agent/venv

# 1. install the package into the Hermes venv (runtime deps: none)
uv pip install --python $VENV/bin/python -e .

# 2. write the .pth startup hook + show status
$VENV/bin/hermes-claude-code-addon install-pth
$VENV/bin/hermes-claude-code-addon status --import   # seam status, config gate, claude health
```

Ops (all via the installed console script, or `python -m hermes_claude_code.tool …`):

```sh
hermes-claude-code-addon status --import   # seam patch status + config gate + claude binary
hermes-claude-code-addon smoke             # billing smoke: real `claude -p`, assert plan/OAuth lane
hermes-claude-code-addon uninstall-pth     # remove the startup hook
```

Run the tests (pytest comes from the `test` dependency-group; `tests/conftest.py`
puts the Hermes tree on `sys.path`, so no PYTHONPATH plumbing is needed):

```sh
uv run --group test -- pytest tests/ -q     # 61 unit + integration tests
```

Then enable in `~/.hermes/config.yaml`:

```yaml
model:
  provider: anthropic
  claude_code_runtime: claude_code   # remove or set `auto` to disable
```

and restart the gateway/dashboard services. To disable without uninstalling,
set `HERMES_CLAUDE_CODE_ADDON=0`.

## Layout

```
hermes_claude_code/
  cli.py         # `claude -p` subprocess client + build_claude_child_env (billing guard)
  session.py     # per-Hermes-session resume chain, turn orchestration
  projector.py   # stream-json events → Hermes messages
  runtime.py     # run_claude_code_turn: the api_mode entry point
  gate.py        # pure opt-in gate (no Hermes imports)
  _postimport.py # lazy meta_path post-import hook machinery
  patcher.py     # the 4 seams + drift guards + all-or-nothing wiring
  activate.py    # .pth entry point (must never raise)
  tool.py        # ops CLI: install-pth / uninstall-pth / status / smoke
tests/           # unit (fakes) + integration (real Hermes tree)
```

The runtime modules (`cli`/`session`/`projector`/`runtime`) were ported from
the abandoned in-tree fork; the addon machinery is new.
