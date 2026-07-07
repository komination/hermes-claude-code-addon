# hermes-claude-code-addon

[Hermes Agent](https://github.com/NousResearch/hermes-agent) 本体には
**一切手を加えずに**、新しい `claude_code` ランタイムを追加するアドオンです。
各ターンの処理を丸ごと本物の `claude -p` バイナリ（Claude Code の OAuth
トークンで認証）に任せるため、Anthropic の課金は、Hermes 標準の
`anthropic_messages` クライアントで発生する従量課金（"extra usage"）ではなく、
**Pro/Max プランの枠内**で処理されます。

組み込みはすべて実行時のモンキーパッチで行い、`.pth` の起動フックで適用します。
Hermes の checkout 内のファイルは一切書き換えないので、`git status` はクリーン
なまま、`git pull` も問題なく通ります。以前は Hermes を直接改変する fork として
開発していましたが、常に dirty な checkout を upstream に追従させ続けるのは
無理があり、その反省から生まれたのが本方式です。

## 仕組み

venv の `site-packages` に置いた `.pth` ファイルが、Python の起動時に
`hermes_claude_code.activate` を import し、遅延実行される post-import フック
（`sys.meta_path` の finder）を登録します。実際にパッチが当たるのは Hermes が
対象モジュールを import した時点なので、Hermes を使わない Python プロセスには
ほとんど負荷がかかりません。

パッチを当てる箇所（seam）は 4 つ。いずれも、適用前に upstream 側のシンボルが
想定どおりの形かを自分で確かめる wrapper になっています:

| Seam | モジュール | 役割 |
|---|---|---|
| 1 | `hermes_cli.runtime_provider` | `_VALID_API_MODES` に `"claude_code"` を追加し、`_maybe_apply_codex_app_server_runtime` を wrap して opt-in の判定を組み込む |
| 2 | `agent.agent_init.init_agent` | `api_mode="claude_code"` を復元する（素の Hermes は未知のモードとして弾いてしまうため） |
| 3 | `agent.conversation_loop.run_conversation` | 元と同じシグネチャの wrapper。upstream の前処理をなぞったうえで、`claude_code` エージェントのターンだけを `runtime.run_claude_code_turn` に振り分ける |
| 4 | `run_agent.AIAgent.close` | 素の teardown の前に `ClaudeCodeSession` を破棄する（実行中の子プロセスがあれば kill） |

**一箇所でも失敗したら全体を止める:** upstream 側の形が想定から外れていた場合
（シグネチャ、ソース上の目印、dataclass のフィールドの変化）、その seam は
`failed: …` として記録され、ゲートは**どのエージェントも `claude_code` に
切り替えません**。中途半端にパッチが当たった状態で動かすのではなく、はっきり
分かるログを残したうえで、素の Hermes の動作に戻ります。

## 課金ガード（$1,800 の事故を防ぐ）

子プロセスの環境変数に `ANTHROPIC_API_KEY` が紛れ込むと、`claude` バイナリは
何も言わずに従量課金へ切り替わってしまいます。これを防ぐため、子プロセスの
環境は必ず `build_claude_child_env` で組み立てます。
`hermes_subprocess_env(inherit_credentials=False)` を起点に、`ANTHROPIC_*` の
認証・接続先関連の変数と、親から引き継いだ `CLAUDE_CODE_*` 系の変数
（OAuth トークンを除く）をすべて取り除き、`CLAUDE_CODE_OAUTH_TOKEN` だけを
入れ直します。それでも `ANTHROPIC_API_KEY` が残っていた場合は、環境の生成
自体を失敗させます。さらに実行時にも二重に防いでおり、`system:init` イベント
の `apiKeySource` が `"ANTHROPIC_API_KEY"` だった場合は、その場でターンを
打ち切ります。

## インストール

手順は 2 段階です。まず `uv` でパッケージを Hermes の venv に入れ、次に
アドオン付属の CLI で `.pth` 起動フックを書き込みます:

```sh
VENV=~/.hermes/hermes-agent/venv

# 1. パッケージを Hermes の venv にインストール（実行時の依存: なし）
uv pip install --python $VENV/bin/python -e .

# 2. .pth 起動フックを書き込み、状態を確認
$VENV/bin/hermes-claude-code-addon install-pth
$VENV/bin/hermes-claude-code-addon status --import   # seam の状態、config の判定、claude バイナリの確認
```

運用コマンド（いずれもインストールされる console script、または
`python -m hermes_claude_code.tool …` で実行できます）:

```sh
hermes-claude-code-addon status --import   # seam のパッチ状況 + config の判定 + claude バイナリ
hermes-claude-code-addon smoke             # 課金の smoke テスト: 本物の `claude -p` を起動し、プラン/OAuth 側で課金されることを確認
hermes-claude-code-addon uninstall-pth     # 起動フックを削除
```

テストは次のように実行します（pytest は `test` の dependency group に入って
います。`tests/conftest.py` が Hermes のツリーを `sys.path` に加えるので、
PYTHONPATH を通す必要はありません）:

```sh
uv run --group test -- pytest tests/ -q     # unit + integration 全 61 テスト
```

最後に `~/.hermes/config.yaml` で有効化し、

```yaml
model:
  provider: anthropic
  claude_code_runtime: claude_code   # このキーを消すか `auto` にすると無効
```

gateway/dashboard のサービスを再起動します。アンインストールせずに一時的に
止めたいときは、環境変数 `HERMES_CLAUDE_CODE_ADDON=0` を設定してください。

## 構成

```
hermes_claude_code/
  cli.py         # `claude -p` を子プロセスとして扱うクライアント + build_claude_child_env（課金ガード）
  session.py     # Hermes セッションごとの resume の連鎖と、ターンの進行管理
  projector.py   # stream-json のイベントを Hermes のメッセージ形式に変換
  runtime.py     # run_claude_code_turn: api_mode の入口
  gate.py        # opt-in 判定だけを行う純粋なゲート（Hermes に依存しない）
  _postimport.py # 遅延 post-import フックの仕組み（sys.meta_path）
  patcher.py     # 4 つの seam + 形状チェック + 全体を一括で有効/無効にする配線
  activate.py    # .pth の入口（決して例外を投げてはいけない）
  tool.py        # 運用 CLI: install-pth / uninstall-pth / status / smoke
tests/           # unit（fake 使用）+ integration（実際の Hermes ツリーを使用）
```

ランタイム側のモジュール（`cli`/`session`/`projector`/`runtime`）は、廃止した
in-tree fork から移植したものです。アドオンとしての仕組みの部分は新規に書いて
います。
