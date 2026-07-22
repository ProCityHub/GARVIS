---
search:
  exclude: true
---
# GARVIS メモリライフサイクル

GARVIS は現在、プロバイダーに依存しない GGUF ランタイムとは別にローカルの SQLite メモリを保持します。関連性が高く範囲が限定されたコンテキストのみを想起し、すべてのメモリにエビデンスの状態をラベル付けします。

モデルが生成した応答は `model_generated_unverified` として低信頼で保存され、取得によってエビデンスへ昇格することはありません。

自動メンテナンスにより、メモリは次のように移行する場合があります:

`active -> consolidated -> latent -> residual trace`

residual trace（残存トレース）は、最小限の宛先/タグ/キーワードのメタデータのみを保持します。本文は消去され、トレースがモデルのプロンプトに挿入されることは決してありません。

## コマンド

```bash
uv run --no-dev garvis-memory status
uv run --no-dev garvis-memory remember "Use local GGUF" --kind semantic
uv run --no-dev garvis-memory recall "local model"
uv run --no-dev garvis-memory maintain
uv run --no-dev garvis-memory maintain --apply
```

環境:

```bash
export GARVIS_MEMORY_DB="$HOME/.garvis/memory_lifecycle.db"
export GARVIS_MEMORY_POLICY="$HOME/GARVIS/config/garvis_memory_policy.json"
export GARVIS_MEMORY_ENABLED=1
```