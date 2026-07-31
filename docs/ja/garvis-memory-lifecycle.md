---
search:
  exclude: true
---
# GARVIS メモリライフサイクル

GARVIS は現在、ローカルな SQLite メモリを、プロバイダーに依存しない GGUF ランタイムと並行して保持します。関連性が高く、範囲が限定されたコンテキストのみを想起し、すべてのメモリにその証拠ステータスをラベル付けします。

モデル生成の応答は信頼度の低いものとして `model_generated_unverified` に保存され、検索によってそれが証拠に昇格することは決してありません。

自動メンテナンスにより、メモリが次の段階を移行する場合があります:

`active -> consolidated -> latent -> residual trace`

残留トレースは、最小限の宛先/タグ/キーワードのメタデータのみを保持します。完全な文面は消去され、トレースがモデル プロンプトに挿入されることは決してありません。

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