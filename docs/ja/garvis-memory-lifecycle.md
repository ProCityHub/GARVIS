---
search:
  exclude: true
---
# GARVIS メモリライフサイクル

GARVIS は、プロバイダー非依存の GGUF ランタイムのそばにローカルの SQLite メモリを保持します。関連する限定的なコンテキストのみを想起し、すべてのメモリに証拠ステータスのラベル付けを行います。

モデル生成の応答は信頼度が低いものとして `model_generated_unverified` に保存されます。リトリーバルによってそれが証拠に格上げされることは決してありません。

自動メンテナンスにより、メモリは次の段階を移行する場合があります:

`active -> consolidated -> latent -> residual trace`

residual trace では、宛先/タグ/キーワードの最小限のメタデータのみを保持します。本文は消去され、trace はモデルのプロンプトに挿入されることはありません。

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