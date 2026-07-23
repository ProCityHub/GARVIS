---
search:
  exclude: true
---
# GARVIS メモリライフサイクル

GARVIS は、プロバイダーに依存しない GGUF ランタイムの横でローカルの SQLite メモリを保持します。関連する限定的なコンテキストのみを想起し、すべてのメモリに証拠ステータスのラベルを付与します。

モデル生成の応答は、`model_generated_unverified` として低信頼で保存されます。検索によってそれらが証拠に格上げされることはありません。

自動メンテナンスにより、メモリは次の段階を遷移する場合があります:

`active -> consolidated -> latent -> residual trace`

residual trace は、宛先/タグ/キーワードの最小限のメタデータのみを保持します。全文は消去され、trace はモデルのプロンプトに挿入されることは決してありません。

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