---
search:
  exclude: true
---
# GARVIS メモリライフサイクル

GARVIS は現在、プロバイダーに依存しない GGUF ランタイムのほかに、ローカルの SQLite メモリを保持します。関連する限定的なコンテキストのみを想起し、すべてのメモリに証拠ステータスのラベルを付与します。

モデル生成の応答は、低い確信度で `model_generated_unverified` として保存されます。取得によってそれらが証拠へ格上げされることは決してありません。

自動メンテナンスにより、メモリは次の段階へ移行することがあります:

`active -> consolidated -> latent -> residual trace`

残存トレースは、最小限の destination / tag / keyword メタデータのみを保持します。完全な文言は消去され、トレースがモデルのプロンプトに挿入されることは決してありません。

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