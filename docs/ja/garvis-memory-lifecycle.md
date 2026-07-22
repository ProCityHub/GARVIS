---
search:
  exclude: true
---
# GARVIS メモリ ライフサイクル

GARVIS は現在、プロバイダー非依存の GGUF ランタイムのほかに、ローカル SQLite メモリを保持します。関連性があり範囲が限定されたコンテキストのみを想起し、各メモリにその証拠ステータスをラベル付けします。

モデル生成の応答は、低信頼度として `model_generated_unverified` に保存されます。取得によってそれらが証拠に昇格することはありません。

自動メンテナンスにより、メモリは次の段階へ遷移する場合があります:

`active -> consolidated -> latent -> residual trace`

残留トレースは、宛先/タグ/キーワードの最小限のメタデータのみを保持します。文章の全文は消去され、トレースがモデルのプロンプトに挿入されることは決してありません。

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