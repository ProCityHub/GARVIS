---
search:
  exclude: true
---
# GARVIS のメモリ ライフサイクル

GARVIS は、プロバイダーに依存しない GGUF ランタイムのそばで、ローカルの SQLite メモリを保持します。関連する、範囲が制限されたコンテキストのみを想起し、各メモリにそのエビデンス状態のラベルを付けます。

モデル生成の応答は、信頼度が低いものとして `model_generated_unverified` に保存されます。検索によってそれらがエビデンスに昇格されることは決してありません。

自動メンテナンスにより、メモリが次の段階を遷移することがあります:

`active -> consolidated -> latent -> residual trace`

residual trace は、最小限の宛先 / タグ / キーワードのメタデータだけを保持します。完全な文言は消去され、trace はモデルのプロンプトへ挿入されることはありません。

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