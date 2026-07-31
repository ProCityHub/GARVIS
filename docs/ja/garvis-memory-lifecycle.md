---
search:
  exclude: true
---
# GARVIS メモリライフサイクル

<<<<<<< HEAD
GARVIS は現在、ローカルの SQLite メモリを、プロバイダーに依存しない GGUF ランタイムと並行して保持します。関連する、限定されたコンテキストのみを想起し、すべてのメモリにそのエビデンス状態のラベルを付けます。

モデル生成の応答は `model_generated_unverified` として低い確信度で保存されます。リトリーバルによってエビデンスへ昇格することは決してありません。
=======
GARVIS は現在、プロバイダーに依存しない GGUF ランタイムと併せてローカル SQLite メモリを保持します。関連し、範囲が限定されたコンテキストのみを想起し、各メモリにそのエビデンス状態のラベルを付与します。

モデル生成の応答は低い確信度で `model_generated_unverified` として保存され、検索によってエビデンスへ格上げされることはありません。
>>>>>>> origin/main

自動メンテナンスにより、メモリは次の段階を遷移する場合があります:

`active -> consolidated -> latent -> residual trace`

<<<<<<< HEAD
residual trace （残留トレース）は、宛先/タグ/キーワードの最小限のメタデータのみを保持します。本文は消去され、トレースがモデルのプロンプトに挿入されることは決してありません。
=======
残差トレース（residual trace）は最小限の宛先/タグ/キーワードのメタデータのみを保持します。全文は消去され、トレースがモデルのプロンプトに挿入されることは決してありません。
>>>>>>> origin/main

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