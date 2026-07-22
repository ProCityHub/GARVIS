---
search:
  exclude: true
---
# GARVIS ローカル言語ランタイム v1

GARVIS は、プロバイダー非依存のローカル生成経路を備えました。

- モデルの重みはローカルの GGUF ファイルに保持され、Git によって無視されます。
- 推論はローカルでコンパイルされた llama.cpp 実行ファイルを使用します。
- このランタイムではホストされたモデルの API は一切呼び出されません。
- リクエストは生成前に決定的なファイリング用メタデータを受け取ります。
- 外部世界へのアクションは引き続き承認ゲート付きです。
- 暫定的な主張は、事実になるのではなく暫定のままに保たれます。

モデルを読み込まずにファイリングを検査:

```bash
uv run --no-dev garvis-local --show-filing "Maybe this is a scientific hypothesis"
```

ローカルの応答を 1 回実行:

```bash
uv run --no-dev garvis-local "Explain the GARVIS local runtime"
```

既存のクラウド対応の `garvis` コマンドは、この段階では削除されません。ローカル経路がデバイスのスモークテストに合格した後にのみ移行できます。