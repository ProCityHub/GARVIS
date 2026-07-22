---
search:
  exclude: true
---
# GARVIS ローカル言語ランタイム v1

GARVIS は、プロバイダー非依存のローカル生成パスを備えました。

- モデルの重みはローカルの GGUF ファイルのままで、Git によって無視されます。
- 推論はローカルでコンパイルされた llama.cpp 実行ファイルを使用します。
- この runtime はホスト型モデルの API を呼び出しません。
- リクエストは生成の前に決定的なファイリングメタデータを受け取ります。
- 外部でのアクションは引き続き承認が必要です。
- 暫定的な主張は事実になるのではなく、引き続き暫定のままです。

モデルを読み込まずにファイリングを検査:

```bash
uv run --no-dev garvis-local --show-filing "Maybe this is a scientific hypothesis"
```

ローカル応答を 1 回実行:

```bash
uv run --no-dev garvis-local "Explain the GARVIS local runtime"
```

既存のクラウド連携の `garvis` コマンドはこの段階では削除されません。ローカル経路がデバイスのスモークテストに合格した後にのみ移行できます。