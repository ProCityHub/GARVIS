---
search:
  exclude: true
---
# GARVIS ローカル言語ランタイム v1

GARVIS は、プロバイダーに依存しないローカル生成パスを備えました。

- モデルの重みはローカルの GGUF ファイルのままで、Git に無視されます。
- 推論はローカルでコンパイルされた llama.cpp 実行ファイルを使用します。
- このランタイムはホストされたモデル API を呼び出しません。
- リクエストは生成前に決定的なファイリング メタデータを受け取ります。
- 外部世界へのアクションは引き続き承認が必要です。
- 暫定的な主張は事実になるのではなく、引き続き暫定のままです。

モデルを読み込まずにファイリングを確認します:

```bash
uv run --no-dev garvis-local --show-filing "Maybe this is a scientific hypothesis"
```

ローカル応答を 1 回実行します:

```bash
uv run --no-dev garvis-local "Explain the GARVIS local runtime"
```

既存のクラウド バックエンドの `garvis` コマンドはこのフェーズでは削除されません。ローカル パスがデバイスのスモークテストに合格した後にのみ移行できます。