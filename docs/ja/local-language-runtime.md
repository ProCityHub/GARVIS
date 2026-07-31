---
search:
  exclude: true
---
# GARVIS ローカル言語ランタイム v1

GARVIS はプロバイダー非依存のローカル生成パスを備えました。

- モデル重みはローカルの GGUF ファイルのままで、Git によって無視されます。
- 推論はローカルでコンパイルした llama.cpp 実行ファイルを使用します。
- このランタイムはホスト型モデルの API を呼び出しません。
- リクエストは生成前に決定的なファイリング メタデータを受け取ります。
- 外部世界へのアクションは引き続き承認ゲート付きです。
- 暫定的な主張は事実化せず、引き続き暫定のままです。

モデルを読み込まずにファイリングを確認:

```bash
uv run --no-dev garvis-local --show-filing "Maybe this is a scientific hypothesis"
```

ローカル応答を 1 回実行:

```bash
uv run --no-dev garvis-local "Explain the GARVIS local runtime"
```

既存のクラウド バックエンドの `garvis` コマンドはこのフェーズでは削除しません。ローカル パスがデバイスのスモークテストに合格した後にのみ移行できます。