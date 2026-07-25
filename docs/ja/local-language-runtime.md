---
search:
  exclude: true
---
# GARVIS ローカル言語ランタイム v1

GARVIS は、プロバイダーに依存しないローカル生成パスを備えました。

- モデル重みはローカルの GGUF ファイルのままで、 Git によって無視されます。
- 推論はローカルでコンパイルされた llama.cpp 実行ファイルを使用します。
- このランタイムはホストされたモデルの API を呼び出しません。
- リクエストは生成前に決定論的なファイリング メタデータを受け取ります。
- 外部へのアクションは引き続き承認ゲート付きです。
- 暫定的な主張は、事実になるのではなく暫定のままです。

モデルを読み込まずにファイリングを確認:

```bash
uv run --no-dev garvis-local --show-filing "Maybe this is a scientific hypothesis"
```

ローカル応答を 1 件実行:

```bash
uv run --no-dev garvis-local "Explain the GARVIS local runtime"
```

既存のクラウド バックエンド対応の `garvis` コマンドはこの段階では削除されません。ローカル パスがデバイスのスモークテストに合格した後にのみ移行できます。