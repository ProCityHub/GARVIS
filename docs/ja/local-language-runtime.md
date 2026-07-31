---
search:
  exclude: true
---
# GARVIS ローカル言語ランタイム v1

GARVIS は、プロバイダーに依存しないローカル生成パスを備えました。

- モデルの重みはローカルの GGUF ファイルのままで、Git によって無視されます。
- 推論はローカルでコンパイルされた llama.cpp 実行ファイルを使用します。
- このランタイムはホスト型モデルの API を呼び出しません。
- リクエストは生成前に決定的な filing メタデータを受け取ります。
- 外部へのアクションは引き続き承認が必要です。
- 仮の主張は事実化せず、引き続き仮のままです。

モデルを読み込まずに filing を確認:

```bash
uv run --no-dev garvis-local --show-filing "Maybe this is a scientific hypothesis"
```

ローカル応答を 1 回実行:

```bash
uv run --no-dev garvis-local "Explain the GARVIS local runtime"
```

既存のクラウド バックエンドの `garvis` コマンドは、この段階では削除されません。ローカル パスがデバイスのスモークテストに合格した後にのみ移行できます。
