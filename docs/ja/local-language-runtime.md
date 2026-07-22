---
search:
  exclude: true
---
# GARVIS ローカル言語ランタイム v1

GARVIS は、プロバイダーに依存しないローカル生成パスを備えました。

- モデル重みはローカルの GGUF ファイルのままで、Git 管理対象外です。
- 推論はローカルでコンパイルされた llama.cpp 実行ファイルを使用します。
- このランタイムはホスト型モデル API を呼び出しません。
- リクエストは生成前に決定的なファイリング用メタデータを受け取ります。
- 外部へのアクションは承認必須のままです。
- 仮の主張は事実化せず、仮のままです。

モデルを読み込まずにファイリングを確認:

```bash
uv run --no-dev garvis-local --show-filing "Maybe this is a scientific hypothesis"
```

ローカル応答を 1 回実行:

```bash
uv run --no-dev garvis-local "Explain the GARVIS local runtime"
```

既存のクラウド連携の `garvis` コマンドは、このフェーズでは削除されません。ローカルパスがデバイスのスモークテストに合格した後にのみ移行できます。