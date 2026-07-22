---
search:
  exclude: true
---
# GARVIS ローカル言語ランタイム v1

GARVIS は、プロバイダー非依存のローカル生成パスを備えました。

- モデル重みはローカルの GGUF ファイルのままで、Git によって無視されます。
- 推論はローカルでコンパイルした llama.cpp の実行ファイルを使用します。
- このランタイムはホスト型モデル API を呼び出しません。
- リクエストには、生成前に決定的なファイリングのメタデータが付与されます。
- 外界へのアクションは引き続き承認が必要です。
- 暫定的な主張は、事実として確定するのではなく、引き続き暫定のまま扱われます。

モデルを読み込まずにファイリングを検査:

```bash
uv run --no-dev garvis-local --show-filing "Maybe this is a scientific hypothesis"
```

ローカル応答を 1 回実行:

```bash
uv run --no-dev garvis-local "Explain the GARVIS local runtime"
```

既存のクラウド連携の `garvis` コマンドは、このフェーズでは削除されません。ローカル経路がデバイスのスモークテストに合格した後にのみ移行できます。