---
search:
  exclude: true
---
# GARVIS 会話ランタイム

GARVIS 応答スパインは、外部での操作を明示的な人間の管理下に置きつつ、通常の質疑応答を復元します。

## 変更点

このランタイムは 2 つの関心事を分離します。

- **会話:** 質問、説明、分析、計算、下書き、計画、要約、そしてコードは通常どおりに回答します。
- **実行:** リモートデータの送信・公開・削除、稼働中アカウントの変更、金融取引などの副作用は、実行直前に Adrien D Thomas の厳密な承認が必要です。

既定のランタイムには外部のツールは接続されていません。したがってアシスタントは、送信・削除・公開・取引を誤って行うことなく、回答や作業の準備ができます。

## セットアップ

現在のリポジトリのコードベースには Python 3.9 以降を使用します。`uv` でプロジェクトをインストールし、環境に API キーを設定してください。API キーをコミットしないでください。

```bash
uv sync --all-extras --all-packages --group dev
export OPENAI_API_KEY="your-key-here"
```

モデルは `GARVIS_MODEL` で選択できます。既定は `gpt-5.6-luna` です。

```bash
export GARVIS_MODEL="gpt-5.6-luna"
```

## 1 件のリクエストの実行

```bash
uv run garvis "Explain the current heartbeat status"
```

## 対話の開始

```bash
uv run garvis --interactive --session adrien
```

会話履歴は既定で `~/.garvis/sessions.db` に保存されます。使い捨てのセッションには `--no-memory`、別の SQLite データベースを選ぶには `--db PATH` を使用します。

## 承認の動作

`How do I delete an old branch safely?` のようなリクエストは情報提供であり、通常の回答を受け取ります。`Delete the remote branch now` のようなリクエストは実行リクエストとして扱われます。GARVIS は正確なコマンドを用意し、結果の影響を説明できますが、外部での実行は Adrien が厳密に承認し、承認済みツールが接続されるまで保留されます。

## アーキテクチャ

`garvis.assistant.GarvisAssistant` は会話エージェントとセッションメモリを保持します。リクエスト評価はノンブロッキングなメタデータであり、モデル応答を置き換えたり抑制したりしません。アクション承認は質疑応答の境界ではなく、ツールの境界に属します。

著者: **Adrien D Thomas / ProCityHub**.