---
search:
  exclude: true
---
# GARVIS の会話ランタイム

GARVIS の応答の基盤（response spine）は、外部でのアクションを明示的な人間の管理下に置きつつ、通常の質疑応答を復元します。

## 変更点

ランタイムは 2 つの関心事を分離します。

- **会話:** 質問、説明、分析、計算、下書き、計画、要約、およびコードも通常どおり回答します。
- **実行:** 送信、公開、リモートデータの削除、実アカウントの変更、金融取引などの副作用を伴う操作は、実行直前に Adrien D Thomas の明確な承認が必要です。

既定のランタイムには外部ツールは接続されていません。したがってアシスタントは、誤って送信・削除・公開・取引することなく、回答や作業の準備ができます。

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

## 単一リクエストの実行

```bash
uv run garvis "Explain the current heartbeat status"
```

## 対話型の会話を開始

```bash
uv run garvis --interactive --session adrien
```

会話履歴は既定で `~/.garvis/sessions.db` に保存されます。一時セッションには `--no-memory` を、別の SQLite データベースを選ぶには `--db PATH` を使用してください。

## 承認の挙動

`How do I delete an old branch safely?` のようなリクエストは情報提供であり、通常の回答が返ります。`Delete the remote branch now` のようなリクエストは実行リクエストとして扱われます。GARVIS は正確なコマンドを用意し、その影響を説明することはありますが、Adrien が明確に承認し、承認済みのツールが接続されるまで外部アクションは保留のままです。

## アーキテクチャ

`garvis.assistant.GarvisAssistant` は会話用エージェントとセッションメモリを管理します。リクエストの評価はノンブロッキングなメタデータであり、モデル応答を置き換えたり抑制したりしません。アクションの承認は質問応答の境界ではなく、ツールの境界で行われます。

著者: **Adrien D Thomas / ProCityHub**.