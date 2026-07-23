---
search:
  exclude: true
---
# GARVIS 会話ランタイム

GARVIS の応答スパインは、外部世界へのアクションを明示的な人間の制御下に置きつつ、通常の質問応答を復元します。

## 変更点

このランタイムは 2 つの関心事を分離します。

- **会話:** 質問、説明、分析、計算、下書き、計画、要約、およびコードには通常どおりに回答します。
- **実行:** 送信、公開、リモートデータの削除、稼働中のアカウントの変更、金融取引などの副作用を伴う操作には、実行直前に Adrien D Thomas の厳密な承認が必要です。

デフォルトのランタイムには外部世界のツールは接続されていません。したがって、アシスタントは、送信・削除・公開・取引を誤って行うことなく、回答や作業の準備ができます。

## セットアップ

現在のリポジトリのコードベースには Python 3.9 以降を使用します。`uv` でプロジェクトをインストールし、環境に API キーを設定します。API キーをコミットしないでください。

```bash
uv sync --all-extras --all-packages --group dev
export OPENAI_API_KEY="your-key-here"
```

モデルは `GARVIS_MODEL` で選択できます。デフォルトは `gpt-5.6-luna` です。

```bash
export GARVIS_MODEL="gpt-5.6-luna"
```

## 単一リクエストの実行

```bash
uv run garvis "Explain the current heartbeat status"
```

## インタラクティブな会話の開始

```bash
uv run garvis --interactive --session adrien
```

会話履歴はデフォルトで `~/.garvis/sessions.db` に保存されます。`--no-memory` を使用すると一時的なセッションになり、`--db PATH` で別の SQLite データベースを指定できます。

## 承認の挙動

`How do I delete an old branch safely?` のようなリクエストは情報提供であり、通常の回答を返します。`Delete the remote branch now` のようなリクエストは実行リクエストとして扱われます。GARVIS は正確なコマンドを用意し、その結果を説明する場合がありますが、外部アクションは Adrien が厳密に承認して承認済みツールが接続されるまで保留されます。

## アーキテクチャ

`garvis.assistant.GarvisAssistant` は会話用 エージェント とセッションメモリを保持します。リクエストの評価はノンブロッキングなメタデータであり、モデルの応答を置き換えたり抑制したりしません。アクションの承認は質問応答の境界ではなく、ツールの境界に属します。

著者: **Adrien D Thomas / ProCityHub**。