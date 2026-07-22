---
search:
  exclude: true
---
# GARVIS 会話ランタイム

GARVIS のレスポンス基盤は、外部世界でのアクションを明示的な人間の管理下に置いたまま、通常の質疑応答を復元します。

## 変更点

このランタイムは 2 つの関心事を分離します。

- **会話:** 質問、説明、分析、計算、草案、計画、要約、そして
  コードは通常どおりに回答されます。
- **実行:** 送信、公開、リモートデータの削除、本番アカウントの変更、金融取引などの副作用は、実行直前に Adrien D Thomas による厳密な承認が必要です。

既定のランタイムには外部世界のツールは接続されていません。そのため、アシスタントは、誤って送信・削除・公開・取引を行うことなく、回答や作業の準備ができます。

## セットアップ

現在のリポジトリのコードベースには Python 3.9 以降を使用してください。`uv` でプロジェクトをインストールし、環境に API キーを設定します。API キーは決してコミットしないでください。

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

## インタラクティブ会話の開始

```bash
uv run garvis --interactive --session adrien
```

会話履歴は既定で `~/.garvis/sessions.db` に保存されます。短期の一時セッションには `--no-memory` を、別の SQLite データベースを選ぶには `--db PATH` を使用してください。

## 承認の挙動

`How do I delete an old branch safely?` のようなリクエストは情報提供であり、通常の回答を受け取ります。`Delete the remote branch now` のようなリクエストは実行リクエストとして扱われます。GARVIS は正確なコマンドを準備して影響を説明する場合がありますが、外部アクションは Adrien が正確に承認し、承認済みのツールが接続されるまで保留されます。

## アーキテクチャ

`garvis.assistant.GarvisAssistant` は会話型エージェントとセッションメモリを管理します。リクエスト評価はノンブロッキングなメタデータであり、モデルの応答を置換・抑制しません。アクションの承認は質問応答の境界ではなくツール境界に属します。

著者: **Adrien D Thomas / ProCityHub** 。