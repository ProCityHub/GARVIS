---
search:
  exclude: true
---
# GARVIS 会話ランタイム

GARVIS の応答の基幹は、通常の質疑応答を回復しつつ、外部の操作を明示的な人間の管理下に置きます。

## 変更点

このランタイムは 2 つの関心事を分離します。

- **会話:** 質問、説明、分析、計算、下書き、計画、要約、およびコードには通常どおりに応答します。
- **実行:** リモートデータの送信・公開・削除、稼働中アカウントの変更、金融取引などの副作用を伴う操作は、実行直前に Adrien D Thomas による明示的な承認が厳密に必要です。

既定のランタイムには外部のツールは接続されていません。そのため、アシスタントは誤って送信・削除・公開・取引を行うことなく、回答や作業準備ができます。

## セットアップ

現在のリポジトリのコードベースには Python 3.9 以降を使用します。`uv` でプロジェクトをインストールし、環境に API キーを設定します。API キーは決してコミットしないでください。

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

会話履歴は既定で `~/.garvis/sessions.db` に保存されます。使い捨てセッションには `--no-memory`、別の SQLite データベースを選ぶには `--db PATH` を使用します。

## 承認の挙動

`How do I delete an old branch safely?` のような依頼は情報提供であり、通常の回答を受け取ります。`Delete the remote branch now` のような依頼は実行リクエストとして扱われます。GARVIS は正確なコマンドを用意し結果の影響を説明できますが、Adrien が明示的に承認し、承認済みのツールがアタッチされるまで外部アクションは保留のままです。

## アーキテクチャ

`garvis.assistant.GarvisAssistant` は会話エージェントとセッションメモリを管理します。要求の評価はノンブロッキングなメタデータであり、モデル応答を置き換えたり抑制したりしません。アクションの承認は質疑応答の境界ではなく、ツールの境界に位置づけます。

著者: **Adrien D Thomas / ProCityHub**