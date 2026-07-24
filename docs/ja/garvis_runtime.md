---
search:
  exclude: true
---
# GARVIS 対話ランタイム

GARVIS のレスポンス基盤は、外部世界でのアクションを明示的に人間が制御できるように保ちながら、通常の質問応答を取り戻します。

## 変更点

このランタイムは、2 つの関心事を分離します。

- **会話:** 質問、説明、分析、計算、下書き、計画、要約、そしてコードには通常どおりに応答します。
- **実行:** リモートデータの送信・公開・削除、稼働中アカウントの変更、金融取引などの副作用を伴う操作は、実行直前に Adrien D Thomas による厳密な承認が必須です。

デフォルトのランタイムには外部世界のツールは接続されていません。したがって、アシスタントは、誤って送信・削除・公開・取引してしまうことなく回答や作業準備ができます。

## セットアップ

現在のリポジトリのコードベースには Python 3.9 以降を使用します。`uv` でプロジェクトをインストールし、環境に API キーを設定してください。API キーをコミットしてはいけません。

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

会話履歴はデフォルトで `~/.garvis/sessions.db` に保存されます。使い捨てのセッションには `--no-memory`、別の SQLite データベースを使うには `--db PATH` を指定します。

## 承認の挙動

`How do I delete an old branch safely?` のようなリクエストは情報提供であり、通常の回答を受け取ります。`Delete the remote branch now` のようなリクエストは実行リクエストとして扱われます。GARVIS は正確なコマンドを準備して影響を説明する場合がありますが、外部アクションは Adrien が厳密に承認し、承認済みツールが接続されるまで保留されます。

## アーキテクチャ

`garvis.assistant.GarvisAssistant` は対話型エージェントとセッションメモリを所有します。リクエスト評価はノンブロッキングなメタデータであり、モデルの応答を置き換えたり抑制したりしません。アクションの承認は質問応答の境界ではなく、ツールの境界に属します。

著者:  **Adrien D Thomas / ProCityHub**