---
search:
  exclude: true
---
# GARVIS 会話ランタイム

GARVIS の応答スパイン（response spine）は、外界でのアクションを明示的に人間の管理下に置きつつ、通常の質問応答を復元します。

## 変更点

ランタイムは 2 つの関心事を分離します:

- **会話:** 質問、説明、分析、計算、下書き、計画、要約、そしてコードには通常どおりに回答します。
- **実行:** リモートデータの送信・公開・削除、稼働中アカウントの変更、金融取引などの副作用は、実行直前に Adrien D Thomas の正確な承認を必須とします。

デフォルトのランタイムには外界向けのツールは接続されていません。したがって、アシスタントは、送信・削除・公開・取引を誤って行うことなく、回答や作業準備ができます。

## セットアップ

現在のリポジトリのコードベースには Python 3.9 以上を使用します。`uv` でプロジェクトをインストールし、環境に API キーを設定します。API キーをコミットしてはいけません。

```bash
uv sync --all-extras --all-packages --group dev
export OPENAI_API_KEY="your-key-here"
```

モデルは `GARVIS_MODEL` で選択できます。デフォルトは `gpt-5.6-luna` です。

```bash
export GARVIS_MODEL="gpt-5.6-luna"
```

## 単発リクエストの実行

```bash
uv run garvis "Explain the current heartbeat status"
```

## 対話型セッションの開始

```bash
uv run garvis --interactive --session adrien
```

会話履歴はデフォルトで `~/.garvis/sessions.db` に保存されます。`--no-memory` で一時的なセッション、または `--db PATH` で別の SQLite データベースを指定できます。

## 承認の挙動

`How do I delete an old branch safely?` のようなリクエストは情報提供であり、通常の回答を受け取ります。`Delete the remote branch now` のようなリクエストは実行リクエストとして扱われます。GARVIS は正確なコマンドを用意し、結果（影響）を説明する場合がありますが、外部アクションは Adrien が正確に承認し、承認済みのツールが接続されるまで保留されます。

## アーキテクチャ

`garvis.assistant.GarvisAssistant` は会話エージェントとセッションメモリを所有します。リクエスト評価はノンブロッキングのメタデータであり、モデルの応答を置き換えたり抑制したりしません。アクションの承認は、質問応答の境界ではなく、ツールの境界に位置づけます。

著者: **Adrien D Thomas / ProCityHub** 。