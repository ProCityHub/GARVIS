---
search:
  exclude: true
---
# GARVIS 会話ランタイム

GARVIS のレスポンス基盤は、外部世界でのアクションを明示的に人間が制御できるように保ちながら、通常の質疑応答を復元します。

## 変更点

このランタイムは 2 つの関心事を分離します。

- **会話:** 質問、説明、分析、計算、下書き、計画、要約、そしてコードには通常どおりに回答します。
- **実行:** 送信、公開、リモートデータの削除、稼働中アカウントの変更、金融取引などの副作用のある操作は、実行直前に Adrien D Thomas の正確な承認が必要です。

デフォルトのランタイムには外部世界向けツールは接続されていません。したがって、アシスタントは、誤って送信・削除・公開・取引することなく、回答や作業の準備ができます。

## セットアップ

現在のリポジトリのコードベースには Python 3.9 以降を使用します。`uv` でプロジェクトをインストールし、環境に API キーを設定します。API キーをコミットしてはいけません。

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

## 対話型会話の開始

```bash
uv run garvis --interactive --session adrien
```

会話履歴はデフォルトで `~/.garvis/sessions.db` に保存されます。エフェメラルなセッションには `--no-memory` を、別の SQLite データベースを選ぶには `--db PATH` を使用します。

## 承認の挙動

`How do I delete an old branch safely?` のようなリクエストは情報提供目的であり、通常の回答を返します。`Delete the remote branch now` のようなリクエストは実行リクエストとして扱われます。GARVIS は正確なコマンドを用意し結果の影響を説明する場合がありますが、外部アクションは Adrien が正確に承認し、承認済みのツールが接続されるまで保留されます。

## アーキテクチャ

`garvis.assistant.GarvisAssistant` は会話用エージェントとセッションメモリを管理します。リクエスト評価はノンブロッキングなメタデータであり、モデルの応答を置き換えたり抑制したりしません。アクションの承認は質疑応答の境界ではなく、ツールの境界に属します。

著者: **Adrien D Thomas / ProCityHub** 。