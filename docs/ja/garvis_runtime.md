---
search:
  exclude: true
---
# GARVIS 会話ランタイム

GARVIS レスポンススパインは、外部世界でのアクションを明示的な人間の管理下に置きつつ、通常の質問応答を復元します。

## 変更点

このランタイムは 2 つの関心事を分離します。

- **会話:** 質問、説明、分析、計算、下書き、計画、要約、およびコードには、通常どおりに応答します。
- **実行:** リモートデータの送信・公開・削除、稼働中アカウントの変更、金融取引、その他同様の副作用は、実行直前に Adrien D Thomas の厳密な承認が必要です。

デフォルトのランタイムには外部世界のツールが接続されていません。したがってアシスタントは、誤って送信・削除・公開・取引を行うことなく、回答や作業準備ができます。

## セットアップ

現在のリポジトリのコードベースには Python 3.9 以降を使用します。`uv` でプロジェクトをインストールし、その後に環境変数で API キーを設定します。API キーをコミットしてはいけません。

```bash
uv sync --all-extras --all-packages --group dev
export OPENAI_API_KEY="your-key-here"
```

モデルは `GARVIS_MODEL` で選択でき、デフォルトは `gpt-5.6-luna` です。

```bash
export GARVIS_MODEL="gpt-5.6-luna"
```

## 1 件のリクエストを実行

```bash
uv run garvis "Explain the current heartbeat status"
```

## 対話型の会話を開始

```bash
uv run garvis --interactive --session adrien
```

会話履歴はデフォルトで `~/.garvis/sessions.db` に保存されます。使い捨てセッションには `--no-memory` を、別の SQLite データベースを選ぶには `--db PATH` を使用します。

## 承認の挙動

`How do I delete an old branch safely?` のようなリクエストは情報提供であり、通常の回答を受け取ります。`Delete the remote branch now` のようなリクエストは実行リクエストとして扱われます。GARVIS は正確なコマンドを用意し、その結果を説明する場合がありますが、外部でのアクションは、Adrien が厳密に承認し、承認済みのツールが接続されるまで保留されます。

## アーキテクチャ

`garvis.assistant.GarvisAssistant` は会話用 エージェント とセッションメモリを所有します。リクエストの評価はノンブロッキングのメタデータであり、モデルのレスポンスを置き換えたり抑制したりしません。アクションの承認は質問応答の境界ではなく、ツールの境界に属します。

著者:  **Adrien D Thomas / ProCityHub** 。