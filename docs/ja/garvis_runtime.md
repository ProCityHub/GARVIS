---
search:
  exclude: true
---
# GARVIS 会話ランタイム

GARVIS の応答の骨格は、外部世界での操作を明示的な人間の管理下に置いたまま、通常の質疑応答を復元します。

## 変更点

ランタイムは 2 つの関心事を分離します。

- **会話:** 質問、説明、分析、計算、下書き、計画、要約、そしてコードには通常どおりに回答します。
- **実行:** 送信、公開、リモートデータの削除、稼働中アカウントの変更、金融取引などの副作用を伴う操作は、実行直前に Adrien D Thomas の正確な承認が必要です。

既定のランタイムには外部世界のツールは接続されていません。したがって、アシスタントは誤って送信・削除・公開・取引することなく、回答や下準備ができます。

## セットアップ

現在のリポジトリのコードベースには Python 3.9 以降を使用します。`uv` でプロジェクトをインストールし、環境に API キーを設定します。API キーをコミットしないでください。

```bash
uv sync --all-extras --all-packages --group dev
export OPENAI_API_KEY="your-key-here"
```

モデルは `GARVIS_MODEL` で選択でき、既定は `gpt-5.6-luna` です。

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

会話履歴は既定で `~/.garvis/sessions.db` に保存されます。一時セッションには `--no-memory`、別の SQLite データベースを使用するには `--db PATH` を指定します。

## 承認動作

`How do I delete an old branch safely?` のようなリクエストは情報提供であり、通常の回答を受け取ります。`Delete the remote branch now` のようなリクエストは実行リクエストとして扱われます。GARVIS は正確なコマンドを用意して影響を説明する場合がありますが、Adrien が正確に承認し、承認済みツールが接続されるまで、外部アクションは保留のままです。

## アーキテクチャ

`garvis.assistant.GarvisAssistant` は会話エージェントとセッションメモリを所有します。リクエスト評価はノンブロッキングなメタデータであり、モデルの応答を置き換えたり抑制したりしません。アクションの承認は質問応答の境界ではなく、ツールの境界で行われます。

Authorship: **Adrien D Thomas / ProCityHub**.