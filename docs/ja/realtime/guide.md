---
search:
  exclude: true
---
# ガイド

このガイドでは、 OpenAI Agents SDK の realtime 機能を使って音声対応の AI エージェントを構築する方法を詳しく説明します。

!!! warning "ベータ機能"
Realtime エージェントはベータ版です。実装の改善に伴い、非互換の変更が入る可能性があります。

## 概要

Realtime エージェントは、会話のフローを可能にし、音声とテキストの入力をリアルタイムに処理して、リアルタイム音声で応答します。OpenAI の Realtime API への永続的な接続を維持し、低レイテンシで自然な音声対話や、割り込みへのスムーズな対応が可能です。

## アーキテクチャ

### コアコンポーネント

realtime システムはいくつかの主要コンポーネントで構成されます。

-  **RealtimeAgent**: instructions、tools、ハンドオフ で構成された エージェント。
-  **RealtimeRunner**: 構成を管理します。`runner.run()` を呼び出してセッションを取得できます。
-  **RealtimeSession**: 単一の対話セッション。通常は ユーザー が会話を開始するたびに作成し、会話が完了するまで維持します。
-  **RealtimeModel**: 基盤となるモデルのインターフェース（通常は OpenAI の WebSocket 実装）

### セッションフロー

一般的な realtime セッションの流れは次のとおりです。

1. **RealtimeAgent を作成** し、instructions、tools、ハンドオフ を設定します。
2. **RealtimeRunner をセットアップ** し、エージェントと構成オプションを指定します。
3. **セッションを開始** します。`await runner.run()` を使用すると RealtimeSession が返されます。
4. **音声またはテキストのメッセージを送信** します。`send_audio()` または `send_message()` を使用します。
5. **イベントをリッスン** します。セッションを反復処理して、音声出力、文字起こし、ツール呼び出し、ハンドオフ、エラーなどのイベントを受け取ります。
6. **割り込みを処理** します。ユーザー が エージェント の発話にかぶせた場合、現在の音声生成は自動的に停止します。

セッションは会話履歴を保持し、realtime モデルとの永続的な接続を管理します。

## エージェント構成

RealtimeAgent は通常の Agent クラスと同様に動作しますが、いくつか重要な違いがあります。完全な API の詳細は、[`RealtimeAgent`][agents.realtime.agent.RealtimeAgent] の API リファレンスをご覧ください。

通常の エージェント との主な相違点:

-  モデルの選択は エージェント レベルではなく、セッション レベルで構成します。
-  structured outputs のサポートはありません（`outputType` は未サポート）。
-  音声は エージェント ごとに構成できますが、最初の エージェント が発話した後は変更できません。
-  ツール、ハンドオフ、instructions などの他の機能は同様に動作します。

## セッション構成

### モデル設定

セッション構成では、基盤となる realtime モデルの動作を制御できます。モデル名（`gpt-realtime` など）、音声の選択（alloy、echo、fable、onyx、nova、shimmer）、およびサポートされるモダリティ（テキストや音声）を構成できます。音声フォーマットは入力と出力の両方で設定でき、既定は PCM16 です。

### 音声設定

音声設定では、セッションが音声入出力をどのように処理するかを制御します。Whisper などのモデルを使用した入力音声の文字起こし、言語設定、分野特有の用語の精度を高めるための文字起こしプロンプトを構成できます。ターン検出の設定では、音声活動検出のしきい値、無音時間、検出された発話の前後のパディングなど、エージェント がいつ応答を開始・終了すべきかを制御します。

## ツールと関数

### ツールの追加

通常の エージェント と同様に、realtime エージェントは会話中に実行される 関数ツール をサポートします。

```python
from agents import function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Your weather API logic here
    return f"The weather in {city} is sunny, 72°F"

@function_tool
def book_appointment(date: str, time: str, service: str) -> str:
    """Book an appointment."""
    # Your booking logic here
    return f"Appointment booked for {service} on {date} at {time}"

agent = RealtimeAgent(
    name="Assistant",
    instructions="You can help with weather and appointments.",
    tools=[get_weather, book_appointment],
)
```

## ハンドオフ

### ハンドオフの作成

ハンドオフ により、専門化された エージェント 間で会話を引き継ぐことができます。

```python
from agents.realtime import realtime_handoff

# Specialized agents
billing_agent = RealtimeAgent(
    name="Billing Support",
    instructions="You specialize in billing and payment issues.",
)

technical_agent = RealtimeAgent(
    name="Technical Support",
    instructions="You handle technical troubleshooting.",
)

# Main agent with handoffs
main_agent = RealtimeAgent(
    name="Customer Service",
    instructions="You are the main customer service agent. Hand off to specialists when needed.",
    handoffs=[
        realtime_handoff(billing_agent, tool_description="Transfer to billing support"),
        realtime_handoff(technical_agent, tool_description="Transfer to technical support"),
    ]
)
```

## イベント処理

セッションは、セッションオブジェクトを反復処理することでリッスンできるイベントをストリーミングします。イベントには、音声出力チャンク、文字起こし結果、ツールの実行開始と終了、エージェント のハンドオフ、エラーが含まれます。特にハンドリングすべき主要イベントは次のとおりです。

-  **audio**: エージェント の応答からの raw 音声データ
-  **audio_end**: エージェント の発話が完了
-  **audio_interrupted**: ユーザー が エージェント を割り込み
-  **tool_start/tool_end**: ツール実行のライフサイクル
-  **handoff**: エージェント のハンドオフが発生
-  **error**: 処理中にエラーが発生

完全なイベントの詳細は、[`RealtimeSessionEvent`][agents.realtime.events.RealtimeSessionEvent] を参照してください。

## ガードレール

realtime エージェントでサポートされるのは出力ガードレールのみです。これらのガードレールはデバウンスされ、リアルタイム生成中のパフォーマンス問題を避けるために（毎語ではなく）定期的に実行されます。既定のデバウンス長は 100 文字ですが、構成可能です。

ガードレールは `RealtimeAgent` に直接アタッチするか、セッションの `run_config` を通じて指定できます。両方のソースのガードレールが併用されます。

```python
from agents.guardrail import GuardrailFunctionOutput, OutputGuardrail

def sensitive_data_check(context, agent, output):
    return GuardrailFunctionOutput(
        tripwire_triggered="password" in output,
        output_info=None,
    )

agent = RealtimeAgent(
    name="Assistant",
    instructions="...",
    output_guardrails=[OutputGuardrail(guardrail_function=sensitive_data_check)],
)
```

ガードレールがトリガーされると、`guardrail_tripped` イベントが生成され、エージェント の現在の応答を割り込むことがあります。デバウンス動作により、安全性とリアルタイム性能要件のバランスを取ります。テキスト エージェント と異なり、realtime エージェントはガードレールがトリガーされても Exception を送出しません。

## 音声処理

[`session.send_audio(audio_bytes)`][agents.realtime.session.RealtimeSession.send_audio] を使用して音声をセッションに送信するか、[`session.send_message()`][agents.realtime.session.RealtimeSession.send_message] を使用してテキストを送信します。

音声出力については、`audio` イベントをリッスンし、任意の音声ライブラリで音声データを再生してください。ユーザー が エージェント を割り込んだ際に即時に再生を停止し、キューにある音声をクリアするために、`audio_interrupted` イベントも必ずリッスンしてください。

## 直接モデルアクセス

基盤となるモデルにアクセスして、カスタムリスナーを追加したり高度な操作を実行したりできます。

```python
# Add a custom listener to the model
session.model.add_listener(my_custom_listener)
```

これにより、接続を低レベルに制御する必要がある高度なユースケースに向けて、[`RealtimeModel`][agents.realtime.model.RealtimeModel] インターフェースへ直接アクセスできます。

## 例

完全な動作するサンプルは、[examples/realtime ディレクトリ](https://github.com/openai/openai-agents-python/tree/main/examples/realtime) を参照してください。UI コンポーネントの有無それぞれのデモを含みます。