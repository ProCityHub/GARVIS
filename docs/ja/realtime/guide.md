---
search:
  exclude: true
---
# ガイド

このガイドでは、OpenAI Agents SDK のリアルタイム機能を用いた音声対応 AI エージェントの構築方法を詳しく説明します。

!!! warning "Beta feature"
Realtime エージェントはベータ版です。実装の改善に伴い、互換性に影響する変更が発生する可能性があります。

## 概要

Realtime エージェントは、会話フローを可能にし、音声およびテキスト入力をリアルタイムで処理し、リアルタイム音声で応答します。OpenAI の Realtime API と永続接続を維持し、低レイテンシで自然な音声会話と、割り込みへの優雅な対応を実現します。

## アーキテクチャ

### コアコンポーネント

リアルタイムシステムは、いくつかの主要コンポーネントで構成されます。

-   **RealtimeAgent**: instructions、tools、handoffs を設定したエージェント。
-   **RealtimeRunner**: 設定を管理します。`runner.run()` を呼び出してセッションを取得できます。
-   **RealtimeSession**: 単一の対話セッション。通常、ユーザーが会話を開始するたびに 1 つ作成し、会話が終了するまで維持します。
-   **RealtimeModel**: 基盤となるモデルインターフェース（通常は OpenAI の WebSocket 実装）

### セッションフロー

一般的なリアルタイムセッションは、次のフローに従います。

1. instructions、tools、handoffs を用いて **RealtimeAgent を作成** します。
2. エージェントと設定オプションで **RealtimeRunner をセットアップ** します。
3. `await runner.run()` を使用して **セッションを開始** し、RealtimeSession が返されます。
4. `send_audio()` または `send_message()` を使って **音声またはテキストメッセージを送信** します。
5. セッションを反復処理して **イベントを受信** します。イベントには音声出力、トランスクリプト、ツール呼び出し、ハンドオフ、エラーが含まれます。
6. ユーザーがエージェントの発話に割り込んだ場合の **割り込み処理** を行います。現在の音声生成は自動的に停止します。

セッションは会話履歴を保持し、リアルタイムモデルとの永続接続を管理します。

## エージェント設定

RealtimeAgent は、通常の Agent クラスとほぼ同様に動作しますが、一部に重要な違いがあります。完全な API の詳細は、[`RealtimeAgent`][agents.realtime.agent.RealtimeAgent] の API リファレンスを参照してください。

通常のエージェントとの主な違い:

-   モデルの選択はエージェントレベルではなく、セッションレベルで設定します。
-   structured outputs のサポートはありません（`outputType` は未対応）。
-   声質（voice）はエージェントごとに設定できますが、最初のエージェントが発話した後は変更できません。
-   その他の機能（tools、handoffs、instructions）は同様に動作します。

## セッション設定

### モデル設定

セッション設定では、基盤となるリアルタイムモデルの動作を制御できます。モデル名（`gpt-realtime` など）、声質の選択（alloy、echo、fable、onyx、nova、shimmer）、および対応モダリティ（テキストや音声）を設定できます。音声フォーマットは入力・出力の双方で設定でき、既定は PCM16 です。

### 音声設定

音声設定では、セッションが音声の入出力をどのように扱うかを制御します。Whisper などのモデルを使用した入力音声の書き起こし、言語設定、ドメイン固有用語の精度向上のためのトランスクリプションプロンプトを設定できます。ターン検出設定では、エージェントがいつ応答を開始・停止するかを制御でき、音声活動検出のしきい値、無音の長さ、検出された発話の前後パディングなどを指定できます。

## ツールと関数

### ツールの追加

通常のエージェントと同様に、リアルタイムエージェントは会話中に実行される 関数ツール をサポートします。

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

ハンドオフにより、特化したエージェント間で会話を引き継げます。

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

セッションは、セッションオブジェクトを反復処理することでリッスン可能なイベントをストリームします。イベントには、音声出力チャンク、書き起こし結果、ツール実行の開始と終了、エージェントのハンドオフ、エラーが含まれます。特に処理すべき主要イベントは次のとおりです。

-   **audio**: エージェントの応答からの raw な音声データ
-   **audio_end**: エージェントの発話が完了
-   **audio_interrupted**: ユーザーがエージェントを割り込み
-   **tool_start/tool_end**: ツール実行のライフサイクル
-   **handoff**: エージェント間のハンドオフが発生
-   **error**: 処理中にエラーが発生

イベントの詳細は、[`RealtimeSessionEvent`][agents.realtime.events.RealtimeSessionEvent] を参照してください。

## ガードレール

Realtime エージェントでサポートされるのは出力ガードレールのみです。パフォーマンス問題を避けるため、これらのガードレールはデバウンスされ、（各単語ごとではなく）定期的に実行されます。既定のデバウンス長は 100 文字ですが、設定可能です。

ガードレールは `RealtimeAgent` に直接アタッチするか、セッションの `run_config` 経由で指定できます。両方の経路から提供されたガードレールは併せて実行されます。

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

ガードレールがトリガーされると、`guardrail_tripped` イベントが生成され、エージェントの現在の応答を中断する場合があります。デバウンス動作により、安全性とリアルタイム性能要件のバランスを取ります。テキストエージェントと異なり、Realtime エージェントはガードレールがトリップしても **Exception** を送出しません。

## 音声処理

音声は [`session.send_audio(audio_bytes)`][agents.realtime.session.RealtimeSession.send_audio] を使用してセッションに送信し、テキストは [`session.send_message()`][agents.realtime.session.RealtimeSession.send_message] を使用して送信します。

音声出力については、`audio` イベントをリッスンし、任意のオーディオライブラリで再生してください。ユーザーがエージェントを割り込んだ際に即座に再生を停止し、キュー済みの音声をクリアするため、`audio_interrupted` イベントも必ずリッスンしてください。

## モデルへの直接アクセス

基盤となるモデルにアクセスして、カスタムリスナーの追加や高度な操作を実行できます。

```python
# Add a custom listener to the model
session.model.add_listener(my_custom_listener)
```

これにより、接続を低レベルで制御する必要がある高度なユースケース向けに、[`RealtimeModel`][agents.realtime.model.RealtimeModel] インターフェースへ直接アクセスできます。

## コード例

動作する完全なコード例は、UI コンポーネントの有無それぞれのデモを含む [examples/realtime ディレクトリ](https://github.com/openai/openai-agents-python/tree/main/examples/realtime) を参照してください。