---
search:
  exclude: true
---
# ガイド

このガイドでは、OpenAI Agents SDK の realtime 機能を使って音声対応の AI エージェントを構築する方法を詳しく説明します。

!!! warning "Beta feature"
Realtime エージェントはベータ版です。実装の改善に伴い、破壊的な変更が発生する可能性があります。

## 概要

Realtime エージェントは、会話のフローを可能にし、音声およびテキスト入力をリアルタイムで処理し、realtime 音声で応答します。OpenAI の Realtime API との永続的な接続を維持し、低レイテンシで自然な音声対話と、割り込みへのスムーズな対応を実現します。

## アーキテクチャ

### コアコンポーネント

realtime システムは、いくつかの主要コンポーネントで構成されます。

- **RealtimeAgent**: instructions、tools、ハンドオフで構成されたエージェントです。
- **RealtimeRunner**: 設定を管理します。`runner.run()` を呼び出してセッションを取得できます。
- **RealtimeSession**: 単一の対話セッションです。通常、ユーザーが会話を開始するたびに作成し、会話が終了するまで維持します。
- **RealtimeModel**: 基盤となるモデルのインターフェース（通常は OpenAI の WebSocket 実装）

### セッションフロー

一般的な realtime セッションは次のフローに従います。

1. **RealtimeAgent を作成** し、instructions、tools、ハンドオフを設定します。
2. **RealtimeRunner をセットアップ** し、エージェントと設定オプションを指定します。
3. `await runner.run()` を使って **セッションを開始** し、RealtimeSession を受け取ります。
4. `send_audio()` または `send_message()` を使って **音声またはテキストメッセージを送信** します。
5. セッションを反復処理して **イベントをリッスン** します。イベントには、音声出力、トランスクリプト、ツール呼び出し、ハンドオフ、エラーが含まれます。
6. ユーザーがエージェントの発話にかぶせた場合の **割り込み処理** を行います。これにより現在の音声生成は自動的に停止します。

セッションは会話履歴を保持し、realtime モデルとの永続的な接続を管理します。

## エージェント設定

RealtimeAgent は通常の Agent クラスと同様に動作しますが、いくつか重要な違いがあります。完全な API の詳細は、[`RealtimeAgent`][agents.realtime.agent.RealtimeAgent] の API リファレンスをご覧ください。

通常のエージェントとの主な違い:

- モデルの選択はエージェントレベルではなく、セッションレベルで設定します。
- structured outputs のサポートはありません（`outputType` はサポートされません）。
- ボイスはエージェント単位で設定できますが、最初のエージェントが発話した後は変更できません。
- tools、ハンドオフ、instructions など他の機能は同様に動作します。

## セッション設定

### モデル設定

セッション設定では、基盤となる realtime モデルの動作を制御できます。モデル名（`gpt-realtime` など）、ボイス選択（alloy、echo、fable、onyx、nova、shimmer）、対応モダリティ（テキストおよび/または音声）を設定できます。音声フォーマットは入力と出力の両方で設定でき、既定は PCM16 です。

### 音声設定

音声設定は、セッションが音声入力と出力をどのように扱うかを制御します。Whisper などのモデルを使った入力音声の文字起こし、言語の指定、ドメイン特有の用語の精度を高めるためのトランスクリプションプロンプトを設定できます。ターン検出設定では、エージェントが応答を開始/停止する条件を制御でき、音声活動検出のしきい値、無音時間、検出された発話の前後のパディングを指定できます。

## ツールと関数

### ツールの追加

通常のエージェントと同様に、realtime エージェントは会話中に実行される関数ツールをサポートします。

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

ハンドオフにより、専門特化したエージェント間で会話を引き継ぐことができます。

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

セッションはイベントをストリーミングし、セッションオブジェクトを反復処理してリッスンできます。イベントには、音声出力チャンク、文字起こし結果、ツール実行の開始/終了、エージェントのハンドオフ、エラーが含まれます。特に対応すべき主なイベントは次のとおりです。

- **audio**: エージェントの応答からの raw 音声データ
- **audio_end**: エージェントの発話が終了
- **audio_interrupted**: ユーザーがエージェントを割り込み
- **tool_start/tool_end**: ツール実行のライフサイクル
- **handoff**: エージェントのハンドオフが発生
- **error**: 処理中にエラーが発生

完全なイベントの詳細は、[`RealtimeSessionEvent`][agents.realtime.events.RealtimeSessionEvent] を参照してください。

## ガードレール

realtime エージェントでサポートされるのは出力ガードレールのみです。これらのガードレールはデバウンスされ、リアルタイム生成中のパフォーマンス問題を避けるために（毎語ではなく）一定間隔で実行されます。既定のデバウンス長は 100 文字ですが、変更可能です。

ガードレールは `RealtimeAgent` に直接アタッチするか、セッションの `run_config` で指定できます。両方のソースからのガードレールは併用されます。

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

ガードレールが発火すると、`guardrail_tripped` イベントが生成され、エージェントの現在の応答を中断できます。デバウンス動作により、安全性とリアルタイム性能要件のバランスを取ります。テキストエージェントと異なり、realtime エージェントはガードレールが発火しても Exception を送出しません。

## 音声処理

[`session.send_audio(audio_bytes)`][agents.realtime.session.RealtimeSession.send_audio] を使用して音声をセッションに送信するか、[`session.send_message()`][agents.realtime.session.RealtimeSession.send_message] を使用してテキストを送信します。

音声出力については、`audio` イベントをリッスンし、任意の音声ライブラリで音声データを再生します。ユーザーがエージェントを割り込んだ際に即座に再生を停止し、キュー内の音声をクリアするために、`audio_interrupted` イベントも必ずリッスンしてください。

## 直接モデルアクセス

低レベルのカスタムリスナーを追加したり高度な操作を行うために、基盤となるモデルにアクセスできます。

```python
# Add a custom listener to the model
session.model.add_listener(my_custom_listener)
```

これにより、接続をより低レベルで制御する必要がある高度なユースケース向けに、[`RealtimeModel`][agents.realtime.model.RealtimeModel] インターフェースへ直接アクセスできます。

## コード例

動作する完全なコード例は、[examples/realtime ディレクトリ](https://github.com/openai/openai-agents-python/tree/main/examples/realtime) を参照してください。UI コンポーネントの有無それぞれのデモが含まれています。