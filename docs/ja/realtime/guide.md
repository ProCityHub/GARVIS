---
search:
  exclude: true
---
# ガイド

このガイドでは、OpenAI Agents SDK の realtime 機能を用いて音声対応の AI エージェントを構築する方法を詳しく説明します。

!!! warning "ベータ機能"
リアルタイム エージェントはベータ版です。実装の改善に伴い、互換性のない変更が発生する可能性があります。

## 概要

リアルタイム エージェントは、会話の流れを可能にし、音声とテキストの入力をリアルタイムで処理して、リアルタイム音声で応答します。OpenAI の Realtime API と持続的に接続し、低レイテンシで自然な音声会話を実現し、割り込みにもスムーズに対応します。

## アーキテクチャ

### コアコンポーネント

リアルタイム システムは、いくつかの主要コンポーネントで構成されます。

-   **RealtimeAgent**: instructions、tools、およびハンドオフで構成された エージェント。
-   **RealtimeRunner**: 設定を管理します。`runner.run()` を呼び出してセッションを取得できます。
-   **RealtimeSession**: 1 回の対話セッション。通常、ユーザー が会話を開始するたびに作成し、会話が終了するまで維持します。
-   **RealtimeModel**: 基盤となるモデルのインターフェース（通常は OpenAI の WebSocket 実装）

### セッションフロー

一般的なリアルタイム セッションは、次の流れに従います。

1. **RealtimeAgent を作成** し、instructions、tools、ハンドオフを設定します。
2. **RealtimeRunner をセットアップ** し、エージェントと設定オプションを指定します。
3. `await runner.run()` を使って **セッションを開始** し、RealtimeSession を取得します。
4. `send_audio()` または `send_message()` を使用して **音声またはテキスト メッセージを送信** します。
5. セッションを反復処理して **イベントをリッスン** します。イベントには、音声出力、書き起こし、ツール呼び出し、ハンドオフ、エラーが含まれます。
6. ユーザー が話し始めたら現在の音声生成を自動的に停止する **割り込みへの対応** を行います。

セッションは会話履歴を保持し、リアルタイム モデルとの持続的な接続を管理します。

## エージェント設定

RealtimeAgent は通常の Agent クラスとほぼ同様に動作しますが、いくつか重要な違いがあります。API の詳細は、[`RealtimeAgent`][agents.realtime.agent.RealtimeAgent] の API リファレンスをご覧ください。

通常の エージェント との主な相違点:

-   モデル選択は エージェント レベルではなく、セッション レベルで設定します。
-   structured output のサポートはありません（`outputType` は未対応です）。
-   音声は エージェント ごとに設定できますが、最初の エージェント が話し始めた後は変更できません。
-   tools、ハンドオフ、instructions などの他の機能は同様に動作します。

## セッション設定

### モデル設定

セッション設定では、基盤となるリアルタイム モデルの動作を制御できます。モデル名（`gpt-realtime` など）、ボイス選択（alloy、echo、fable、onyx、nova、shimmer）、サポートするモダリティ（テキストや音声）を設定できます。音声フォーマットは入力・出力の両方で指定でき、既定は PCM16 です。

### 音声設定

音声設定では、セッションが音声の入出力をどのように扱うかを制御します。Whisper などのモデルを使用した入力音声の書き起こし、言語設定、ドメイン固有用語の精度を高めるための書き起こしプロンプトを指定できます。ターン検出の設定では、音声活動検出のしきい値、無音継続時間、検出音声の前後のパディングなどを調整して、エージェント が応答を開始・停止するタイミングを制御します。

## ツールと関数

### ツールの追加

通常の エージェント と同様に、リアルタイム エージェント は会話中に実行される 関数ツール をサポートします。

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

ハンドオフを使うと、専門化した エージェント 間で会話を引き継げます。

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

セッションは、セッション オブジェクトを反復処理することでリッスン可能なイベントをストリーミングします。イベントには、音声出力チャンク、書き起こし結果、ツール実行の開始と終了、エージェントのハンドオフ、エラーが含まれます。特に処理すべき主なイベントは次のとおりです。

-   **audio**: エージェント の応答からの raw 音声データ
-   **audio_end**: エージェント の発話が完了
-   **audio_interrupted**: ユーザー が エージェント に割り込んだ
-   **tool_start/tool_end**: ツール実行のライフサイクル
-   **handoff**: エージェント のハンドオフが発生
-   **error**: 処理中にエラーが発生

イベントの詳細は、[`RealtimeSessionEvent`][agents.realtime.events.RealtimeSessionEvent] を参照してください。

## ガードレール

リアルタイム エージェント でサポートされるのは出力ガードレールのみです。パフォーマンス問題を避けるため、これらのガードレールはデバウンスされ、（毎語ではなく）一定間隔で実行されます。既定のデバウンス長は 100 文字ですが、変更可能です。

ガードレールは `RealtimeAgent` に直接アタッチするか、セッションの `run_config` で指定できます。両方の経路から提供されたガードレールは併せて実行されます。

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

ガードレールがトリガーされると、`guardrail_tripped` イベントが生成され、エージェント の現在の応答を中断することがあります。デバウンス動作により、安全性とリアルタイム性能要件のバランスが取られます。テキスト エージェント と異なり、リアルタイム エージェント はガードレールがトリップしても例外を **発生させません**。

## 音声処理

[`session.send_audio(audio_bytes)`][agents.realtime.session.RealtimeSession.send_audio] を使用して音声を、[`session.send_message()`][agents.realtime.session.RealtimeSession.send_message] を使用してテキストをセッションに送信します。

音声出力については、`audio` イベントをリッスンし、任意の音声ライブラリでデータを再生してください。ユーザー が エージェント に割り込んだ際に即座に再生を停止し、キュー済み音声をクリアするために、`audio_interrupted` イベントも必ずリッスンしてください。

## モデルへの直接アクセス

低レベルのリスナー追加や高度な操作のために、基盤となるモデルへアクセスできます。

```python
# Add a custom listener to the model
session.model.add_listener(my_custom_listener)
```

これにより、接続を低レベルで制御する必要がある高度なユースケース向けに、[`RealtimeModel`][agents.realtime.model.RealtimeModel] インターフェースへ直接アクセスできます。

## コード例

完全な動作サンプルは、[examples/realtime ディレクトリ](https://github.com/openai/openai-agents-python/tree/main/examples/realtime) を参照してください。UI コンポーネントの有無それぞれのデモが含まれています。