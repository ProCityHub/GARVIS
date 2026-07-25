---
search:
  exclude: true
---
# ガイド

このガイドでは、 OpenAI Agents SDK の realtime 機能を用いて音声対応の AI エージェントを構築する方法を詳しく説明します。

!!! warning "ベータ機能"
Realtime エージェントはベータ版です。実装の改善に伴い、破壊的な変更が発生する可能性があります。

## 概要

Realtime エージェントは、会話フローを可能にし、音声およびテキスト入力をリアルタイムに処理し、リアルタイム音声で応答します。 OpenAI の Realtime API との永続接続を維持し、低遅延で自然な音声会話や、割り込みへのスムーズな対応を実現します。

## アーキテクチャ

### コアコンポーネント

realtime システムは、いくつかの主要コンポーネントで構成されます。

-   **RealtimeAgent**: instructions、tools、handoffs で構成されたエージェント。
-   **RealtimeRunner**: 設定を管理します。`runner.run()` を呼び出してセッションを取得できます。
-   **RealtimeSession**: 単一の対話セッション。通常、ユーザーが会話を開始するたびに 1 つ作成し、会話が完了するまで保持します。
-   **RealtimeModel**: 基盤となるモデルインターフェース（通常は OpenAI の WebSocket 実装）

### セッションフロー

一般的な realtime セッションは次のフローに従います。

1. **RealtimeAgent を作成する** — instructions、tools、handoffs を設定します。
2. **RealtimeRunner を設定する** — エージェントと構成オプションを指定します。
3. **セッションを開始する** — `await runner.run()` を使用し、`RealtimeSession` が返されます。
4. **音声またはテキストメッセージを送信する** — `send_audio()` または `send_message()` を使用します。
5. **イベントをリッスンする** — セッションを反復処理して、音声出力、書き起こし、ツール呼び出し、ハンドオフ、エラーなどのイベントを受け取ります。
6. **割り込みに対応する** — ユーザーがエージェントの発話に被せた場合、現在の音声生成は自動で停止します。

セッションは会話履歴を保持し、realtime モデルとの永続接続を管理します。

## エージェントの設定

RealtimeAgent は通常の Agent クラスと同様に動作しますが、いくつかの重要な相違点があります。 API の詳細は [`RealtimeAgent`][agents.realtime.agent.RealtimeAgent] のリファレンスをご覧ください。

通常のエージェントとの主な違い:

-   モデルの選択はエージェントレベルではなくセッションレベルで設定します。
-   structured outputs はサポートされません（`outputType` は未対応）。
-   音声はエージェントごとに設定できますが、最初のエージェントが話し始めた後は変更できません。
-   tools、handoffs、instructions などその他の機能は同様に動作します。

## セッションの設定

### モデル設定

セッション設定では、基盤となる realtime モデルの動作を制御できます。モデル名（例: `gpt-realtime`）、音声の選択（alloy、echo、fable、onyx、nova、shimmer）、サポートするモダリティ（テキストおよび/または音声）を構成できます。音声フォーマットは入力と出力の両方に設定でき、デフォルトは PCM16 です。

### 音声設定

音声設定では、セッションが音声入力と出力をどのように扱うかを制御します。 Whisper などのモデルを用いた入力音声の書き起こし、言語設定、ドメイン固有用語の精度を高めるための書き起こしプロンプトを設定できます。発話区間検出（ターン検出）の設定では、エージェントが応答を開始・終了すべきタイミングを制御し、音声活動検出のしきい値、無音の長さ、検出された発話の前後パディングなどを指定できます。

## ツールと関数

### ツールの追加

通常のエージェントと同様に、realtime エージェントは会話中に実行される 関数ツール をサポートします。

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

ハンドオフにより、専門化されたエージェント間で会話を移譲できます。

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

セッションはイベントをストリーミングし、セッションオブジェクトを反復処理することでリッスンできます。イベントには、音声出力チャンク、書き起こし結果、ツール実行の開始・終了、エージェントのハンドオフ、エラーが含まれます。重要なイベントは次のとおりです。

-   **audio**: エージェントの応答からの raw 音声データ
-   **audio_end**: エージェントの発話が終了
-   **audio_interrupted**: ユーザーがエージェントを割り込み
-   **tool_start/tool_end**: ツール実行のライフサイクル
-   **handoff**: エージェントのハンドオフが発生
-   **error**: 処理中にエラーが発生

イベントの詳細は [`RealtimeSessionEvent`][agents.realtime.events.RealtimeSessionEvent] を参照してください。

## ガードレール

realtime エージェントでサポートされるのは出力 ガードレール のみです。これらのガードレールはデバウンスされ、リアルタイム生成時のパフォーマンス低下を避けるために（毎語ではなく）一定間隔で実行されます。デフォルトのデバウンス長は 100 文字ですが、変更可能です。

ガードレールは `RealtimeAgent` に直接アタッチするか、セッションの `run_config` を通じて提供できます。両方のソースのガードレールは併用されます。

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

ガードレールがトリガーされると、`guardrail_tripped` イベントが生成され、エージェントの現在の応答を割り込むことがあります。デバウンス動作により、安全性とリアルタイム性能要件のバランスを取ります。テキストエージェントと異なり、realtime エージェントはガードレールが作動しても例外をスローしません。

## 音声処理

[`session.send_audio(audio_bytes)`][agents.realtime.session.RealtimeSession.send_audio] で音声を、[`session.send_message()`][agents.realtime.session.RealtimeSession.send_message] でテキストをセッションに送信します。

音声出力については `audio` イベントをリッスンし、任意の音声ライブラリで音声データを再生してください。ユーザーがエージェントを割り込んだ際にすぐ再生を停止し、キューにある音声をクリアするため、`audio_interrupted` イベントも必ずリッスンしてください。

## 直接的なモデルアクセス

基盤となるモデルにアクセスし、カスタムリスナーの追加や高度な操作を実行できます。

```python
# Add a custom listener to the model
session.model.add_listener(my_custom_listener)
```

これにより、接続をより低レベルで制御する必要がある高度なユースケース向けに、[`RealtimeModel`][agents.realtime.model.RealtimeModel] インターフェースへ直接アクセスできます。

## コード例

完全に動作するコード例は、[examples/realtime ディレクトリ](https://github.com/openai/openai-agents-python/tree/main/examples/realtime) を参照してください。 UI コンポーネントの有無それぞれのデモを含みます。