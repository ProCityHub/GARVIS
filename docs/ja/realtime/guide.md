---
search:
  exclude: true
---
# ガイド

このガイドでは、OpenAI Agents SDK のリアルタイム機能を用いて音声対応の AI エージェントを構築する方法を詳しく説明します。

!!! warning "Beta feature"
リアルタイム エージェントはベータ版です。実装の改善に伴い、互換性のない変更が発生する可能性があります。

## 概要

リアルタイム エージェントは、会話のフローを可能にし、音声とテキストの入力をリアルタイムで処理し、リアルタイム音声で応答します。OpenAI の Realtime API との永続的な接続を維持し、低レイテンシで自然な音声対話と割り込みへの優雅な対応を実現します。

## アーキテクチャ

### コアコンポーネント

リアルタイム システムは、いくつかの主要コンポーネントで構成されます。

-   **RealtimeAgent**: instructions、tools、handoffs で構成されたエージェント。
-   **RealtimeRunner**: 構成を管理します。`runner.run()` を呼び出してセッションを取得できます。
-   **RealtimeSession**: 単一の対話セッション。通常、ユーザーが会話を開始するたびに 1 つ作成し、会話が終わるまで維持します。
-   **RealtimeModel**: 基盤となるモデル インターフェース（通常は OpenAI の WebSocket 実装）

### セッションフロー

典型的なリアルタイム セッションは次のフローに従います。

1.  instructions、tools、handoffs を用いて **RealtimeAgent を作成** します。
2.  エージェントと構成オプションで **RealtimeRunner をセットアップ** します。
3.  `await runner.run()` を使用して **セッションを開始** します。これにより RealtimeSession が返されます。
4.  `send_audio()` または `send_message()` を使用して **音声またはテキスト メッセージを送信** します。
5.  セッションをイテレーションして **イベントを受信** します。イベントには音声出力、書き起こし、ツール呼び出し、ハンドオフ、エラーが含まれます。
6.  ユーザーがエージェントの発話にかぶせて話した際の **割り込みを処理** します。これにより現在の音声生成は自動的に停止します。

セッションは会話履歴を維持し、リアルタイム モデルとの永続的な接続を管理します。

## エージェント構成

RealtimeAgent は通常の Agent クラスと同様に動作しますが、いくつかの重要な違いがあります。API の詳細は [`RealtimeAgent`][agents.realtime.agent.RealtimeAgent] の API リファレンスをご覧ください。

通常のエージェントとの主な違い:

-   モデルの選択はエージェント レベルではなくセッション レベルで構成します。
-   structured output はサポートされません（`outputType` は非対応）。
-   音声はエージェントごとに構成できますが、最初のエージェントが話し始めた後は変更できません。
-   ツール、ハンドオフ、instructions などの他の機能は同様に動作します。

## セッション構成

### モデル設定

セッション構成では、基盤となるリアルタイム モデルの動作を制御できます。モデル名（`gpt-realtime` など）、音声の選択（alloy、echo、fable、onyx、nova、shimmer）、対応モダリティ（テキストおよび/または音声）を構成できます。音声フォーマットは入力・出力の両方に設定でき、デフォルトは PCM16 です。

### オーディオ設定

オーディオ設定は、セッションが音声の入出力をどのように処理するかを制御します。Whisper のようなモデルを使った入力音声の文字起こし、言語設定、ドメイン特有の用語の精度向上に役立つトランスクリプション プロンプトを指定できます。発話区切り検出（ターン検出）設定では、エージェントが応答を開始・停止するタイミングを制御し、音声活動検出のしきい値、無音時間、検出された発話の前後パディングなどを設定できます。

## ツールと関数

### ツールの追加

通常のエージェントと同様に、リアルタイム エージェントは会話中に実行される 関数ツール をサポートします。

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

ハンドオフにより、専門化したエージェント間で会話を引き継げます。

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

-   **audio**: エージェントの応答からの生のオーディオデータ
-   **audio_end**: エージェントの発話完了
-   **audio_interrupted**: ユーザーがエージェントを割り込み
-   **tool_start/tool_end**: ツール実行のライフサイクル
-   **handoff**: エージェントのハンドオフが発生
-   **error**: 処理中にエラーが発生

イベントの詳細は [`RealtimeSessionEvent`][agents.realtime.events.RealtimeSessionEvent] を参照してください。

## ガードレール

リアルタイム エージェントでは出力ガードレールのみがサポートされます。パフォーマンス問題を避けるため、これらのガードレールはデバウンスされ、（毎語ではなく）定期的に実行されます。デフォルトのデバウンス長は 100 文字ですが、設定可能です。

ガードレールは `RealtimeAgent` に直接アタッチするか、セッションの `run_config` を介して提供できます。両方のソースのガードレールは併用されます。

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

ガードレールが発火すると、`guardrail_tripped` イベントを生成し、エージェントの現在の応答を割り込むことがあります。デバウンス動作は、安全性とリアルタイム性能要件のバランスを取るのに役立ちます。テキスト エージェントと異なり、リアルタイム エージェントはガードレール発火時に例外を **発生させません**。

## オーディオ処理

[`session.send_audio(audio_bytes)`][agents.realtime.session.RealtimeSession.send_audio] を使用して音声をセッションに送信するか、[`session.send_message()`][agents.realtime.session.RealtimeSession.send_message] を使用してテキストを送信します。

音声出力については、`audio` イベントをリッスンし、任意のオーディオ ライブラリで音声データを再生してください。ユーザーがエージェントを割り込んだ際に即時に再生を停止し、キュー済みの音声をクリアするため、`audio_interrupted` イベントを必ずリッスンしてください。

## モデルへの直接アクセス

基盤となるモデルにアクセスして、カスタム リスナーを追加したり高度な操作を実行できます。

```python
# Add a custom listener to the model
session.model.add_listener(my_custom_listener)
```

これにより、接続をより低レベルで制御する必要がある高度なユースケース向けに、[`RealtimeModel`][agents.realtime.model.RealtimeModel] インターフェースへ直接アクセスできます。

## コード例

完全な動作コードは、UI コンポーネントの有無それぞれのデモを含む [examples/realtime ディレクトリ](https://github.com/openai/openai-agents-python/tree/main/examples/realtime) を参照してください。