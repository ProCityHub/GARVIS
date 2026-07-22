---
search:
  exclude: true
---
# ガイド

このガイドでは、OpenAI Agents SDK のリアルタイム機能を用いて音声対応の AI エージェントを構築する方法を詳しく説明します。

!!! warning "ベータ機能"
リアルタイム エージェントはベータ版です。実装の改善に伴い、互換性が壊れる変更が発生する可能性があります。

## 概要

リアルタイム エージェントは、会話フローを可能にし、音声およびテキスト入力をリアルタイムで処理し、リアルタイム音声で応答します。OpenAI の Realtime API との永続的な接続を維持し、低レイテンシで自然な音声対話と、割り込みへのスムーズな対応を実現します。

## アーキテクチャ

### コアコンポーネント

リアルタイム システムは、以下の主要コンポーネントで構成されます。

- **RealtimeAgent**: instructions、tools、handoffs を設定したエージェント
- **RealtimeRunner**: 構成を管理します。`runner.run()` を呼び出してセッションを取得できます。
- **RealtimeSession**: 1 回の対話セッション。通常、ユーザーが会話を開始するたびに作成し、会話が終了するまで維持します。
- **RealtimeModel**: 基盤となるモデル インターフェース（通常は OpenAI の WebSocket 実装）

### セッションフロー

一般的なリアルタイム セッションは、次のフローに従います。

1. instructions、tools、handoffs を使って **RealtimeAgent を作成** します。
2. エージェントと構成オプションで **RealtimeRunner を設定** します。
3. `await runner.run()` を使用して **セッションを開始** します。これにより RealtimeSession が返されます。
4. `send_audio()` または `send_message()` を使用して **音声またはテキスト メッセージを送信** します。
5. セッションを反復処理して **イベントをリッスン** します。イベントには音声出力、文字起こし、ツール呼び出し、ハンドオフ、エラーが含まれます。
6. ユーザーがエージェントに被せて話したときの **割り込みを処理** します。これにより現在の音声生成は自動的に停止します。

セッションは会話履歴を保持し、リアルタイム モデルとの永続接続を管理します。

## エージェント構成

RealtimeAgent は、通常の Agent クラスと同様に動作しますが、いくつかの重要な相違点があります。完全な API の詳細は、[`RealtimeAgent`][agents.realtime.agent.RealtimeAgent] の API リファレンスをご覧ください。

通常のエージェントとの主な違い:

- モデルの選択はエージェント レベルではなく、セッション レベルで構成します。
- structured outputs はサポートされません（`outputType` は非対応）。
- 音声はエージェントごとに設定できますが、最初のエージェントが話し始めた後は変更できません。
- ツール、ハンドオフ、instructions などの他の機能は同様に機能します。

## セッション構成

### モデル設定

セッション構成では、基盤となるリアルタイム モデルの動作を制御できます。モデル名（`gpt-realtime` など）、音声の選択（alloy、echo、fable、onyx、nova、shimmer）、および対応するモダリティ（テキストや音声）を設定できます。音声の入出力フォーマットは設定可能で、デフォルトは PCM16 です。

### オーディオ構成

オーディオ設定は、セッションが音声の入出力をどのように扱うかを制御します。Whisper などのモデルを使用した入力音声の文字起こし、言語設定、専門用語の精度を高めるための文字起こしプロンプトを構成できます。ターン検出の設定では、エージェントが応答を開始・終了すべきタイミングを制御し、音声活動検出のしきい値、無音時間、検出された音声の前後のパディングなどを指定できます。

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

ハンドオフにより、専門特化したエージェント間で会話を移譲できます。

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

セッションは、セッション オブジェクトを反復処理することでリッスンできるイベントをストリーム配信します。イベントには、音声出力チャンク、文字起こし結果、ツール実行の開始と終了、エージェントのハンドオフ、エラーなどが含まれます。特に処理すべき主なイベントは次のとおりです。

- **audio**: エージェントの応答からの raw 音声データ
- **audio_end**: エージェントが話し終えた
- **audio_interrupted**: ユーザーがエージェントを割り込んだ
- **tool_start/tool_end**: ツール実行のライフサイクル
- **handoff**: エージェントのハンドオフが発生
- **error**: 処理中にエラーが発生

完全なイベントの詳細は、[`RealtimeSessionEvent`][agents.realtime.events.RealtimeSessionEvent] を参照してください。

## ガードレール

出力 ガードレール のみがリアルタイム エージェントでサポートされています。これらのガードレールはデバウンスされ、パフォーマンス上の問題を避けるため、リアルタイム生成中に毎語ではなく定期的に実行されます。デフォルトのデバウンス長は 100 文字ですが、設定可能です。

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

ガードレールがトリガーされると、`guardrail_tripped` イベントが生成され、エージェントの現在の応答を中断することがあります。デバウンスの挙動により、安全性とリアルタイム性能要件のバランスを取ります。テキスト エージェントと異なり、リアルタイム エージェントはガードレールが作動しても **Exception** を発生させません。

## オーディオ処理

[`session.send_audio(audio_bytes)`][agents.realtime.session.RealtimeSession.send_audio] を使用して音声をセッションに送信するか、[`session.send_message()`][agents.realtime.session.RealtimeSession.send_message] を使用してテキストを送信します。

音声出力については、`audio` イベントをリッスンし、任意のオーディオ ライブラリで音声データを再生します。ユーザーがエージェントを割り込んだ際にすぐに再生を停止し、キューにある音声をクリアするため、`audio_interrupted` イベントも必ずリッスンしてください。

## モデルへの直接アクセス

基盤となるモデルにアクセスして、カスタム リスナーを追加したり、高度な操作を実行したりできます。

```python
# Add a custom listener to the model
session.model.add_listener(my_custom_listener)
```

これにより、接続を低レベルで制御する必要がある高度なユースケース向けに、[`RealtimeModel`][agents.realtime.model.RealtimeModel] インターフェースへ直接アクセスできます。

## コード例

完全に動作するコード例は、UI コンポーネントあり／なしのデモを含む [examples/realtime ディレクトリ](https://github.com/openai/openai-agents-python/tree/main/examples/realtime) をご覧ください。