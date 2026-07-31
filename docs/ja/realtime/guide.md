---
search:
  exclude: true
---
# ガイド

<<<<<<< HEAD
このガイドでは、OpenAI Agents SDK のリアルタイム機能を用いて音声対応の AI エージェントを構築する方法を詳しく説明します。

!!! warning "ベータ機能"
リアルタイム エージェントはベータ版です。実装の改善に伴い、互換性が壊れる変更が発生する可能性があります。

## 概要

リアルタイム エージェントは、会話フローを可能にし、音声およびテキスト入力をリアルタイムで処理し、リアルタイム音声で応答します。OpenAI の Realtime API との永続的な接続を維持し、低レイテンシで自然な音声対話と、割り込みへのスムーズな対応を実現します。
=======
<<<<<<< HEAD
このガイドでは、 OpenAI Agents SDK のリアルタイム機能を用いて、音声対応の AI エージェントを構築する方法を詳しく説明します。

!!! warning "Beta feature"
Realtime エージェントはベータ版です。実装の改善に伴い、互換性が壊れる変更が発生する可能性があります。

## 概要

Realtime エージェントは、会話型のフローを可能にし、音声およびテキスト入力をリアルタイムに処理し、リアルタイム音声で応答します。 OpenAI の Realtime API と永続的な接続を維持し、低レイテンシで自然な音声対話と、割り込みへのスムーズな対応を実現します。
=======
このガイドでは、OpenAI Agents SDK の realtime 機能を用いて音声対応の AI エージェントを構築する方法を詳しく説明します。

!!! warning "ベータ機能"
Realtime エージェントはベータ版です。実装の改善に伴い、破壊的変更が発生する可能性があります。

## 概要

Realtime エージェントは、会話のフローを可能にし、音声とテキストの入力をリアルタイムに処理し、realtime 音声で応答します。OpenAI の Realtime API との永続接続を維持し、低遅延で自然な音声対話と、割り込みへの柔軟な対応を実現します。
>>>>>>> origin/main
>>>>>>> origin/main

## アーキテクチャ

### 中核コンポーネント

<<<<<<< HEAD
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
=======
<<<<<<< HEAD
リアルタイムシステムはいくつかの主要コンポーネントで構成されます。

-   **RealtimeAgent**: instructions、tools、ハンドオフで構成されたエージェントです。
-   **RealtimeRunner**: 設定を管理します。`runner.run()` を呼び出してセッションを取得できます。
-   **RealtimeSession**: 単一の対話セッションです。通常、ユーザーが会話を開始するたびに作成し、会話が終了するまで維持します。
-   **RealtimeModel**: 基盤となるモデルインターフェースです（通常は OpenAI の WebSocket 実装）。

### セッションフロー

一般的なリアルタイムセッションは次のフローに従います。

1. **RealtimeAgent を作成** し、instructions、tools、ハンドオフを設定します。
2. **RealtimeRunner をセットアップ** し、エージェントと設定オプションを渡します。
3. **セッションを開始** し、`await runner.run()` を使用して RealtimeSession を取得します。
4. **音声またはテキストメッセージを送信** し、`send_audio()` または `send_message()` を使用します。
5. **イベントをリッスン** し、セッションを反復処理して受け取ります。イベントには、音声出力、書き起こし、ツール呼び出し、ハンドオフ、エラーが含まれます。
6. **割り込みへの対応**。ユーザーがエージェントの発話に重ねて話した場合、現在の音声生成が自動的に停止します。

セッションは会話履歴を保持し、リアルタイムモデルとの永続的な接続を管理します。
=======
realtime システムは、次の主要コンポーネントで構成されます。
>>>>>>> origin/main

-   **RealtimeAgent**: instructions、tools、handoffs で構成されたエージェント。
-   **RealtimeRunner**: 設定を管理します。`runner.run()` を呼び出してセッションを取得できます。
-   **RealtimeSession**: 単一の対話セッション。通常、ユーザーが会話を開始するたびに作成し、会話が終了するまで維持します。
-   **RealtimeModel**: 基盤となるモデルのインターフェース（通常は OpenAI の WebSocket 実装）

<<<<<<< HEAD
RealtimeAgent は、通常の Agent クラスと同様に動作しますが、いくつかの重要な相違点があります。完全な API の詳細は、[`RealtimeAgent`][agents.realtime.agent.RealtimeAgent] の API リファレンスをご覧ください。

通常のエージェントとの主な違い:

- モデルの選択はエージェント レベルではなく、セッション レベルで構成します。
- structured outputs はサポートされません（`outputType` は非対応）。
- 音声はエージェントごとに設定できますが、最初のエージェントが話し始めた後は変更できません。
- ツール、ハンドオフ、instructions などの他の機能は同様に機能します。
=======
### セッションフロー

一般的な realtime セッションは次のフローに従います。

1. **RealtimeAgent を作成** し、instructions、tools、handoffs を設定します。
2. **RealtimeRunner をセットアップ** し、エージェントと設定オプションを渡します。
3. **セッションを開始** し、`await runner.run()` を使用して RealtimeSession を取得します。
4. **音声またはテキストのメッセージを送信** し、`send_audio()` または `send_message()` を使用します。
5. **イベントをリッスン** し、セッションを反復処理して受け取ります。イベントには、音声出力、文字起こし、ツール呼び出し、ハンドオフ、エラーが含まれます。
6. **割り込みを処理** します。ユーザーがエージェントの発話に被せた場合、現在の音声生成は自動的に停止します。
>>>>>>> origin/main

セッションは会話履歴を保持し、realtime モデルとの永続接続を管理します。
>>>>>>> origin/main

## エージェント設定

<<<<<<< HEAD
セッション構成では、基盤となるリアルタイム モデルの動作を制御できます。モデル名（`gpt-realtime` など）、音声の選択（alloy、echo、fable、onyx、nova、shimmer）、および対応するモダリティ（テキストや音声）を設定できます。音声の入出力フォーマットは設定可能で、デフォルトは PCM16 です。
=======
<<<<<<< HEAD
RealtimeAgent は通常の Agent クラスと同様に動作しますが、いくつかの重要な違いがあります。完全な API の詳細は、[`RealtimeAgent`][agents.realtime.agent.RealtimeAgent] の API リファレンスをご覧ください。

通常のエージェントとの主な違い:

-   モデルの選択はエージェントレベルではなく、セッションレベルで設定します。
-   structured output はサポートされません（`outputType` は使用できません）。
-   音声はエージェントごとに設定できますが、最初のエージェントが話し始めた後は変更できません。
-   ツール、ハンドオフ、instructions など、その他の機能は同じように動作します。
=======
RealtimeAgent は通常の Agent クラスと同様に動作しますが、いくつか重要な違いがあります。API の詳細は、[`RealtimeAgent`][agents.realtime.agent.RealtimeAgent] の API リファレンスをご覧ください。

通常のエージェントとの主な相違点:

-   モデルの選択はエージェント レベルではなく、セッション レベルで構成します。
-   structured output のサポートはありません（`outputType` はサポートされません）。
-   音声はエージェントごとに設定できますが、最初のエージェントが発話した後は変更できません。
-   その他の機能（tools、handoffs、instructions）は同様に動作します。
>>>>>>> origin/main

## セッション設定

### マデル設定

<<<<<<< HEAD
セッション設定により、基盤となるリアルタイムモデルの動作を制御できます。モデル名（`gpt-realtime` など）、ボイスの選択（alloy、echo、fable、onyx、nova、shimmer）、対応モダリティ（テキストおよび/または音声）を設定できます。音声フォーマットは入力と出力の両方に対して設定でき、デフォルトは PCM16 です。
>>>>>>> origin/main

### オーディオ構成

<<<<<<< HEAD
オーディオ設定は、セッションが音声の入出力をどのように扱うかを制御します。Whisper などのモデルを使用した入力音声の文字起こし、言語設定、専門用語の精度を高めるための文字起こしプロンプトを構成できます。ターン検出の設定では、エージェントが応答を開始・終了すべきタイミングを制御し、音声活動検出のしきい値、無音時間、検出された音声の前後のパディングなどを指定できます。
=======
音声設定は、セッションが音声入力と出力をどのように扱うかを制御します。 Whisper などのモデルを使用して入力音声の書き起こしを設定し、言語設定を指定し、ドメイン固有用語の精度を高めるために書き起こしプロンプトを提供できます。ターン検出の設定により、エージェントがいつ応答を開始・終了するかを制御でき、音声活動検出のしきい値、無音時間、検出された発話の前後パディングなどを指定できます。
=======
セッション構成では、基盤となる realtime モデルの動作を制御できます。モデル名（`gpt-realtime` など）、音声の選択（alloy、echo、fable、onyx、nova、shimmer）、および対応するモダリティ（テキストや音声）を設定できます。音声の入出力それぞれのフォーマットを設定でき、既定は PCM16 です。

### 音声設定

音声設定では、セッションが音声の入出力をどう扱うかを制御します。Whisper のようなモデルを用いた入力音声の文字起こし、言語設定、ドメイン特有の用語の精度を高めるための文字起こしプロンプトを設定できます。ターン検出設定では、エージェントがいつ応答を開始・終了すべきかを制御し、音声活動検出のしきい値、無音時間、検出音声の前後パディングなどを調整できます。
>>>>>>> origin/main
>>>>>>> origin/main

## ツールと関数

### ツールの追加

<<<<<<< HEAD
通常のエージェントと同様に、リアルタイム エージェントは会話中に実行される 関数ツール をサポートします。
=======
<<<<<<< HEAD
通常のエージェントと同様に、リアルタイムエージェントは会話中に実行される関数ツールをサポートします。
=======
通常のエージェントと同様に、realtime エージェントは会話中に実行される 関数ツール をサポートします。
>>>>>>> origin/main
>>>>>>> origin/main

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

<<<<<<< HEAD
ハンドオフにより、専門特化したエージェント間で会話を移譲できます。
=======
ハンドオフにより、特化したエージェント間で会話を引き継ぐことができます。
>>>>>>> origin/main

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

<<<<<<< HEAD
セッションは、セッション オブジェクトを反復処理することでリッスンできるイベントをストリーム配信します。イベントには、音声出力チャンク、文字起こし結果、ツール実行の開始と終了、エージェントのハンドオフ、エラーなどが含まれます。特に処理すべき主なイベントは次のとおりです。

- **audio**: エージェントの応答からの raw 音声データ
- **audio_end**: エージェントが話し終えた
- **audio_interrupted**: ユーザーがエージェントを割り込んだ
- **tool_start/tool_end**: ツール実行のライフサイクル
- **handoff**: エージェントのハンドオフが発生
- **error**: 処理中にエラーが発生
=======
<<<<<<< HEAD
セッションはイベントをストリーミングし、セッションオブジェクトを反復処理することでリッスンできます。イベントには、音声出力チャンク、書き起こし結果、ツール実行の開始と終了、エージェントのハンドオフ、エラーが含まれます。主に対応すべきイベントは以下のとおりです。

-   **audio**: エージェントの応答からの音声データ
-   **audio_end**: エージェントの発話が完了
=======
セッションはイベントをストリーミングし、セッションオブジェクトを反復処理することでリッスンできます。イベントには、音声出力チャンク、文字起こし結果、ツール実行の開始と終了、エージェントのハンドオフ、エラーが含まれます。主に扱うべきイベントは次のとおりです。
>>>>>>> origin/main

-   **audio**: エージェントの応答からの raw 音声データ
-   **audio_end**: エージェントの発話が終了
>>>>>>> origin/main
-   **audio_interrupted**: ユーザーがエージェントを割り込み
-   **tool_start/tool_end**: ツール実行のライフサイクル
-   **handoff**: エージェントのハンドオフが発生
-   **error**: 処理中にエラーが発生

完全なイベント詳細は [`RealtimeSessionEvent`][agents.realtime.events.RealtimeSessionEvent] を参照してください。

## ガードレール

<<<<<<< HEAD
出力 ガードレール のみがリアルタイム エージェントでサポートされています。これらのガードレールはデバウンスされ、パフォーマンス上の問題を避けるため、リアルタイム生成中に毎語ではなく定期的に実行されます。デフォルトのデバウンス長は 100 文字ですが、設定可能です。

ガードレールは `RealtimeAgent` に直接アタッチするか、セッションの `run_config` を通じて提供できます。両方のソースのガードレールは併用されます。
=======
Realtime エージェントでは出力ガードレールのみがサポートされます。これらのガードレールはデバウンスされ、リアルタイム生成中のパフォーマンス問題を避けるために（毎語ではなく）定期的に実行されます。デフォルトのデバウンス長は 100 文字ですが、設定可能です。

<<<<<<< HEAD
ガードレールは `RealtimeAgent` に直接アタッチするか、セッションの `run_config` を通じて提供できます。両方のソースからのガードレールは併用されて実行されます。
=======
ガードレールは `RealtimeAgent` に直接アタッチするか、セッションの `run_config` を通じて提供できます。両方のソースからのガードレールは併用して実行されます。
>>>>>>> origin/main
>>>>>>> origin/main

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

<<<<<<< HEAD
ガードレールがトリガーされると、`guardrail_tripped` イベントが生成され、エージェントの現在の応答を中断することがあります。デバウンスの挙動により、安全性とリアルタイム性能要件のバランスを取ります。テキスト エージェントと異なり、リアルタイム エージェントはガードレールが作動しても **Exception** を発生させません。
=======
<<<<<<< HEAD
ガードレールがトリガーされると、`guardrail_tripped` イベントが生成され、エージェントの現在の応答を割り込む場合があります。デバウンス動作は、安全性とリアルタイム性能要件のバランスを取るのに役立ちます。テキストエージェントと異なり、リアルタイムエージェントはガードレールがトリップしても Exception を発生させません。
=======
ガードレールがトリガーされると、`guardrail_tripped` イベントが生成され、エージェントの現在の応答を中断できる場合があります。デバウンス動作により、安全性とリアルタイム性能要件のバランスが取られます。テキスト エージェントと異なり、realtime エージェントはガードレールがトリップしても Exception を送出しません。
>>>>>>> origin/main
>>>>>>> origin/main

## オーディオ処理

[`session.send_audio(audio_bytes)`][agents.realtime.session.RealtimeSession.send_audio] を使用して音声をセッションに送信するか、[`session.send_message()`][agents.realtime.session.RealtimeSession.send_message] を使用してテキストを送信します。

<<<<<<< HEAD
音声出力については、`audio` イベントをリッスンし、任意のオーディオ ライブラリで音声データを再生します。ユーザーがエージェントを割り込んだ際にすぐに再生を停止し、キューにある音声をクリアするため、`audio_interrupted` イベントも必ずリッスンしてください。

## モデルへの直接アクセス

基盤となるモデルにアクセスして、カスタム リスナーを追加したり、高度な操作を実行したりできます。
=======
<<<<<<< HEAD
音声出力については、`audio` イベントをリッスンし、任意の音声ライブラリで音声データを再生してください。ユーザーがエージェントを割り込んだ際にすぐ再生を停止し、キューにある音声をクリアするため、`audio_interrupted` イベントも必ずリッスンしてください。
=======
音声出力については、`audio` イベントをリッスンし、任意の音声ライブラリで再生してください。ユーザーがエージェントを割り込んだ際に即座に再生を停止し、キューされた音声をクリアするため、`audio_interrupted` イベントも必ずリッスンしてください。
>>>>>>> origin/main

## モデルへの直接アクセス

<<<<<<< HEAD
基盤となるモデルにアクセスし、カスタムリスナーの追加や高度な操作を実行できます。
=======
基盤となるモデルにアクセスして、カスタム リスナーの追加や高度な操作を行えます。
>>>>>>> origin/main
>>>>>>> origin/main

```python
# Add a custom listener to the model
session.model.add_listener(my_custom_listener)
```

<<<<<<< HEAD
これにより、接続を低レベルで制御する必要がある高度なユースケース向けに、[`RealtimeModel`][agents.realtime.model.RealtimeModel] インターフェースへ直接アクセスできます。

## コード例

完全に動作するコード例は、UI コンポーネントあり／なしのデモを含む [examples/realtime ディレクトリ](https://github.com/openai/openai-agents-python/tree/main/examples/realtime) をご覧ください。
=======
<<<<<<< HEAD
これにより、接続の低レベル制御が必要な高度なユースケースに向けて、[`RealtimeModel`][agents.realtime.model.RealtimeModel] インターフェースへ直接アクセスできます。
=======
これは、接続を低レベルで制御する必要がある高度なユースケース向けに、[`RealtimeModel`][agents.realtime.model.RealtimeModel] インターフェースへの直接アクセスを提供します。
>>>>>>> origin/main

## コード例

<<<<<<< HEAD
動作する完全なコード例は、[examples/realtime ディレクトリ](https://github.com/openai/openai-agents-python/tree/main/examples/realtime) を参照してください。 UI コンポーネントの有無それぞれのデモが含まれています。
=======
完全な動作コード例は、UI コンポーネントの有無それぞれのデモを含む [examples/realtime ディレクトリ](https://github.com/openai/openai-agents-python/tree/main/examples/realtime) をご覧ください。
>>>>>>> origin/main
>>>>>>> origin/main
