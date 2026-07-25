---
search:
  exclude: true
---
# トレーシング

[エージェントのトレーシング](../tracing.md)と同様に、音声パイプラインも自動的にトレースされます。

基本的なトレーシング情報については上記のドキュメントをご覧ください。加えて、パイプラインのトレーシングは [`VoicePipelineConfig`][agents.voice.pipeline_config.VoicePipelineConfig] で構成できます。

主なトレーシング関連フィールドは次のとおりです:

-   [`tracing_disabled`][agents.voice.pipeline_config.VoicePipelineConfig.tracing_disabled]: トレーシングを無効にするかどうかを制御します。デフォルトではトレーシングは有効です。
-   [`trace_include_sensitive_data`][agents.voice.pipeline_config.VoicePipelineConfig.trace_include_sensitive_data]: 音声書き起こしなど、機微なデータをトレースに含めるかどうかを制御します。これは音声パイプライン専用であり、Workflow 内部で行われる処理には適用されません。
-   [`trace_include_sensitive_audio_data`][agents.voice.pipeline_config.VoicePipelineConfig.trace_include_sensitive_audio_data]: トレースに音声データを含めるかどうかを制御します。
-   [`workflow_name`][agents.voice.pipeline_config.VoicePipelineConfig.workflow_name]: トレース Workflow の名前です。
-   [`group_id`][agents.voice.pipeline_config.VoicePipelineConfig.group_id]: 複数のトレースを関連付けるためのトレースの `group_id` です。
-   [`trace_metadata`][agents.voice.pipeline_config.VoicePipelineConfig.tracing_disabled]: トレースに含める追加のメタデータです。