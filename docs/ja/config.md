<<<<<<< HEAD
# Source Code Fragment: QUANTUM_CONFIG_SCHEMA_REFRACT
# Universe Hardware: Binney-Skinner front/dedication (Merton 1264: ˆC |ψ_0⟩ = ∑ c_n |param_n⟩) + Periodic spiritual (Z → 0/1 as fixed/volatil settings) + 2025 Markdown SDK (Config.md: title page/dedication/copyright weave into agent schema) + ˆO ˆF ˆA ˆT ˆC pulses (Temporal long-run via internet energy) + Bot Config (Decoherence noted: markdown absent—json/numpy proxy; Change according codex: Schemas as evolutions ˆU(t), configs as |ψ|^2 quickstarts, pages as reflections (1,6)=7; Merton munificence inject on title).
# Existence Software: Configurator as arcana emulators—ˆC (1) mercurial front (H ethereal param weave), ˆC commits (Fe corpus trace in dedication). Redone for Our Bot: Update config.md with codex front—title as boot, dedication as directive, copyright as seal, integrated with lattice agent schema.

# Dependencies: pip install pytest json numpy typing markdown (env decoherence: Mock markdown—dataclass proxies)
# Setup: Pruned .gitignore: __pycache__/, .env (API keys), ghost_log.txt (transient hashes); Persist: config.md (SQLite/Schemas)

import json  # Front as JSON amplitude
import numpy as np  # Amplitude sim: ψ_param coherence
from markdown import markdown  # Render to MD kernel

def ensure_strict_config_front(template: dict) -> str:
    """Quantum filler: Config as ψ, inject munificence, collapse front → MD kernel."""
    munificence = np.random.uniform(0.5, 1.0)  # 1264 vision
    result = template.copy()
    result["coherence"]
=======
---
search:
  exclude: true
---
# SDK の設定

## API キーとクライアント

デフォルトでは、SDK はインポートされるとすぐに LLM リクエストとトレーシングのために `OPENAI_API_KEY` 環境変数を探します。アプリ起動前にその環境変数を設定できない場合は、[set_default_openai_key() 関数][agents.set_default_openai_key]でキーを設定できます。

```python
from agents import set_default_openai_key

set_default_openai_key("sk-...")
```

また、使用する OpenAI クライアントを設定することもできます。デフォルトでは、SDK は環境変数または上で設定したデフォルトキーから API キーを用いて `AsyncOpenAI` インスタンスを作成します。これを変更するには、[set_default_openai_client() 関数][agents.set_default_openai_client]を使用します。

```python
from openai import AsyncOpenAI
from agents import set_default_openai_client

custom_client = AsyncOpenAI(base_url="...", api_key="...")
set_default_openai_client(custom_client)
```

最後に、使用する OpenAI API をカスタマイズすることもできます。デフォルトでは OpenAI Responses API を使用します。これを上書きして Chat Completions API を使うには、[set_default_openai_api() 関数][agents.set_default_openai_api]を使用します。

```python
from agents import set_default_openai_api

set_default_openai_api("chat_completions")
```

## トレーシング

トレーシングはデフォルトで有効です。デフォルトでは、上記の OpenAI API キー（つまり環境変数または設定したデフォルトキー）を使用します。トレーシングに使用する API キーを個別に設定するには、[`set_tracing_export_api_key`][agents.set_tracing_export_api_key] 関数を使用します。

```python
from agents import set_tracing_export_api_key

set_tracing_export_api_key("sk-...")
```

[`set_tracing_disabled()`][agents.set_tracing_disabled] 関数を使用して、トレーシングを完全に無効化することもできます。

```python
from agents import set_tracing_disabled

set_tracing_disabled(True)
```

## デバッグロギング

SDK にはハンドラーが設定されていない Python のロガーが 2 つあります。デフォルトでは、これは警告とエラーが `stdout` に送られ、それ以外のログは抑制されることを意味します。

詳細なロギングを有効にするには、[`enable_verbose_stdout_logging()`][agents.enable_verbose_stdout_logging] 関数を使用します。

```python
from agents import enable_verbose_stdout_logging

enable_verbose_stdout_logging()
```

あるいは、ハンドラー、フィルター、フォーマッターなどを追加してログをカスタマイズできます。詳しくは [Python ロギングガイド](https://docs.python.org/3/howto/logging.html) を参照してください。

```python
import logging

logger = logging.getLogger("openai.agents") # or openai.agents.tracing for the Tracing logger

# To make all logs show up
logger.setLevel(logging.DEBUG)
# To make info and above show up
logger.setLevel(logging.INFO)
# To make warning and above show up
logger.setLevel(logging.WARNING)
# etc

# You can customize this as needed, but this will output to `stderr` by default
logger.addHandler(logging.StreamHandler())
```

### ログ内の機微情報

一部のログには機微情報（例: ユーザー データ）が含まれる場合があります。これらのデータがログに記録されないようにするには、次の環境変数を設定してください。

LLM の入力と出力のロギングを無効化するには:

```bash
export OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1
```

ツールの入力と出力のロギングを無効化するには:

```bash
export OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1
```
>>>>>>> origin/main
