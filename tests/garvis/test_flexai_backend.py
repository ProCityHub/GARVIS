import ast
import json
from pathlib import Path

import pytest

from garvis.flexai_backend import (
    FLEXAI_CHAT_COMPLETIONS_URL,
    FlexAIConfigurationError,
    FlexAIResponseError,
    FlexAITransportError,
    chat_completion,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def test_missing_key_fails_before_transport():
    called = {"value": False}

    def fake_post(*args, **kwargs):
        called["value"] = True
        raise AssertionError("transport must not execute")

    with pytest.raises(FlexAIConfigurationError, match="FLEXAI_API_KEY"):
        chat_completion(
            "flexai/test-model",
            [{"role": "user", "content": "hello"}],
            env={},
            post=fake_post,
        )

    assert called["value"] is False


def test_explicit_namespace_and_empty_identifier_fail_closed():
    with pytest.raises(FlexAIConfigurationError):
        chat_completion(
            "vendor/test-model",
            [{"role": "user", "content": "hello"}],
            env={"FLEXAI_API_KEY": "test-key"},
            post=lambda *a, **k: None,
        )

    with pytest.raises(FlexAIConfigurationError, match="empty"):
        chat_completion(
            "flexai/",
            [{"role": "user", "content": "hello"}],
            env={"FLEXAI_API_KEY": "test-key"},
            post=lambda *a, **k: None,
        )


def test_request_contract_and_raw_response_preserved():
    secret = "test-flexai-key-not-real"
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "id": "offline-fixture",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "candidate answer",
                        }
                    }
                ],
            }
        )

    result = chat_completion(
        "flexai/example-model",
        [{"role": "user", "content": "test"}],
        env={"FLEXAI_API_KEY": secret},
        timeout=12.5,
        post=fake_post,
    )

    assert captured["url"] == FLEXAI_CHAT_COMPLETIONS_URL
    assert captured["url"] == "https://api.flex.ai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer " + secret
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["json"] == {
        "model": "example-model",
        "messages": [{"role": "user", "content": "test"}],
    }
    assert secret not in json.dumps(captured["json"], sort_keys=True)
    assert captured["timeout"] == 12.5

    assert result.model == "example-model"
    assert result.text == "candidate answer"
    assert result.raw["id"] == "offline-fixture"


def test_http_error_does_not_echo_secret_or_response_body():
    secret = "secret-that-must-not-leak"

    def fake_post(*args, **kwargs):
        return FakeResponse(
            {"error": "provider body should remain unreported"},
            status_code=503,
        )

    with pytest.raises(FlexAITransportError) as excinfo:
        chat_completion(
            "flexai/example-model",
            [{"role": "user", "content": "test"}],
            env={"FLEXAI_API_KEY": secret},
            post=fake_post,
        )

    message = str(excinfo.value)
    assert secret not in message
    assert "provider body should remain unreported" not in message
    assert "503" in message


def test_invalid_provider_json_shape_fails_closed():
    with pytest.raises(FlexAIResponseError):
        chat_completion(
            "flexai/example-model",
            [{"role": "user", "content": "test"}],
            env={"FLEXAI_API_KEY": "test-key"},
            post=lambda *a, **k: FakeResponse({"choices": []}),
        )


def test_backend_imports_no_agent_or_provider_sdk():
    source = Path("src/garvis/flexai_backend.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint(
        {"openai", "agents", "langchain", "langgraph", "flexai"}
    )
