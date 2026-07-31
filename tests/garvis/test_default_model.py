import pytest

from garvis.assistant import DEFAULT_MODEL, GarvisAssistant


def test_default_model_is_verified_api_model() -> None:
    assert DEFAULT_MODEL == "gpt-5.1"


def test_assistant_uses_default_model_when_unspecified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GARVIS_MODEL", raising=False)

    assistant = GarvisAssistant(persist_memory=False)
    assert assistant.model == DEFAULT_MODEL
