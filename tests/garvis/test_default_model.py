from garvis.assistant import DEFAULT_MODEL, GarvisAssistant


def test_default_model_is_verified_api_model() -> None:
    assert DEFAULT_MODEL == "gpt-4.1-mini"


def test_assistant_uses_default_model_when_unspecified() -> None:
    assistant = GarvisAssistant(persist_memory=False)
    assert assistant.model == DEFAULT_MODEL
