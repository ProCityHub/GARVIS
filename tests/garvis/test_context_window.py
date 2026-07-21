from garvis.context_window import (
    bounded_session_input,
    emergency_current_input,
    is_context_length_error,
    make_bounded_session_input,
)


def test_large_old_item_is_not_replayed() -> None:
    history = [
        {"role": "user", "content": "x" * 20_000},
        {"role": "assistant", "content": "recent answer"},
    ]
    current = [{"role": "user", "content": "current question"}]

    result = bounded_session_input(history, current)

    assert result[-1] == current[0]
    assert any(item.get("content") == "recent answer" for item in result)
    assert not any(item.get("content") == "x" * 20_000 for item in result)


def test_context_errors_are_not_replayed() -> None:
    history = [
        {"role": "user", "content": "Error code 400: context_length_exceeded"},
        {"role": "assistant", "content": "usable context"},
    ]

    result = bounded_session_input(
        history,
        [{"role": "user", "content": "hello"}],
    )

    assert [item["content"] for item in result] == ["usable context", "hello"]


def test_callback_uses_enriched_current_turn() -> None:
    callback = make_bounded_session_input("enriched request")
    result = callback([], [{"role": "user", "content": "stored request"}])

    assert result == [{"role": "user", "content": "enriched request"}]


def test_emergency_input_is_bounded() -> None:
    assert len(emergency_current_input("z" * 50_000)) <= 12_100


def test_context_error_detection() -> None:
    assert is_context_length_error(
        RuntimeError("Your input exceeds the context window of this model")
    )
    assert not is_context_length_error(RuntimeError("network unavailable"))
