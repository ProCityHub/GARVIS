from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from garvis.resilient_runtime import (
    ResilientGarvisRuntime,
    split_wrapped_prompt,
)


class FakeLedger:
    def __init__(self) -> None:
        self.turns: list[dict[str, str]] = []
        self.events: list[str] = []

    def append(self, role: str, content: str) -> None:
        self.events.append(f"append:{role}")
        self.turns.append({"role": role, "content": content})


class CapturingModel:
    def __init__(self, reply: str = "test reply") -> None:
        self.reply = reply
        self.messages: list[dict[str, str]] = []

    def __call__(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        del client, model
        self.messages = list(messages)
        return self.reply


def fake_build_context(
    system_prompt: str,
    ledger: FakeLedger,
) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, *ledger.turns[-30:]]


def test_split_wrapped_prompt_removes_repeated_wrapper() -> None:
    context, message = split_wrapped_prompt(
        "constitutional controls\n\n"
        "ADRIEN'S CURRENT MESSAGE:\n"
        "remember this line"
    )

    assert context == "constitutional controls"
    assert message == "remember this line"


@pytest.mark.asyncio
async def test_user_is_persisted_before_model_and_reply_before_return() -> None:
    ledger = FakeLedger()
    model = CapturingModel("persisted reply")

    runtime = ResilientGarvisRuntime(
        model="test-model",
        session_name="test-session",
        repository_root=Path.cwd(),
        client=object(),
        ledger=ledger,
        build_messages=fake_build_context,
        call_model=model,
    )

    reply = await runtime.respond("hello")

    assert reply.text == "persisted reply"
    assert ledger.events == ["append:user", "append:assistant"]
    assert ledger.turns == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "persisted reply"},
    ]


@pytest.mark.asyncio
async def test_wrapper_context_is_not_written_into_ledger() -> None:
    ledger = FakeLedger()
    model = CapturingModel()

    runtime = ResilientGarvisRuntime(
        model="test-model",
        session_name="test-session",
        repository_root=Path.cwd(),
        client=object(),
        ledger=ledger,
        build_messages=fake_build_context,
        call_model=model,
    )

    await runtime.respond(
        "large constitutional wrapper\n\n"
        "ADRIEN'S CURRENT MESSAGE:\n"
        "small current message"
    )

    assert ledger.turns[0] == {
        "role": "user",
        "content": "small current message",
    }
    assert "large constitutional wrapper" in model.messages[0]["content"]
    assert "large constitutional wrapper" not in str(ledger.turns)


@pytest.mark.asyncio
async def test_failed_model_call_preserves_user_input() -> None:
    ledger = FakeLedger()

    def failing_model(
        client: Any,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        del client, model, messages
        raise RuntimeError("simulated 429")

    runtime = ResilientGarvisRuntime(
        model="test-model",
        session_name="test-session",
        repository_root=Path.cwd(),
        client=object(),
        ledger=ledger,
        build_messages=fake_build_context,
        call_model=failing_model,
    )

    with pytest.raises(RuntimeError, match="simulated 429"):
        await runtime.respond("survive the crash")

    assert ledger.turns == [
        {"role": "user", "content": "survive the crash"},
    ]


# RESILIENT REMOTE CANONICAL RESEARCH MEMORY TESTS

@pytest.mark.asyncio
async def test_resilient_research_memory_is_transient_system_context() -> None:
    sentinel = "RESILIENT_RESEARCH_MEMORY_SENTINEL_817"

    ledger = FakeLedger()
    model = CapturingModel("research answer")
    provider_calls: list[tuple[str, str]] = []

    def memory_provider(
        query: str,
        session_id: str,
    ) -> str:
        provider_calls.append((query, session_id))
        return sentinel

    runtime = ResilientGarvisRuntime(
        model="test-model",
        session_name="research-session",
        repository_root=Path.cwd(),
        client=object(),
        ledger=ledger,
        build_messages=fake_build_context,
        call_model=model,
        research_memory_provider=memory_provider,
    )

    prompt = "Research current Python release changes"

    reply = await runtime.respond(prompt)

    assert reply.text == "research answer"

    assert provider_calls == [
        (prompt, "research-session")
    ]

    assert ledger.turns == [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "research answer"},
    ]

    assert sentinel not in str(ledger.turns)

    assert sentinel in model.messages[0]["content"]
    assert "ADVISORY CONTEXT ONLY" in (
        model.messages[0]["content"]
    )

    user_messages = [
        item["content"]
        for item in model.messages
        if item.get("role") == "user"
    ]

    assert user_messages[-1] == prompt
    assert sentinel not in user_messages[-1]


@pytest.mark.asyncio
async def test_resilient_memory_failure_blocks_model_inference() -> None:
    ledger = FakeLedger()
    model = CapturingModel("must not execute")

    def unavailable_memory(
        query: str,
        session_id: str,
    ) -> str:
        del query, session_id
        raise RuntimeError("memory unavailable")

    runtime = ResilientGarvisRuntime(
        model="test-model",
        session_name="research-session",
        repository_root=Path.cwd(),
        client=object(),
        ledger=ledger,
        build_messages=fake_build_context,
        call_model=model,
        research_memory_provider=unavailable_memory,
    )

    prompt = "Research current lattice evidence"

    with pytest.raises(
        RuntimeError,
        match="mandatory research memory unavailable",
    ):
        await runtime.respond(prompt)

    assert model.messages == []

    assert ledger.turns == [
        {"role": "user", "content": prompt}
    ]


@pytest.mark.asyncio
async def test_resilient_nonresearch_does_not_consult_research_memory() -> None:
    ledger = FakeLedger()
    model = CapturingModel("ordinary answer")
    provider_calls: list[str] = []

    def memory_provider(
        query: str,
        session_id: str,
    ) -> str:
        provider_calls.append(query + session_id)
        return "should not be used"

    runtime = ResilientGarvisRuntime(
        model="test-model",
        session_name="ordinary-session",
        repository_root=Path.cwd(),
        client=object(),
        ledger=ledger,
        build_messages=fake_build_context,
        call_model=model,
        research_memory_provider=memory_provider,
    )

    await runtime.respond(
        "Explain how drywall compound cures"
    )

    assert provider_calls == []



@pytest.mark.asyncio
async def test_resilient_recent_change_phrase_requires_research_memory() -> None:
    ledger = FakeLedger()
    model = CapturingModel("research answer")
    calls: list[str] = []

    def provider(query: str, session_id: str) -> str:
        calls.append(query)
        return "ADVISORY_MEMORY_CONTEXT"

    runtime = ResilientGarvisRuntime(
        model="test-model",
        session_name="classifier-regression",
        repository_root=Path.cwd(),
        client=object(),
        ledger=ledger,
        build_messages=fake_build_context,
        call_model=model,
        research_memory_provider=provider,
    )

    prompt = "What changed recently in OpenAI API documentation?"

    await runtime.respond(prompt)

    assert calls == [prompt]
