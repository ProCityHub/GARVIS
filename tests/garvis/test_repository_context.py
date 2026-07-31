from pathlib import Path

import pytest

from garvis.assistant import GarvisAssistant
from garvis.repository_context import (
    LOCAL_MAX_CONTEXT_CHARS,
    build_query_repository_context,
    build_repository_context,
    ground_message,
    select_repository_files,
    should_ground_repository,
)


class FakeResult:
    final_output = "Grounded answer"


class FakeRunner:
    def __init__(self) -> None:
        self.input = ""

    async def __call__(self, agent, run_input, **kwargs):
        self.input = run_input
        return FakeResult()


def test_grounding_classifier_targets_repository_questions() -> None:
    assert should_ground_repository("Explain the repository architecture")
    assert should_ground_repository("What tests exist?")
    assert not should_ground_repository("What is the weather?")


def test_snapshot_reads_only_allowlisted_files(tmp_path: Path) -> None:
    allowed = tmp_path / "src" / "garvis" / "assistant.py"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("ACTIVE_ASSISTANT")

    secret = tmp_path / "secret.txt"
    secret.write_text("DO_NOT_INCLUDE")

    snapshot = build_repository_context(tmp_path)

    assert "ACTIVE_ASSISTANT" in snapshot
    assert "DO_NOT_INCLUDE" not in snapshot


def test_query_grounding_selects_relevant_files_and_excludes_secrets(tmp_path: Path) -> None:
    runtime = tmp_path / "src" / "garvis" / "local_language_runtime.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("def local_runtime():\n    return 'repository grounding'\n")

    secret = tmp_path / "src" / "garvis" / "credentials.py"
    secret.write_text("PASSWORD = 'never include'\n")

    selected = select_repository_files(tmp_path, "Explain local runtime repository grounding")
    context = build_query_repository_context(
        tmp_path,
        "Explain local runtime repository grounding",
    )

    assert "src/garvis/local_language_runtime.py" in selected
    assert "local_runtime" in context
    assert "never include" not in context
    assert len(context) <= LOCAL_MAX_CONTEXT_CHARS


def test_query_grounding_size_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "src" / "garvis"
    root.mkdir(parents=True)
    for index in range(12):
        (root / f"module_{index}.py").write_text("repository code " * 500)

    first = build_query_repository_context(tmp_path, "repository code")
    second = build_query_repository_context(tmp_path, "repository code")

    assert first == second
    assert len(first) <= LOCAL_MAX_CONTEXT_CHARS


def test_ground_message_labels_repository_evidence(tmp_path: Path) -> None:
    allowed = tmp_path / "src" / "garvis" / "cli.py"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("ACTIVE_CLI")

    grounded = ground_message("Explain the code", tmp_path)

    assert "[GARVIS read-only repository evidence]" in grounded
    assert "ACTIVE_CLI" in grounded
    assert "[User request]" in grounded


@pytest.mark.asyncio
async def test_repository_question_receives_grounded_context(tmp_path: Path) -> None:
    allowed = tmp_path / "src" / "garvis" / "assistant.py"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("VERIFIED_IMPLEMENTATION")

    runner = FakeRunner()
    assistant = GarvisAssistant(
        runner=runner,
        persist_memory=False,
        repository_root=tmp_path,
    )

    reply = await assistant.respond("Explain this repository architecture")

    assert reply.text == "Grounded answer"
    assert "VERIFIED_IMPLEMENTATION" in runner.input
    assert "read-only repository evidence" in runner.input


@pytest.mark.asyncio
async def test_ordinary_question_is_not_given_repository_snapshot(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    assistant = GarvisAssistant(
        runner=runner,
        persist_memory=False,
        repository_root=tmp_path,
    )

    await assistant.respond("Is the heartbeat operating?")

    assert runner.input == "Is the heartbeat operating?"
