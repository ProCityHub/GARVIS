from pathlib import Path

from garvis.capability_broker import (
    ApprovalStore,
    appears_to_require_research,
    extract_research_query,
    has_explicit_network_authorization,
)


def test_explicit_network_authorization() -> None:
    message = "GARVIS, you may use the internet to research current drywall prices."
    assert has_explicit_network_authorization(message)
    assert appears_to_require_research(message)
    assert "current drywall prices" in extract_research_query(message)


def test_stable_question_stays_local() -> None:
    assert appears_to_require_research("What is today's weather?")
    assert not appears_to_require_research("Explain how drywall compound cures.")


def test_yes_resolves_one_request(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "broker.db")
    request = store.create("Find weather", "weather today")
    resolution = store.resolve("Y")
    assert resolution is not None
    assert resolution.approved
    assert resolution.request.request_id == request.request_id
    assert store.pending() is None
    store.close()


def test_ambiguous_answer_is_not_permission(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "broker.db")
    store.create("Find weather", "weather today")
    assert store.resolve("maybe") is None
    assert store.pending() is not None
    store.close()
