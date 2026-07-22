from pathlib import Path

from garvis.capability_broker import ApprovalStore
from garvis.capability_runtime import CapabilityAwareRuntime
from garvis.internet_research import ResearchReport, ResearchSource


class FakeLocal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def respond(self, message: str, *, external_context: str = "") -> str:
        self.calls.append((message, external_context))
        return "sourced answer [S1]"


class FakeResearcher:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def research(self, query: str) -> ResearchReport:
        self.queries.append(query)
        return ResearchReport(
            query,
            (ResearchSource("Source", "https://example.com", "example.com", "evidence"),),
            "test",
        )


def test_nonresearch_stays_local(tmp_path: Path) -> None:
    local = FakeLocal()
    research = FakeResearcher()
    runtime = CapabilityAwareRuntime(
        local,
        approval_store=ApprovalStore(tmp_path / "broker.db"),
        researcher=research,
    )
    assert runtime.respond("Explain drywall finishing") == "sourced answer [S1]"
    assert research.queries == []
    assert local.calls == [("Explain drywall finishing", "")]
    runtime.close()


def test_request_then_yes(tmp_path: Path) -> None:
    local = FakeLocal()
    research = FakeResearcher()
    runtime = CapabilityAwareRuntime(
        local,
        approval_store=ApprovalStore(tmp_path / "broker.db"),
        researcher=research,
    )
    assert "Approve? [Y/N]" in runtime.respond("What is today's weather in Philadelphia?")
    assert runtime.respond("yes") == "sourced answer [S1]"
    assert research.queries == ["What is today's weather in Philadelphia?"]
    assert "PUBLIC INTERNET RESEARCH CONTEXT" in local.calls[0][1]
    runtime.close()


def test_inline_authorization_runs_once(tmp_path: Path) -> None:
    local = FakeLocal()
    research = FakeResearcher()
    runtime = CapabilityAwareRuntime(
        local,
        approval_store=ApprovalStore(tmp_path / "broker.db"),
        researcher=research,
    )
    result = runtime.respond("GARVIS, you may use the internet to research current drywall prices.")
    assert result == "sourced answer [S1]"
    assert research.queries == ["current drywall prices."]
    assert runtime.approval_store.pending() is None
    runtime.close()
