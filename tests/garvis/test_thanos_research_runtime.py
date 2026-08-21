from pathlib import Path

from garvis.capability_broker import ApprovalStore
from garvis.capability_runtime import (
    CapabilityAwareRuntime,
    CapabilityRuntimeConfig,
)
from garvis.creator_authority import CreatorAuthority
from garvis.internet_research import ResearchReport, ResearchSource
from garvis.local_file_access import LocalFileAccessStore


class FakeLocal:
    def __init__(self, repository_root: Path, answer: str) -> None:
        self.repository_root = repository_root
        self.answer = answer
        self.calls = []

    def respond(
        self,
        message: str,
        *,
        external_context: str = "",
        workspace_context: str = "",
    ) -> str:
        self.calls.append(
            (message, external_context, workspace_context)
        )
        return self.answer


class PrimaryResearcher:
    def __init__(self) -> None:
        self.queries = []

    def research(self, query: str) -> ResearchReport:
        self.queries.append(query)
        return ResearchReport(
            query,
            (
                ResearchSource(
                    "Python documentation",
                    "https://docs.python.org/3/",
                    "docs.python.org",
                    "Official Python documentation.",
                    "Official Python documentation.",
                ),
            ),
            "test-primary",
        )


def build_runtime(
    tmp_path: Path,
    authority: CreatorAuthority,
    answer: str,
):
    local = FakeLocal(tmp_path, answer)
    researcher = PrimaryResearcher()
    runtime = CapabilityAwareRuntime(
        local,
        approval_store=ApprovalStore(tmp_path / "broker.db"),
        local_access_store=LocalFileAccessStore(tmp_path / "local.db"),
        researcher=researcher,
        config=CapabilityRuntimeConfig("creator"),
        creator_authority=authority,
    )
    return runtime, local, researcher


def test_runtime_config_defaults_to_creator(monkeypatch) -> None:
    monkeypatch.delenv("GARVIS_NETWORK_MODE", raising=False)
    assert (
        CapabilityRuntimeConfig.from_environment().network_mode
        == "creator"
    )


def test_legacy_thanos_environment_migrates_to_creator(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GARVIS_NETWORK_MODE", "thanos")
    assert (
        CapabilityRuntimeConfig.from_environment().network_mode
        == "creator"
    )


def test_creator_researches_without_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "GARVIS_HOME",
        str(tmp_path / "garvis-home"),
    )
    runtime, local, researcher = build_runtime(
        tmp_path,
        CreatorAuthority(),
        (
            "Research answer [S1]\n"
            "GARVIS_MATH_CLAIMS_JSON=[]"
        ),
    )
    result = runtime.respond(
        "Research current Python documentation"
    )
    assert "Approve? [Y/N]" not in result
    assert researcher.queries == [
        "Research current Python documentation"
    ]
    assert local.calls
    runtime.close()


def test_disabled_creator_authority_cannot_research(
    tmp_path: Path,
) -> None:
    runtime, local, researcher = build_runtime(
        tmp_path,
        CreatorAuthority(enabled=False),
        "should not run",
    )
    result = runtime.respond(
        "Research current Python documentation"
    )
    assert "standing internet research unavailable" in result
    assert researcher.queries == []
    assert local.calls == []
    runtime.close()
