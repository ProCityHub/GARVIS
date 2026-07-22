from pathlib import Path

from garvis.capability_broker import ApprovalStore
from garvis.capability_runtime import CapabilityAwareRuntime
from garvis.internet_research import ResearchReport, ResearchSource
from garvis.local_file_access import LocalFileAccessStore


class FakeLocal:
    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root
        self.calls: list[tuple[str, str, str]] = []

    def respond(
        self,
        message: str,
        *,
        external_context: str = "",
        workspace_context: str = "",
    ) -> str:
        self.calls.append((message, external_context, workspace_context))
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


def make_runtime(tmp_path: Path, local: FakeLocal, research: FakeResearcher):
    return CapabilityAwareRuntime(
        local,
        approval_store=ApprovalStore(tmp_path / "broker.db"),
        local_access_store=LocalFileAccessStore(tmp_path / "local.db"),
        researcher=research,
    )


def test_nonresearch_stays_local(tmp_path: Path) -> None:
    local = FakeLocal(tmp_path)
    research = FakeResearcher()
    runtime = make_runtime(tmp_path, local, research)
    assert runtime.respond("Explain drywall finishing") == "sourced answer [S1]"
    assert research.queries == []
    assert local.calls == [("Explain drywall finishing", "", "")]
    runtime.close()


def test_request_then_yes(tmp_path: Path) -> None:
    local = FakeLocal(tmp_path)
    research = FakeResearcher()
    runtime = make_runtime(tmp_path, local, research)
    assert "Approve? [Y/N]" in runtime.respond("What is today's weather in Philadelphia?")
    assert runtime.respond("yes") == "sourced answer [S1]"
    assert research.queries == ["What is today's weather in Philadelphia?"]
    assert "PUBLIC INTERNET RESEARCH CONTEXT" in local.calls[0][1]
    assert local.calls[0][2] == ""
    runtime.close()


def test_inline_authorization_runs_once(tmp_path: Path) -> None:
    local = FakeLocal(tmp_path)
    research = FakeResearcher()
    runtime = make_runtime(tmp_path, local, research)
    result = runtime.respond("GARVIS, you may use the internet to research current drywall prices.")
    assert result == "sourced answer [S1]"
    assert research.queries == ["current drywall prices."]
    assert runtime.approval_store.pending() is None
    runtime.close()


def test_local_file_request_then_yes(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("verified local note", encoding="utf-8")
    local = FakeLocal(tmp_path)
    research = FakeResearcher()
    runtime = make_runtime(tmp_path, local, research)

    request = runtime.respond(f'Read file "{note}"')
    assert "GARVIS requests one-task local file access permission" in request
    assert "Data leaving phone: None" in request

    assert runtime.respond("y") == "sourced answer [S1]"
    assert research.queries == []
    assert "APPROVED READ-ONLY LOCAL FILE EVIDENCE" in local.calls[0][2]
    assert "verified local note" in local.calls[0][2]
    runtime.close()


def test_local_file_denial_reads_nothing(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("do not read", encoding="utf-8")
    local = FakeLocal(tmp_path)
    research = FakeResearcher()
    runtime = make_runtime(tmp_path, local, research)

    assert "Approve? [Y/N]" in runtime.respond(f'Read file "{note}"')
    assert runtime.respond("n") == "Local file access denied. No files were read."
    assert local.calls == []
    runtime.close()


def test_directory_list_returns_without_calling_model(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("do not open me", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    local = FakeLocal(tmp_path)
    research = FakeResearcher()
    runtime = make_runtime(tmp_path, local, research)

    assert "Approve? [Y/N]" in runtime.respond(f'List files in "{tmp_path}"')
    result = runtime.respond("y")

    assert "Read-only top-level listing" in result
    assert "alpha.txt" in result
    assert "folder/" in result
    assert "do not open me" not in result
    assert local.calls == []
    assert research.queries == []
    runtime.close()
