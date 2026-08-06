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


def test_file_search_returns_exact_matches_without_calling_model(tmp_path: Path) -> None:
    source = tmp_path / "settings.py"
    source.write_text(
        'ROOTS = os.getenv("GARVIS_LOCAL_ACCESS_ROOTS", "")\n',
        encoding="utf-8",
    )
    local = FakeLocal(tmp_path)
    research = FakeResearcher()
    runtime = make_runtime(tmp_path, local, research)

    request = runtime.respond(f'Search files in "{tmp_path}" for GARVIS_LOCAL_ACCESS_ROOTS')
    assert "Approve? [Y/N]" in request

    result = runtime.respond("y")

    assert "Read-only text matches" in result
    assert "settings.py:1:" in result
    assert "GARVIS_LOCAL_ACCESS_ROOTS" in result
    assert "is set to" not in result
    assert local.calls == []
    assert research.queries == []
    runtime.close()
# GARVIS_18_BRAIN_CAPABILITY_TESTS_V1

from pathlib import Path as _CouncilPath

from garvis.capability_broker import ApprovalStore as _CouncilApprovalStore
from garvis.capability_runtime import (
    CapabilityAwareRuntime as _CouncilRuntime,
    CapabilityRuntimeConfig as _CouncilRuntimeConfig,
)
from garvis.local_file_access import (
    LocalFileAccessStore as _CouncilLocalAccessStore,
)


class _CouncilFakeLocal:
    def __init__(self, root: _CouncilPath) -> None:
        self.repository_root = root
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
        return "local answer"


class _CouncilNeverResearch:
    def __init__(self) -> None:
        self.calls = 0

    def research(self, _query: str):
        self.calls += 1
        raise AssertionError(
            "research must not run when protected consultation fails"
        )


class _CouncilFailingSupervisor:
    def consult(self, _message: str, *, protected_action: bool = False):
        raise RuntimeError("simulated council failure")


class _CouncilPassingReport:
    request_sha256 = "0" * 64
    consultation_available = True
    council_participation_count = 10
    angel_participation_count = 8
    operational_authorization = False


class _CouncilPassingSupervisor:
    def __init__(self) -> None:
        self.calls = []

    def consult(self, message: str, *, protected_action: bool = False):
        self.calls.append((message, protected_action))
        return _CouncilPassingReport()


def _make_council_runtime(
    tmp_path,
    *,
    supervisor,
    researcher=None,
    mode="approval",
):
    local = _CouncilFakeLocal(tmp_path)
    actual_researcher = researcher or _CouncilNeverResearch()

    runtime = _CouncilRuntime(
        local,
        approval_store=_CouncilApprovalStore(
            tmp_path / "council-broker.db"
        ),
        local_access_store=_CouncilLocalAccessStore(
            tmp_path / "council-local.db"
        ),
        researcher=actual_researcher,
        config=_CouncilRuntimeConfig(mode),
        heartbeat_supervisor=supervisor,
        session_id="council-test",
    )

    return runtime, local, actual_researcher


def test_council_failure_is_fail_soft_for_ordinary_local_answer(
    tmp_path,
):
    runtime, local, _researcher = _make_council_runtime(
        tmp_path,
        supervisor=_CouncilFailingSupervisor(),
        mode="off",
    )

    try:
        assert runtime.respond("Explain the architecture.") == "local answer"
        assert local.calls
    finally:
        runtime.close()


def test_council_failure_is_fail_closed_before_research_execution(
    tmp_path,
):
    researcher = _CouncilNeverResearch()
    runtime, _local, _ = _make_council_runtime(
        tmp_path,
        supervisor=_CouncilFailingSupervisor(),
        researcher=researcher,
        mode="approval",
    )

    try:
        request = runtime.respond(
            "Research current Python documentation"
        )
        assert "Approve? [Y/N]" in request

        result = runtime.respond("yes")

        assert "council unavailable" in result.casefold()
        assert "not executed" in result.casefold()
        assert researcher.calls == 0
    finally:
        runtime.close()


def test_supervisor_is_consulted_without_receiving_authority(
    tmp_path,
):
    supervisor = _CouncilPassingSupervisor()
    runtime, _local, _researcher = _make_council_runtime(
        tmp_path,
        supervisor=supervisor,
        mode="off",
    )

    try:
        assert runtime.respond("Give a local answer.") == "local answer"
        assert supervisor.calls == [
            ("Give a local answer.", False)
        ]
        assert runtime.last_council_report.operational_authorization is False
    finally:
        runtime.close()


# GARVIS_18_BRAIN_AUDIT_SECURITY_TESTS_V1

class _SecurityCaptureApprovalStore:
    def __init__(self):
        self.events = []

    def audit(self, event, **kwargs):
        self.events.append((event, kwargs))

    def close(self):
        return None


def test_council_failure_audit_excludes_exception_message(
    tmp_path,
):
    audit_store = _SecurityCaptureApprovalStore()
    secret = "audit-secret-that-must-not-persist-88888"

    local = _CouncilFakeLocal(tmp_path)
    runtime = _CouncilRuntime(
        local,
        approval_store=audit_store,
        local_access_store=_CouncilLocalAccessStore(
            tmp_path / "security-audit-local.db"
        ),
        researcher=_CouncilNeverResearch(),
        config=_CouncilRuntimeConfig("off"),
        heartbeat_supervisor=_CouncilFailingSupervisor(),
        session_id="security-audit-test",
    )

    try:
        allowed = runtime._consult_council(
            f"api_key={secret}",
            protected=True,
        )
    finally:
        runtime.close()

    assert allowed is False

    matching = [
        kwargs["detail"]
        for event, kwargs in audit_store.events
        if event == "council_consultation_failed"
    ]

    assert len(matching) == 1

    detail = matching[0]
    rendered = repr(detail)

    assert detail["error_code"] == (
        "COUNCIL_CONSULTATION_FAILED"
    )
    assert detail["error_type"] == "RuntimeError"
    assert "simulated council failure" not in rendered
    assert secret not in rendered
    assert "error" not in detail
