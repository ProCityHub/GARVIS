from pathlib import Path

from garvis.capability_broker import ApprovalStore
from garvis.capability_runtime import (
    CapabilityAwareRuntime,
    CapabilityRuntimeConfig,
)
from garvis.internet_research import ResearchReport, ResearchSource
from garvis.local_file_access import LocalFileAccessStore
from garvis.research_governance import govern_research_answer
from garvis.thanos_mode import (
    ThanosAuthorizationStore,
    create_authorization,
    revoke_authorization,
)


class FakeLocal:
    def __init__(self, repository_root: Path, answer: str) -> None:
        self.repository_root = repository_root
        self.answer = answer
        self.calls: list[tuple[str, str, str]] = []

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
        self.queries: list[str] = []

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
                    "Official Python language and library documentation.",
                ),
            ),
            "test-primary",
        )


def build_runtime(
    tmp_path: Path,
    *,
    store: ThanosAuthorizationStore,
    answer: str,
):
    local = FakeLocal(tmp_path, answer)
    researcher = PrimaryResearcher()

    runtime = CapabilityAwareRuntime(
        local,
        approval_store=ApprovalStore(
            tmp_path / "broker.db"
        ),
        local_access_store=LocalFileAccessStore(
            tmp_path / "local.db"
        ),
        researcher=researcher,
        config=CapabilityRuntimeConfig("thanos"),
        thanos_store=store,
    )

    return runtime, local, researcher


def test_runtime_config_preserves_approval_default(monkeypatch) -> None:
    monkeypatch.delenv(
        "GARVIS_NETWORK_MODE",
        raising=False,
    )

    assert (
        CapabilityRuntimeConfig
        .from_environment()
        .network_mode
        == "approval"
    )


def test_runtime_config_accepts_thanos(monkeypatch) -> None:
    monkeypatch.setenv(
        "GARVIS_NETWORK_MODE",
        "thanos",
    )

    assert (
        CapabilityRuntimeConfig
        .from_environment()
        .network_mode
        == "thanos"
    )


def test_active_thanos_researches_without_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "GARVIS_HOME",
        str(tmp_path / "garvis-home"),
    )

    store = ThanosAuthorizationStore(
        tmp_path / "thanos.json"
    )

    store.append(create_authorization())

    runtime, local, researcher = build_runtime(
        tmp_path,
        store=store,
        answer=(
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
    assert "HYPERCUBE_HEARTBEAT" in result
    assert "SNAPSHOT_VALIDATION=PASS" in result
    assert "EVIDENCE_GATE=PASS" in result
    assert "HYPERCUBE_ACCEPTANCE=PASS" in result

    runtime.close()


def test_revoked_thanos_cannot_research(
    tmp_path: Path,
) -> None:
    store = ThanosAuthorizationStore(
        tmp_path / "thanos.json"
    )

    active = store.append(
        create_authorization()
    )

    store.append(
        revoke_authorization(
            active,
            reason="test revocation",
        )
    )

    runtime, local, researcher = build_runtime(
        tmp_path,
        store=store,
        answer="should not run",
    )

    result = runtime.respond(
        "Research current Python documentation"
    )

    assert (
        "standing internet research unavailable"
        in result
    )
    assert researcher.queries == []
    assert local.calls == []

    runtime.close()


def test_hypercube_recalculates_numeric_claim(
    tmp_path: Path,
) -> None:
    report = ResearchReport(
        "research math",
        (
            ResearchSource(
                "Python documentation",
                "https://docs.python.org/3/",
                "docs.python.org",
                "Official documentation.",
                "Official documentation.",
            ),
        ),
        "test-primary",
    )

    answer = (
        "Candidate numerical test [S1]\n"
        'GARVIS_MATH_CLAIMS_JSON=['
        '{"claim_id":"M1",'
        '"expression":"(6 / 10) + (4 / 10)",'
        '"expected":"1",'
        '"tolerance":"1e-12",'
        '"meaning":"normalization test"}]'
    )

    clean, result = govern_research_answer(
        "research mathematics for a hypercube model",
        answer,
        report,
        tmp_path,
        session_id="test",
        garvis_home=tmp_path / "garvis-home",
    )

    assert "GARVIS_MATH_CLAIMS_JSON" not in clean
    assert result["snapshot_validation"] == "PASS"
    assert result["math_verification_status"] == "PASS"
    assert result["hypercube_acceptance"] == "PASS"
    assert result["math_verification"][0]["actual"] == "1"
