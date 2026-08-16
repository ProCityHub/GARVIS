"""Join the local runtime, approval brokers, internet research, and memory."""

from __future__ import annotations

from .heartbeat_supervisor import (
    CouncilAdvisoryReport,
    FullAgentHeartbeatSupervisor,
)

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .capability_broker import (
    ApprovalRequest,
    ApprovalStore,
    appears_to_require_research,
    extract_research_query,
    has_explicit_network_authorization,
)
from .internet_research import InternetResearchClient, ResearchError, ResearchReport
from .research_governance import (
    govern_research_answer,
    render_governance_status,
    research_verification_contract,
)
from .thanos_mode import (
    ThanosAction,
    ThanosAuthorizationStore,
    ThanosError,
    permits,
)
from .local_file_access import (
    LocalAccessError,
    LocalAccessRequest,
    LocalFileAccessStore,
    appears_to_require_local_access,
    execute_local_access,
    parse_local_access_request,
)


class LocalResponder(Protocol):
    repository_root: Path

    def respond(
        self,
        message: str,
        *,
        external_context: str = "",
        workspace_context: str = "",
    ) -> str: ...


class Researcher(Protocol):
    def research(self, query: str) -> ResearchReport: ...


class HeartbeatSupervisor(Protocol):
    def consult(
        self,
        request: str,
        *,
        protected_action: bool = False,
    ) -> CouncilAdvisoryReport: ...


@dataclass(frozen=True)
class CapabilityRuntimeConfig:
    network_mode: str = "approval"

    @classmethod
    def from_environment(cls) -> CapabilityRuntimeConfig:
        mode = os.getenv("GARVIS_NETWORK_MODE", "approval").strip().casefold()
        return cls(mode if mode in {"off", "approval", "thanos"} else "approval")


# GARVIS_18_BRAIN_AUDIT_SECURITY_REPAIR_V1
def _safe_exception_audit_detail(
    exc: BaseException,
    *,
    code: str,
) -> dict[str, str]:
    """Return bounded audit metadata without retaining exception text."""
    return {
        "error_code": code,
        "error_type": type(exc).__name__,
    }


class CapabilityAwareRuntime:
    def __init__(
        self,
        local_runtime: LocalResponder,
        *,
        approval_store: ApprovalStore | None = None,
        local_access_store: LocalFileAccessStore | None = None,
        researcher: Researcher | None = None,
        config: CapabilityRuntimeConfig | None = None,
        thanos_store: ThanosAuthorizationStore | None = None,
        heartbeat_supervisor: HeartbeatSupervisor | None = None,
        session_id: str = "default",
    ) -> None:
        self.local_runtime = local_runtime
        self.approval_store = approval_store or ApprovalStore()
        self.local_access_store = local_access_store or LocalFileAccessStore()
        self.researcher = researcher or InternetResearchClient()
        self.config = config or CapabilityRuntimeConfig.from_environment()
        default_thanos_store = Path(
            os.getenv(
                "GARVIS_THANOS_STORE",
                str(
                    Path.home()
                    / ".garvis"
                    / "thanos"
                    / "authorization.json"
                ),
            )
        ).expanduser()
        self.thanos_store = (
            thanos_store
            or ThanosAuthorizationStore(
                default_thanos_store
            )
        )
        self.session_id = session_id

        repository_root = Path(
            getattr(
                self.local_runtime,
                "repository_root",
                Path.cwd(),
            )
        )
        self.heartbeat_supervisor = (
            heartbeat_supervisor
            or FullAgentHeartbeatSupervisor(repository_root)
        )
        self.last_council_report: CouncilAdvisoryReport | None = None

    def _consult_council(
        self,
        message: str,
        *,
        protected: bool,
    ) -> bool:
        try:
            report = self.heartbeat_supervisor.consult(
                message,
                protected_action=protected,
            )
            self.last_council_report = report

            try:
                self.approval_store.audit(
                    "council_consulted",
                    session_id=self.session_id,
                    detail={
                        "request_sha256": report.request_sha256,
                        "protected": protected,
                        "consultation_available": (
                            report.consultation_available
                        ),
                        "council_participation_count": (
                            report.council_participation_count
                        ),
                        "angel_participation_count": (
                            report.angel_participation_count
                        ),
                        "operational_authorization": False,
                    },
                )
            except Exception:
                pass

            return (
                report.consultation_available
                or not protected
            )
        except Exception as exc:
            try:
                self.approval_store.audit(
                    "council_consultation_failed",
                    session_id=self.session_id,
                    detail={
                        "protected": protected,
                        **_safe_exception_audit_detail(
                            exc,
                            code="COUNCIL_CONSULTATION_FAILED",
                        ),
                    },
                )
            except Exception:
                pass

            return not protected

    def close(self) -> None:
        self.approval_store.close()
        self.local_access_store.close()

    def _thanos_research_authorized(self) -> tuple[bool, str]:
        """Verify standing THANOS authority before internet research."""

        try:
            authorization = self.thanos_store.load()

            if authorization is None:
                return False, "THANOS standing authorization is absent"

            permits(
                authorization,
                ThanosAction.RESEARCH,
            )

        except ThanosError as exc:
            return False, str(exc)

        return True, ""

    def _remember(self, request: str, answer: str, report: ResearchReport) -> None:
        try:
            from .memory_lifecycle import EvidenceStatus, MemoryKind, MemoryStore

            evidence = (
                EvidenceStatus.EVIDENCE_SUPPORTED
                if report.distinct_domains >= 2
                else EvidenceStatus.PROVISIONAL
            )
            urls = " ".join(
                f"[S{index}] {source.url}" for index, source in enumerate(report.sources, 1)
            )
            content = (
                f"Research question: {request} Local synthesis: {answer[:1800]} Sources: {urls}"
            )
            with MemoryStore.from_environment() as store:
                store.remember(
                    content,
                    session_id=self.session_id,
                    kind=MemoryKind.SEMANTIC,
                    evidence_status=evidence,
                    source="internet_research",
                    destination="epistemic_registry",
                    tags=("internet_research", report.provider),
                    salience=0.58,
                    confidence=0.72 if report.distinct_domains >= 2 else 0.48,
                )
        except Exception:
            return

    def _execute_research(
        self,
        request: ApprovalRequest,
        *,
        governed: bool = False,
    ) -> str:
        if not self._consult_council(
            request.original_request,
            protected=True,
        ):
            return (
                "GARVIS council unavailable. "
                "Protected research was not executed."
            )

        self.approval_store.audit(
            "network_research_started",
            session_id=self.session_id,
            request_id=request.request_id,
            detail={
                "query": request.research_query,
                "governed": governed,
            },
        )

        governance = None

        try:
            report = self.researcher.research(
                request.research_query
            )

            model_request = (
                research_verification_contract(
                    request.original_request
                )
                if governed
                else request.original_request
            )

            answer = self.local_runtime.respond(
                model_request,
                external_context=report.render_context(),
            )

            if governed:
                answer, governance = govern_research_answer(
                    request.original_request,
                    answer,
                    report,
                    self.local_runtime.repository_root,
                    session_id=self.session_id,
                )

                answer = (
                    answer
                    + "\n\n"
                    + render_governance_status(
                        governance
                    )
                )

        except ResearchError as exc:
            self.approval_store.audit(
                "network_research_failed",
                session_id=self.session_id,
                request_id=request.request_id,
                detail=_safe_exception_audit_detail(
                    exc,
                    code="RUNTIME_BOUNDARY_EXCEPTION",
                ),
            )

            return "GARVIS research error: " + str(exc)

        except Exception as exc:
            self.approval_store.audit(
                "network_research_failed",
                session_id=self.session_id,
                request_id=request.request_id,
                detail=_safe_exception_audit_detail(
                    exc,
                    code="RUNTIME_BOUNDARY_EXCEPTION",
                ),
            )

            return "GARVIS research failed safely: " + str(exc)

        self._remember(
            request.original_request,
            answer,
            report,
        )

        detail = {
            "provider": report.provider,
            "sources": len(report.sources),
            "distinct_domains": report.distinct_domains,
            "governed": governed,
        }

        if governance is not None:
            detail["hypercube_acceptance"] = (
                governance["hypercube_acceptance"]
            )

        self.approval_store.audit(
            "network_research_completed",
            session_id=self.session_id,
            request_id=request.request_id,
            detail=detail,
        )

        return answer

    def _execute_local_access(self, request: LocalAccessRequest) -> str:
        if not self._consult_council(
            request.original_request,
            protected=True,
        ):
            return (
                "GARVIS council unavailable. "
                "Protected local access was not executed."
            )

        self.local_access_store.audit(
            "local_access_started",
            session_id=self.session_id,
            request_id=request.request_id,
            detail={"target": request.target_path, "operation": request.operation},
        )
        try:
            report = execute_local_access(request, self.local_runtime.repository_root)
            if request.operation == "list":
                answer = f"Read-only top-level listing for {report.target_path}:\n{report.content}"
            elif request.operation == "search":
                answer = (
                    f"Read-only text matches for {request.search_query!r} "
                    f"in {report.target_path}:\n{report.content}"
                )
            else:
                answer = self.local_runtime.respond(
                    request.original_request,
                    workspace_context=report.render_context(),
                )
        except LocalAccessError as exc:
            self.local_access_store.audit(
                "local_access_failed",
                session_id=self.session_id,
                request_id=request.request_id,
                detail=_safe_exception_audit_detail(
                    exc,
                    code="RUNTIME_BOUNDARY_EXCEPTION",
                ),
            )
            return f"GARVIS local access denied safely: {exc}"
        except Exception as exc:
            self.local_access_store.audit(
                "local_access_failed",
                session_id=self.session_id,
                request_id=request.request_id,
                detail=_safe_exception_audit_detail(
                    exc,
                    code="RUNTIME_BOUNDARY_EXCEPTION",
                ),
            )
            return f"GARVIS local access failed safely: {exc}"
        self.local_access_store.audit(
            "local_access_completed",
            session_id=self.session_id,
            request_id=request.request_id,
            detail={"target": request.target_path, "operation": request.operation},
        )
        return answer

    def respond(self, message: str) -> str:
        # GARVIS_18_BRAIN_LIVE_INTEGRATION_V1
        # Ordinary consultation is fail-soft and never grants authority.
        self._consult_council(message, protected=False)

        local_resolution = self.local_access_store.resolve(
            message,
            session_id=self.session_id,
        )
        if local_resolution is not None:
            if not local_resolution.approved:
                return "Local file access denied. No files were read."
            return self._execute_local_access(local_resolution.request)

        network_resolution = self.approval_store.resolve(
            message,
            session_id=self.session_id,
        )
        if network_resolution is not None:
            if not network_resolution.approved:
                return "Network research denied. No internet request was made."
            return self._execute_research(network_resolution.request)

        if appears_to_require_local_access(message):
            try:
                target, operation, search_query = parse_local_access_request(
                    message,
                    self.local_runtime.repository_root,
                )
            except LocalAccessError as exc:
                return f"GARVIS local access request error: {exc}"
            request = self.local_access_store.create(
                message,
                target,
                operation,
                search_query,
                session_id=self.session_id,
            )
            return request.render()

        if not appears_to_require_research(message):
            return self.local_runtime.respond(message)

        if self.config.network_mode == "off":
            return "Internet research is disabled. No network request was made."

        if self.config.network_mode == "thanos":
            allowed, reason = self._thanos_research_authorized()

            if not allowed:
                self.approval_store.audit(
                    "standing_authority_rejected",
                    session_id=self.session_id,
                    detail={"reason": reason},
                )

                return (
                    "GARVIS standing internet research unavailable: "
                    + reason
                )

            request = self.approval_store.create(
                message,
                extract_research_query(message),
                session_id=self.session_id,
            )

            resolution = self.approval_store.resolve(
                "approve",
                session_id=self.session_id,
            )

            if resolution is None:
                return (
                    "GARVIS could not activate standing research authority."
                )

            self.approval_store.audit(
                "standing_authority_used",
                session_id=self.session_id,
                request_id=resolution.request.request_id,
                detail={
                    "action": ThanosAction.RESEARCH.value
                },
            )

            return self._execute_research(
                resolution.request,
                governed=True,
            )

        request = self.approval_store.create(
            message,
            extract_research_query(message),
            session_id=self.session_id,
        )

        if has_explicit_network_authorization(message):
            resolution = self.approval_store.resolve(
                "approve",
                session_id=self.session_id,
            )

            if resolution is None:
                return (
                    "GARVIS could not record the one-time approval safely."
                )

            return self._execute_research(
                resolution.request
            )

        return request.render()
