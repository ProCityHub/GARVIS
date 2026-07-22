"""Join the local runtime, approval brokers, internet research, and memory."""

from __future__ import annotations

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


@dataclass(frozen=True)
class CapabilityRuntimeConfig:
    network_mode: str = "approval"

    @classmethod
    def from_environment(cls) -> CapabilityRuntimeConfig:
        mode = os.getenv("GARVIS_NETWORK_MODE", "approval").strip().casefold()
        return cls(mode if mode in {"off", "approval"} else "approval")


class CapabilityAwareRuntime:
    def __init__(
        self,
        local_runtime: LocalResponder,
        *,
        approval_store: ApprovalStore | None = None,
        local_access_store: LocalFileAccessStore | None = None,
        researcher: Researcher | None = None,
        config: CapabilityRuntimeConfig | None = None,
        session_id: str = "default",
    ) -> None:
        self.local_runtime = local_runtime
        self.approval_store = approval_store or ApprovalStore()
        self.local_access_store = local_access_store or LocalFileAccessStore()
        self.researcher = researcher or InternetResearchClient()
        self.config = config or CapabilityRuntimeConfig.from_environment()
        self.session_id = session_id

    def close(self) -> None:
        self.approval_store.close()
        self.local_access_store.close()

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

    def _execute_research(self, request: ApprovalRequest) -> str:
        self.approval_store.audit(
            "network_research_started",
            session_id=self.session_id,
            request_id=request.request_id,
            detail={"query": request.research_query},
        )
        try:
            report = self.researcher.research(request.research_query)
            answer = self.local_runtime.respond(
                request.original_request,
                external_context=report.render_context(),
            )
        except ResearchError as exc:
            self.approval_store.audit(
                "network_research_failed",
                session_id=self.session_id,
                request_id=request.request_id,
                detail={"error": str(exc)},
            )
            return f"GARVIS research error: {exc}"
        except Exception as exc:
            self.approval_store.audit(
                "network_research_failed",
                session_id=self.session_id,
                request_id=request.request_id,
                detail={"error": str(exc)},
            )
            return f"GARVIS research failed safely: {exc}"
        self._remember(request.original_request, answer, report)
        self.approval_store.audit(
            "network_research_completed",
            session_id=self.session_id,
            request_id=request.request_id,
            detail={
                "provider": report.provider,
                "sources": len(report.sources),
                "distinct_domains": report.distinct_domains,
            },
        )
        return answer

    def _execute_local_access(self, request: LocalAccessRequest) -> str:
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
                detail={"error": str(exc)},
            )
            return f"GARVIS local access denied safely: {exc}"
        except Exception as exc:
            self.local_access_store.audit(
                "local_access_failed",
                session_id=self.session_id,
                request_id=request.request_id,
                detail={"error": str(exc)},
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

        request = self.approval_store.create(
            message,
            extract_research_query(message),
            session_id=self.session_id,
        )
        if has_explicit_network_authorization(message):
            resolution = self.approval_store.resolve("approve", session_id=self.session_id)
            if resolution is None:
                return "GARVIS could not record the one-time approval safely."
            return self._execute_research(resolution.request)
        return request.render()
