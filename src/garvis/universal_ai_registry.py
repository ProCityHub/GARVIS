"""GARVIS Universal AI Router V1 prototype.

Creator / conceptual architect: Adrien D. Thomas

This module is intentionally non-executing:
- it does not call model providers;
- it does not read secret values into reports;
- it does not control Android apps;
- it does not open sockets or start servers.

It inventories candidate AI organs, normalizes provider identity, records
capability metadata, and produces Hypercube perspective scheduling plans.
Provider output remains candidate information until GARVIS verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple


class CandidateType(str, Enum):
    GARVIS_LOCAL = "garvis_local"
    REMOTE_API = "remote_api"
    ANDROID_APP = "android_app"
    LOCAL_RUNTIME = "local_runtime"
    LOCAL_SERVER = "local_server"
    GARVIS_ORGAN = "garvis_organ"


class AdapterKind(str, Enum):
    GARVIS_LOCAL = "garvis_local"
    OPENAI_NATIVE = "openai_native"
    ANTHROPIC_NATIVE = "anthropic_native"
    OPENAI_COMPATIBLE = "openai_compatible"
    MANUAL_ONLY = "manual_only"
    MISSING = "missing"
    UNKNOWN = "unknown"
    INTERNAL = "internal"


class Authority(str, Enum):
    GARVIS = "garvis"
    CANDIDATE_ONLY = "candidate_only"
    EVIDENCE_ONLY = "evidence_only"


@dataclass(frozen=True)
class ProviderIdentity:
    provider_id: str
    adapter: AdapterKind
    required_env_names: Tuple[str, ...] = ()
    required_base_url_env: Optional[str] = None
    adapter_supported: bool = True


@dataclass(frozen=True)
class AIOrgan:
    organ_id: str
    candidate_type: CandidateType
    provider_id: str
    model: Optional[str]
    adapter: AdapterKind
    configured: bool
    programmable: bool
    adapter_supported: bool
    declared_capabilities: FrozenSet[str] = field(default_factory=frozenset)
    verified_capabilities: FrozenSet[str] = field(default_factory=frozenset)
    authority: Authority = Authority.CANDIDATE_ONLY
    notes: Tuple[str, ...] = ()

    def supports(self, capability: str, *, verified_only: bool = False) -> bool:
        pool = self.verified_capabilities if verified_only else self.declared_capabilities
        return capability in pool


@dataclass(frozen=True)
class PerspectiveAssignment:
    code: str
    name: str
    owner: str
    reason: str


PERSPECTIVES: Tuple[Tuple[str, str], ...] = (
    ("000", "Literal"),
    ("001", "Context"),
    ("010", "Intent"),
    ("011", "Relation"),
    ("100", "Evidence"),
    ("101", "Possibility"),
    ("110", "Consequence"),
    ("111", "Integration"),
)


def identify_provider(model: str) -> ProviderIdentity:
    """Resolve a model string using the currently observed GARVIS routing families."""
    selected = (model or "").strip()
    lowered = selected.casefold()

    if lowered in {"local", "garvis-local", "garvis/local"}:
        return ProviderIdentity(
            provider_id="garvis",
            adapter=AdapterKind.GARVIS_LOCAL,
            adapter_supported=True,
        )

    if lowered.startswith("anthropic/") or lowered.startswith("claude-") or lowered.startswith("claude/"):
        return ProviderIdentity(
            provider_id="anthropic",
            adapter=AdapterKind.ANTHROPIC_NATIVE,
            required_env_names=("ANTHROPIC_API_KEY",),
        )

    if lowered.startswith("openrouter/"):
        return ProviderIdentity(
            provider_id="openrouter",
            adapter=AdapterKind.OPENAI_COMPATIBLE,
            required_env_names=("OPENROUTER_API_KEY",),
        )

    if lowered.startswith("groq/"):
        return ProviderIdentity(
            provider_id="groq",
            adapter=AdapterKind.OPENAI_COMPATIBLE,
            required_env_names=("GROQ_API_KEY",),
        )

    if lowered.startswith("grok/"):
        return ProviderIdentity(
            provider_id="xai",
            adapter=AdapterKind.OPENAI_COMPATIBLE,
            required_env_names=("XAI_API_KEY",),
        )

    if lowered.startswith("compatible/"):
        return ProviderIdentity(
            provider_id="compatible",
            adapter=AdapterKind.OPENAI_COMPATIBLE,
            required_env_names=("GARVIS_COMPAT_API_KEY",),
            required_base_url_env="GARVIS_COMPAT_BASE_URL",
        )

    if (
        lowered.startswith("openai/")
        or lowered.startswith("gpt-")
        or lowered == "o1"
        or lowered.startswith("o1-")
        or lowered == "o3"
        or lowered.startswith("o3-")
        or lowered == "o4"
        or lowered.startswith("o4-")
    ):
        return ProviderIdentity(
            provider_id="openai",
            adapter=AdapterKind.OPENAI_NATIVE,
            required_env_names=("OPENAI_API_KEY",),
        )

    # Gemini is intentionally discoverable but not treated as already integrated.
    if lowered.startswith("gemini/"):
        return ProviderIdentity(
            provider_id="gemini",
            adapter=AdapterKind.MISSING,
            required_env_names=("GEMINI_API_KEY",),
            adapter_supported=False,
        )

    # Security-hardening rule: unknown provider identity fails closed.
    # Environment-variable presence is availability metadata only. It does not
    # establish provider identity, capability, trust, or execution authority.
    return ProviderIdentity(
        provider_id="unknown",
        adapter=AdapterKind.UNKNOWN,
        required_env_names=(),
        adapter_supported=False,
    )


def _present(env: Mapping[str, str], name: str) -> bool:
    return bool((env.get(name) or "").strip())


def configuration_present(identity: ProviderIdentity, env: Mapping[str, str]) -> bool:
    if any(not _present(env, name) for name in identity.required_env_names):
        return False
    if identity.required_base_url_env and not _present(env, identity.required_base_url_env):
        return False
    return True


def remote_model_organ(
    model: str,
    *,
    env: Mapping[str, str],
    declared_capabilities: Iterable[str] = ("text",),
    verified_capabilities: Iterable[str] = (),
) -> AIOrgan:
    identity = identify_provider(model)
    configured = (
        False
        if identity.provider_id == "unknown"
        else configuration_present(identity, env)
    )
    notes: List[str] = []
    if identity.provider_id == "gemini":
        notes.append("provider discovered; dedicated GARVIS adapter not yet established")
    if identity.provider_id == "unknown":
        notes.append("unknown_provider_rejected_fail_closed")
    if not identity.adapter_supported:
        notes.append("adapter_missing")
    if not configured:
        notes.append("configuration_missing")

    return AIOrgan(
        organ_id=f"remote:{model}",
        candidate_type=CandidateType.REMOTE_API,
        provider_id=identity.provider_id,
        model=model,
        adapter=identity.adapter,
        configured=configured,
        programmable=bool(identity.adapter_supported and configured),
        adapter_supported=identity.adapter_supported,
        declared_capabilities=frozenset(declared_capabilities),
        verified_capabilities=frozenset(verified_capabilities),
        authority=Authority.CANDIDATE_ONLY,
        notes=tuple(notes),
    )


def garvis_local_organ() -> AIOrgan:
    return AIOrgan(
        organ_id="garvis:local",
        candidate_type=CandidateType.GARVIS_LOCAL,
        provider_id="garvis",
        model=None,
        adapter=AdapterKind.GARVIS_LOCAL,
        configured=True,
        programmable=True,
        adapter_supported=True,
        declared_capabilities=frozenset(
            {"text", "reasoning", "context", "integration", "planning"}
        ),
        verified_capabilities=frozenset(),
        authority=Authority.GARVIS,
        notes=("GARVIS-owned local reasoning; capability claims remain testable",),
    )


def garvis_evidence_organ() -> AIOrgan:
    return AIOrgan(
        organ_id="garvis:evidence",
        candidate_type=CandidateType.GARVIS_ORGAN,
        provider_id="garvis",
        model=None,
        adapter=AdapterKind.INTERNAL,
        configured=True,
        programmable=True,
        adapter_supported=True,
        declared_capabilities=frozenset({"evidence", "research", "verification"}),
        verified_capabilities=frozenset(),
        authority=Authority.EVIDENCE_ONLY,
        notes=("evidence is not model self-certification",),
    )


def android_app_organ(
    package_name: str,
    *,
    label: Optional[str] = None,
    programmable: bool = False,
    declared_capabilities: Iterable[str] = (),
) -> AIOrgan:
    """Register an installed AI app without pretending GARVIS can control it."""
    if programmable:
        raise ValueError(
            "Android app control requires a verified integration adapter"
        )
    display = label or package_name
    return AIOrgan(
        organ_id=f"android:{package_name}",
        candidate_type=CandidateType.ANDROID_APP,
        provider_id=display,
        model=None,
        adapter=AdapterKind.MANUAL_ONLY,
        configured=True,
        programmable=False,
        adapter_supported=False,
        declared_capabilities=frozenset(declared_capabilities),
        verified_capabilities=frozenset(),
        authority=Authority.CANDIDATE_ONLY,
        notes=(
            "installed-app presence does not imply a programmable integration",
        ),
    )


class UniversalAIRegistry:
    """Pure metadata registry. It never executes providers."""

    def __init__(self, organs: Iterable[AIOrgan] = ()) -> None:
        self._organs: Dict[str, AIOrgan] = {}
        for organ in organs:
            self.register(organ)

    def register(self, organ: AIOrgan) -> None:
        self._organs[organ.organ_id] = organ

    def all(self) -> Tuple[AIOrgan, ...]:
        return tuple(self._organs[key] for key in sorted(self._organs))

    def get(self, organ_id: str) -> Optional[AIOrgan]:
        return self._organs.get(organ_id)

    def candidates(
        self,
        capability: str = "text",
        *,
        verified_only: bool = False,
        programmable_only: bool = True,
    ) -> Tuple[AIOrgan, ...]:
        found: List[AIOrgan] = []
        for organ in self.all():
            if programmable_only and not organ.programmable:
                continue
            if not organ.configured or not organ.adapter_supported:
                continue
            if organ.supports(capability, verified_only=verified_only):
                found.append(organ)
        return tuple(found)

    def safe_report(self) -> Dict[str, object]:
        """Return metadata only; secret values are never included."""
        return {
            "schema": "garvis.universal_ai_registry.v1",
            "organs": [
                {
                    "organ_id": organ.organ_id,
                    "candidate_type": organ.candidate_type.value,
                    "provider_id": organ.provider_id,
                    "model": organ.model,
                    "adapter": organ.adapter.value,
                    "configured": organ.configured,
                    "programmable": organ.programmable,
                    "adapter_supported": organ.adapter_supported,
                    "declared_capabilities": sorted(organ.declared_capabilities),
                    "verified_capabilities": sorted(organ.verified_capabilities),
                    "authority": organ.authority.value,
                    "notes": list(organ.notes),
                }
                for organ in self.all()
            ],
        }


def build_registry(
    models: Sequence[str],
    *,
    env: Mapping[str, str],
    capability_overrides: Optional[Mapping[str, Iterable[str]]] = None,
    android_packages: Sequence[str] = (),
) -> UniversalAIRegistry:
    overrides = capability_overrides or {}
    registry = UniversalAIRegistry((garvis_local_organ(), garvis_evidence_organ()))

    for model in models:
        capabilities = overrides.get(model, ("text",))
        registry.register(
            remote_model_organ(
                model,
                env=env,
                declared_capabilities=capabilities,
            )
        )

    for package_name in android_packages:
        registry.register(android_app_organ(package_name))

    return registry


def hypercube_provider_plan(
    registry: UniversalAIRegistry,
    *,
    capability: str = "text",
) -> Tuple[PerspectiveAssignment, ...]:
    """Create a non-executing 8-perspective scheduling plan.

    Context, Evidence, and Integration remain GARVIS-owned. Other perspectives
    may be rotated across programmable candidate providers. This is a routing
    plan, not a truth score or provider vote.
    """
    remote = [
        organ
        for organ in registry.candidates(capability)
        if organ.candidate_type is CandidateType.REMOTE_API
    ]
    local = registry.get("garvis:local")
    fallback_owner = local.organ_id if local else "garvis:local"

    candidate_codes = {"000", "010", "011", "101", "110"}
    cursor = 0
    assignments: List[PerspectiveAssignment] = []

    for code, name in PERSPECTIVES:
        if code == "001":
            assignments.append(
                PerspectiveAssignment(code, name, fallback_owner, "GARVIS context/memory")
            )
        elif code == "100":
            assignments.append(
                PerspectiveAssignment(code, name, "garvis:evidence", "evidence is independently grounded")
            )
        elif code == "111":
            assignments.append(
                PerspectiveAssignment(code, name, fallback_owner, "GARVIS integration retains authority")
            )
        elif code in candidate_codes and remote:
            organ = remote[cursor % len(remote)]
            cursor += 1
            assignments.append(
                PerspectiveAssignment(
                    code,
                    name,
                    organ.organ_id,
                    "candidate perspective; output still requires GARVIS verification",
                )
            )
        else:
            assignments.append(
                PerspectiveAssignment(code, name, fallback_owner, "local fallback")
            )

    return tuple(assignments)
