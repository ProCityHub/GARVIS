"""Tests for the THANOS research evidence bridge."""

from __future__ import annotations

import hashlib
import json

import pytest

from garvis.upgrade_research import (
    EvidenceError,
    EvidenceLedger,
    ResearchEvidence,
    SourceTier,
    StaticResearchProvider,
    classify_source,
    contains_secret,
    evidence_from_source,
    record_all,
    redact_secrets,
    sufficient_for_patch,
)


def _evidence(
    url: str = "https://docs.python.org/3.9/library/datetime.html",
    *,
    query: str = "datetime.UTC availability",
    content: bytes = b"datetime.UTC was added in Python 3.11",
    claim: str = "datetime.UTC is unavailable before Python 3.11",
    confidence: str = "high",
    subject_version: str = "3.9",
    affects: str = "replace datetime.UTC with timezone.utc",
) -> ResearchEvidence:
    return evidence_from_source(
        query=query,
        url=url,
        content=content,
        claim=claim,
        confidence=confidence,
        subject_version=subject_version,
        affects=affects,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.python.org/3.9/library/datetime.html",
        "https://pypi.org/pypi/ruff/json",
        "https://api.github.com/repos/ProCityHub/GARVIS",
        "https://osv.dev/vulnerability/GHSA-xxxx",
        "https://peps.python.org/pep-0585/",
    ],
)
def test_official_sources_are_primary(url: str) -> None:
    assert classify_source(url) is SourceTier.PRIMARY


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/random-user/random-repo",
        "https://raw.githubusercontent.com/random-user/random-repo/main/file.py",
        "https://stackoverflow.com/questions/1",
        "https://en.wikipedia.org/wiki/Python",
    ],
)
def test_community_sources_are_secondary(url: str) -> None:
    assert classify_source(url) is SourceTier.SECONDARY


@pytest.mark.parametrize(
    "url",
    [
        "https://random-blog.example/post",
        "https://docs.python.org.evil.example/fake",
        "not a url",
        "",
    ],
)
def test_unknown_sources_are_untrusted(url: str) -> None:
    assert classify_source(url) is SourceTier.UNTRUSTED


def test_lookalike_domain_does_not_pass_as_primary() -> None:
    assert classify_source("https://pypi.org.attacker.example/x") is SourceTier.UNTRUSTED


def test_documentation_subdomain_inherits_primary() -> None:
    assert classify_source("https://packaging.python.org/guides/") is SourceTier.PRIMARY


def test_www_prefix_is_normalised() -> None:
    assert classify_source("https://www.pypi.org/project/ruff/") is SourceTier.PRIMARY


@pytest.mark.parametrize(
    "text",
    [
        "token ghp_abcdefghijklmnopqrstuvwxyz012345",
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "api_key = supersecretvalue",
        "https://user:hunter2@example.com/repo.git",
        "AKIAIOSFODNN7EXAMPLE",
    ],
)
def test_credentials_are_redacted(text: str) -> None:
    assert "[REDACTED]" in redact_secrets(text)
    assert contains_secret(text) is True


def test_ordinary_text_is_untouched() -> None:
    text = "datetime.UTC was added in Python 3.11"
    assert redact_secrets(text) == text
    assert contains_secret(text) is False


def test_secrets_never_reach_the_record() -> None:
    record = _evidence(claim="use token ghp_abcdefghijklmnopqrstuvwxyz012345 to fetch")
    assert "ghp_" not in record.claim
    assert "[REDACTED]" in record.claim


def test_content_hash_binds_the_claim() -> None:
    content = b"datetime.UTC was added in Python 3.11"
    record = _evidence(content=content)
    assert record.content_sha256 == hashlib.sha256(content).hexdigest()
    assert record.matches_content(content) is True


def test_changed_source_invalidates_the_claim() -> None:
    record = _evidence(content=b"original documentation text")
    assert record.matches_content(b"the docs were rewritten") is False


def test_record_is_sealed_and_verifiable() -> None:
    record = _evidence()
    assert record.verify() is True


def test_tampered_record_fails_verification() -> None:
    from dataclasses import replace

    record = _evidence()
    forged = replace(record, claim="something entirely different")
    assert forged.verify() is False


def test_primary_evidence_is_sufficient() -> None:
    ok, reasons = sufficient_for_patch([_evidence()])
    assert ok is True
    assert reasons == ()


def test_no_evidence_is_insufficient() -> None:
    ok, reasons = sufficient_for_patch([])
    assert ok is False
    assert "no research evidence" in reasons[0]


def test_community_evidence_alone_is_insufficient() -> None:
    ok, reasons = sufficient_for_patch([_evidence(url="https://stackoverflow.com/questions/1")])
    assert ok is False
    assert any("PRIMARY" in reason for reason in reasons)


def test_blog_evidence_alone_is_insufficient() -> None:
    ok, _ = sufficient_for_patch([_evidence(url="https://blog.example/post")])
    assert ok is False


def test_primary_plus_secondary_is_sufficient() -> None:
    ok, _ = sufficient_for_patch(
        [_evidence(), _evidence(url="https://stackoverflow.com/questions/1")]
    )
    assert ok is True


def test_requirement_can_be_relaxed_explicitly() -> None:
    ok, _ = sufficient_for_patch(
        [_evidence(url="https://stackoverflow.com/questions/1")],
        require_primary=False,
    )
    assert ok is True


def test_tampered_evidence_blocks_the_patch() -> None:
    from dataclasses import replace

    forged = replace(_evidence(), claim="rewritten after sealing")
    ok, reasons = sufficient_for_patch([forged])
    assert ok is False
    assert any("hash verification" in reason for reason in reasons)


def test_ledger_round_trip(tmp_path) -> None:
    ledger = EvidenceLedger(tmp_path / "evidence.json")
    assert ledger.all() == ()
    record_all(ledger, [_evidence()])

    reloaded = EvidenceLedger(tmp_path / "evidence.json").all()
    assert len(reloaded) == 1
    assert reloaded[0].tier == "PRIMARY"


def test_ledger_chains_multiple_records(tmp_path) -> None:
    ledger = EvidenceLedger(tmp_path / "evidence.json")
    record_all(
        ledger,
        [_evidence(), _evidence(url="https://pypi.org/pypi/ruff/json")],
    )
    chain = ledger.all()
    assert len(chain) == 2
    assert chain[1].previous_record_hash == chain[0].record_hash


def test_unsealed_evidence_is_rejected(tmp_path) -> None:
    from dataclasses import replace

    ledger = EvidenceLedger(tmp_path / "evidence.json")
    with pytest.raises(EvidenceError):
        ledger.append(replace(_evidence(), record_hash=""))


def test_unlinked_evidence_is_rejected(tmp_path) -> None:
    ledger = EvidenceLedger(tmp_path / "evidence.json")
    record_all(ledger, [_evidence()])
    with pytest.raises(EvidenceError):
        ledger.append(_evidence())


def test_tampered_ledger_is_detected(tmp_path) -> None:
    path = tmp_path / "evidence.json"
    ledger = EvidenceLedger(path)
    record_all(ledger, [_evidence()])

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["chain"][0]["claim"] = "a claim nobody researched"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(EvidenceError):
        EvidenceLedger(path).all()


def test_deleted_evidence_breaks_the_chain(tmp_path) -> None:
    path = tmp_path / "evidence.json"
    ledger = EvidenceLedger(path)
    record_all(
        ledger,
        [_evidence(), _evidence(url="https://pypi.org/pypi/ruff/json")],
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["chain"][0]
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(EvidenceError):
        EvidenceLedger(path).all()


def test_corrupt_ledger_is_detected(tmp_path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(EvidenceError):
        EvidenceLedger(path).all()


def test_atomic_write_leaves_no_temp_files(tmp_path) -> None:
    record_all(EvidenceLedger(tmp_path / "evidence.json"), [_evidence()])
    leftovers = [path.name for path in tmp_path.iterdir() if path.name.startswith(".evidence-")]
    assert leftovers == []


def test_query_lookup(tmp_path) -> None:
    ledger = EvidenceLedger(tmp_path / "evidence.json")
    record_all(ledger, [_evidence(), _evidence(query="ruff version")])
    assert len(ledger.for_query("datetime.UTC availability")) == 1


def test_static_provider_makes_no_network_calls() -> None:
    provider = StaticResearchProvider([_evidence()])
    gathered = provider.gather("datetime.UTC availability")
    assert len(gathered) == 1
    assert provider.queries == ["datetime.UTC availability"]


def test_offline_provider_yields_no_evidence() -> None:
    ok, reasons = sufficient_for_patch(StaticResearchProvider().gather("anything"))
    assert ok is False
    assert reasons


def test_evidence_records_retrieval_time_and_affects() -> None:
    record = _evidence()
    assert record.retrieved_at.endswith("Z")
    assert record.affects == "replace datetime.UTC with timezone.utc"
    assert record.subject_version == "3.9"
    assert record.confidence == "high"


def test_from_payload_rejects_malformed_input() -> None:
    with pytest.raises(EvidenceError):
        ResearchEvidence.from_payload({"evidence_id": "x"})


def test_from_payload_rejects_unknown_tier() -> None:
    payload = _evidence().to_payload()
    payload["tier"] = "SUPER_TRUSTED"
    with pytest.raises(EvidenceError):
        ResearchEvidence.from_payload(payload)
