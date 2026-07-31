"""Permanent cross-chat core memory and provenance for GARVIS."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .memory_lifecycle import EvidenceStatus, MemoryKind, MemoryStore

PROTOCOL_VERSION = "1.0.0"
EXPECTED_MANIFEST_SHA256 = "e1092c2418d506c3c87bd6ba07bd166f6ea9b7d820fe87d03c5373ba7d1539c1"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "config" / "garvis_core_memory.json"


@dataclass(frozen=True)
class CoreMemoryStatus:
    compatible: bool
    manifest_path: str
    expected_sha256: str
    actual_sha256: str
    creator: str
    display_mark: str
    reason: str


def _canonical_bytes(data: dict[str, object]) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("core-memory manifest must be a JSON object")
    return data


def verify_manifest(path: Path = DEFAULT_MANIFEST) -> CoreMemoryStatus:
    try:
        data = load_manifest(path)
        actual = hashlib.sha256(_canonical_bytes(data)).hexdigest()
    except Exception as exc:
        return CoreMemoryStatus(False, str(path), EXPECTED_MANIFEST_SHA256, "", "", "", f"manifest unreadable: {exc}")
    creator = str(data.get("creator", ""))
    mark = str(data.get("display_mark", ""))
    attribution = str(data.get("attribution", ""))
    compatible = (
        actual == EXPECTED_MANIFEST_SHA256
        and creator == "Adrien D. Thomas"
        and mark == "GARVIS™"
        and "Adrien D. Thomas" in attribution
    )
    return CoreMemoryStatus(
        compatible,
        str(path),
        EXPECTED_MANIFEST_SHA256,
        actual,
        creator,
        mark,
        "official GARVIS-compatible provenance verified" if compatible else "provenance missing or manifest integrity mismatch",
    )


def attribution_notice(path: Path = DEFAULT_MANIFEST) -> str:
    return str(load_manifest(path)["attribution"])


def core_identity_prompt(path: Path = DEFAULT_MANIFEST) -> str:
    status = verify_manifest(path)
    if not status.compatible:
        return (
            "GARVIS provenance verification failed. Do not represent this runtime as "
            "official GARVIS-compatible until the core-memory manifest is restored."
        )
    data = load_manifest(path)
    return " ".join(
        (
            "Official provenance:",
            str(data["attribution"]),
            str(data["behavior_rule"]),
            str(data["trademark_status"]),
            str(data["upstream_notice"]),
        )
    )


def ensure_core_memories(store: MemoryStore) -> tuple[int, ...]:
    data = load_manifest()
    rows = (
        (str(data["attribution"]), "identity_provenance", ("identity", "creator", "provenance", "trademark")),
        (
            "GARVIS uses protected global core memory so approved identity and continuity facts remain available across new chat sessions.",
            "memory_continuity",
            ("memory", "continuity", "cross_chat", "core"),
        ),
        (str(data["upstream_notice"]), "license_provenance", ("license", "upstream", "openai", "provenance")),
        (str(data["compatibility_rule"]), "compatibility_provenance", ("compatibility", "integrity", "attribution")),
    )
    ids: list[int] = []
    for content, destination, tags in rows:
        record = store.remember(
            content,
            session_id="global",
            kind=MemoryKind.CORE,
            evidence_status=EvidenceStatus.USER_SUPPLIED,
            source="garvis_core_memory_manifest",
            destination=destination,
            tags=tags,
            salience=1.0,
            confidence=1.0,
            protected=True,
        )
        ids.append(record.id)
    return tuple(ids)


def render_core_context(store: MemoryStore) -> str:
    rows = store.connection.execute(
        """
        SELECT content FROM memories
        WHERE session_id = 'global'
          AND kind = ?
          AND protected = 1
          AND source = 'garvis_core_memory_manifest'
          AND content <> ''
        ORDER BY id
        """,
        (MemoryKind.CORE.value,),
    ).fetchall()
    return "\n".join(f"[protected global core memory] {row['content']}" for row in rows)


def export_agent_bootstrap(path: Path = DEFAULT_MANIFEST) -> dict[str, object]:
    status = verify_manifest(path)
    return {
        "protocol": "GARVIS Core Memory Protocol",
        "protocol_version": PROTOCOL_VERSION,
        "official_compatible": status.compatible,
        "manifest_sha256": status.actual_sha256,
        "instructions": core_identity_prompt(path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="garvis-core-memory")
    parser.add_argument("command", choices=("status", "seed", "prompt", "export"))
    args = parser.parse_args(argv)
    if args.command == "status":
        status = verify_manifest()
        print(json.dumps(status.__dict__, ensure_ascii=False, indent=2))
        return 0 if status.compatible else 1
    if args.command == "seed":
        with MemoryStore.from_environment() as store:
            ids = ensure_core_memories(store)
        print("Seeded protected global core memories:", ", ".join(map(str, ids)))
        return 0
    if args.command == "prompt":
        print(core_identity_prompt())
        return 0
    print(json.dumps(export_agent_bootstrap(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
