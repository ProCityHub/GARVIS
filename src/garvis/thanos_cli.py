"""``garvis thanos`` command-line interface.

Project and conceptual architecture: Adrien D. Thomas (ProCityHub/GARVIS).

Every status line this module prints is derived from persisted state. A
subsystem that has not been implemented reports NOT_IMPLEMENTED and exits
non-zero rather than printing ENABLED, so the status block can never claim
a capability that has no code behind it.

Python 3.9 compatible. Termux-safe default store under ``~/.garvis``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from garvis.thanos_mode import (
    ThanosAuthorizationStore,
    ThanosError,
    create_authorization,
    pause_authorization,
    render_status,
    resume_authorization,
    revoke_authorization,
)
from garvis.upgrade_cycle import CycleStore

__all__ = ["build_parser", "main"]

TARGET_VERSION = "2.0.0-beta.1"
PROJECT_DESIGNATION = "GARVIS_VERSION_2_FULL_AGI_BETA"

#: Subsystems required by the THANOS directive that are not yet implemented.
#: Listed explicitly so ``status`` reports absence instead of silence.
_UNIMPLEMENTED = (
    "INTERNET_RESEARCH_WIRING",
    "UPGRADE_WORKSPACE",
    "REPAIR_ENGINE",
    "GITHUB_CI_MONITOR",
    "RUNTIME_HEALTH_CHECK",
    "ROLLBACK",
    "CAPABILITY_REGISTRY",
)


def default_store_root() -> Path:
    """Return the state directory, honouring ``GARVIS_HOME``."""

    override = os.environ.get("GARVIS_HOME")
    if override:
        return Path(override)
    return Path.home() / ".garvis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garvis thanos",
        description="THANOS MODE standing authorization for GARVIS self-upgrade.",
    )
    parser.add_argument(
        "--store-root",
        default=None,
        help="Directory holding THANOS state (default: ~/.garvis).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("enable", help="Create the standing authorization.")
    sub.add_parser("status", help="Print the current THANOS status block.")
    sub.add_parser("pause", help="Suspend autonomous work without revoking.")
    sub.add_parser("resume", help="Resume a paused standing authorization.")
    sub.add_parser("history", help="Print the authorization chain.")
    sub.add_parser("run", help="Run one autonomous upgrade cycle.")
    sub.add_parser("health", help="Report runtime health.")

    revoke = sub.add_parser("revoke", help="Permanently revoke THANOS MODE.")
    revoke.add_argument("--reason", required=True, help="Why the mode is revoked.")

    return parser


def _paths(root: Path) -> tuple[Path, Path]:
    return root / "thanos.json", root / "cycles.json"


def _load(store: ThanosAuthorizationStore):
    try:
        return store.load()
    except ThanosError as error:
        print(f"THANOS_STATE=TAMPERED\nERROR={error}")
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.store_root) if args.store_root else default_store_root()
    auth_path, cycle_path = _paths(root)
    store = ThanosAuthorizationStore(auth_path)

    if args.command == "enable":
        try:
            existing = store.load()
        except ThanosError as error:
            print(f"REFUSED=TAMPERED_STORE\nERROR={error}")
            return 2
        if existing is not None and not existing.is_revoked:
            print("REFUSED=ALREADY_ENABLED")
            print(render_status(existing, target_version=TARGET_VERSION))
            return 1
        if existing is not None and existing.is_revoked:
            # The revoked grant stays in the chain as audit history; the owner
            # issues a new one. Revocation retires an authorization, it does
            # not lock the owner out of the system.
            print(f"SUPERSEDING_REVOKED_AUTHORIZATION={existing.authorization_id}")
            print(f"PREVIOUS_REVOCATION_REASON={existing.revocation_reason}")
        previous = existing.record_hash if existing is not None else "0" * 64
        record = store.append(create_authorization(previous_record_hash=previous))
        print(render_status(record, target_version=TARGET_VERSION))
        print(f"PROJECT_DESIGNATION={PROJECT_DESIGNATION}")
        print("# The lines above state what is AUTHORIZED, not what is built.")
        for subsystem in _UNIMPLEMENTED:
            print(f"{subsystem}=NOT_IMPLEMENTED")
        return 0

    if args.command == "status":
        record = _load(store)
        print(render_status(record, target_version=TARGET_VERSION))
        print(f"PROJECT_DESIGNATION={PROJECT_DESIGNATION}")
        active = CycleStore(cycle_path).resume()
        if active is None:
            print("ACTIVE_CYCLE=NONE")
        else:
            print(f"ACTIVE_CYCLE={active.cycle_id}")
            print(f"ACTIVE_CYCLE_STATE={active.state.value}")
            if active.blocker:
                print(f"ACTIVE_CYCLE_BLOCKER={active.blocker}")
        for subsystem in _UNIMPLEMENTED:
            print(f"{subsystem}=NOT_IMPLEMENTED")
        return 0

    if args.command in {"pause", "resume"}:
        record = _load(store)
        if record is None:
            print("REFUSED=NO_AUTHORIZATION")
            return 1
        try:
            mutate = pause_authorization if args.command == "pause" else resume_authorization
            print(render_status(store.append(mutate(record)), target_version=TARGET_VERSION))
        except ThanosError as error:
            print(f"REFUSED={error}")
            return 1
        return 0

    if args.command == "revoke":
        record = _load(store)
        if record is None:
            print("REFUSED=NO_AUTHORIZATION")
            return 1
        try:
            revoked = store.append(revoke_authorization(record, reason=args.reason))
        except ThanosError as error:
            print(f"REFUSED={error}")
            return 1
        print(render_status(revoked, target_version=TARGET_VERSION))
        return 0

    if args.command == "history":
        try:
            chain = store.history()
        except ThanosError as error:
            print(f"THANOS_STATE=TAMPERED\nERROR={error}")
            return 2
        if not chain:
            print("AUTHORIZATION_CHAIN=EMPTY")
            return 0
        for index, record in enumerate(chain):
            state = "REVOKED" if record.is_revoked else ("PAUSED" if record.paused else "ENABLED")
            print(f"{index}\t{record.updated_at}\t{state}\t{record.record_hash[:16]}")
        print(f"CHAIN_LENGTH={len(chain)}")
        return 0

    if args.command in {"run", "health"}:
        subsystem = "AUTONOMOUS_REPAIR_LOOP" if args.command == "run" else "RUNTIME_HEALTH_CHECK"
        print(f"{subsystem}=NOT_IMPLEMENTED")
        print("REASON=required modules are not present in this build")
        print("MISSING=" + ",".join(_UNIMPLEMENTED))
        return 3

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
