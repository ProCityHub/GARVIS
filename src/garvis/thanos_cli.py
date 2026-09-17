"""Legacy ``garvis thanos`` compatibility entry point.

THANOS is not an active GARVIS authority source. Adrien D. Thomas is the
creator and final human authority for GARVIS. Automatic runtime liveness is
provided by the GARVIS heartbeat service.

Historical THANOS code may remain for provenance; this CLI no longer creates
or consumes THANOS standing authority.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from garvis.creator_authority import (
    AUTHORITY_SOURCE,
    CREATOR,
    CREATOR_ASSERTION,
)
from garvis.heartbeat_service import AutomaticHeartbeatService

__all__ = ["build_parser", "main"]


def default_store_root() -> Path:
    override = os.environ.get("GARVIS_HOME")
    if override:
        return Path(override)
    return Path.home() / ".garvis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garvis thanos",
        description=(
            "Legacy alias. GARVIS heartbeat authority is creator-owned."
        ),
    )
    parser.add_argument("--store-root", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "enable",
        "status",
        "pause",
        "resume",
        "history",
        "run",
        "health",
    ):
        sub.add_parser(command)
    revoke = sub.add_parser("revoke")
    revoke.add_argument("--reason", required=True)
    return parser


def _legacy_status() -> None:
    print("THANOS=LEGACY_ONLY")
    print("THANOS_OPERATIONAL_AUTHORITY=DISABLED")
    print("AUTHORITY_SOURCE=" + AUTHORITY_SOURCE)
    print("CREATOR=" + CREATOR)
    print("CREATOR_ASSERTION=" + CREATOR_ASSERTION)


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    root = (
        Path(args.store_root)
        if args.store_root
        else default_store_root()
    )
    heartbeat_root = root / "heartbeat"

    if args.command == "run":
        service = AutomaticHeartbeatService(
            heartbeat_root,
            interval_seconds=0.0,
        )
        try:
            state = service.run_once()
            _legacy_status()
            print("HEARTBEAT_CYCLE=" + state.cycle_id)
            print("HEARTBEAT_STATUS=" + state.status.value.upper())
            return 0 if state.status.value == "completed" else 3
        finally:
            service.close()

    if args.command == "health":
        service = AutomaticHeartbeatService(heartbeat_root)
        try:
            _legacy_status()
            print(json.dumps(service.health(), sort_keys=True))
            return 0
        finally:
            service.close()

    _legacy_status()
    print("RESULT=NO_THANOS_AUTHORITY_MUTATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
