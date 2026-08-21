"""Termux/local command surface for GARVIS automatic heartbeat."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from .creator_authority import CREATOR
from .heartbeat_service import AutomaticHeartbeatService


def default_root() -> Path:
    home = Path(
        os.environ.get(
            "GARVIS_HOME",
            str(Path.home() / ".garvis"),
        )
    )
    return home / "heartbeat"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="garvis-heartbeat")
    parser.add_argument("--root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run-once")
    sub.add_parser("daemon")
    sub.add_parser("health")
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    service = AutomaticHeartbeatService(
        args.root or default_root()
    )
    try:
        if args.command == "run-once":
            state = service.run_once()
            print(
                json.dumps(
                    {
                        "heartbeat": (
                            "ACTIVE"
                            if state.status.value == "completed"
                            else state.status.value.upper()
                        ),
                        "cycle_id": state.cycle_id,
                        "artifact_sha256": state.artifact_sha256,
                        "creator": CREATOR,
                    },
                    sort_keys=True,
                )
            )
            return 0 if state.status.value == "completed" else 3

        if args.command == "daemon":
            print(
                "HEARTBEAT=STARTING CREATOR=" + CREATOR,
                flush=True,
            )
            service.run_forever()
            return 0

        health = service.health()
        print(json.dumps(health, sort_keys=True))
        return 0 if health.get("heartbeat_running") else 3
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
