"""Inspect GARVIS phone capabilities and capability audit events."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .capability_broker import ApprovalStore
from .local_file_access import LocalFileAccessStore
from .phone_capabilities import scan_phone_capabilities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="garvis-capabilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan")
    sub.add_parser("pending")
    sub.add_parser("file-pending")
    audit = sub.add_parser("audit")
    audit.add_argument("--limit", type=int, default=20)
    file_audit = sub.add_parser("file-audit")
    file_audit.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        print(json.dumps(scan_phone_capabilities(), indent=2, sort_keys=True))
        return 0
    if args.command in {"file-pending", "file-audit"}:
        with LocalFileAccessStore() as store:
            if args.command == "file-pending":
                pending = store.pending()
                print(pending.render() if pending else "No pending local file approval.")
                return 0
            print(json.dumps(store.recent_audit(args.limit), indent=2, sort_keys=True))
            return 0
    with ApprovalStore() as store:
        if args.command == "pending":
            pending = store.pending()
            print(pending.render() if pending else "No pending capability approval.")
            return 0
        if args.command == "audit":
            print(json.dumps(store.recent_audit(args.limit), indent=2, sort_keys=True))
            return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
