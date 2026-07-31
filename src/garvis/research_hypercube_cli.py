"""Command line for the GARVIS research-to-Hypercube verification bridge."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from garvis.research_hypercube_bridge import BridgeError, run_bridge


def _default_home() -> Path:
    return Path(os.getenv("GARVIS_HOME", str(Path.home() / ".garvis")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m garvis.research_hypercube_cli",
        description=(
            "Research the public internet, let GARVIS create a structured "
            "cognitive packet, and independently verify its arithmetic."
        ),
    )
    parser.add_argument("query", nargs="+", help="Research objective for GARVIS.")
    parser.add_argument(
        "--model",
        default=os.getenv("GARVIS_RESEARCH_MODEL", "compatible/grok-4.5"),
        help="Configured GARVIS reasoning model. Default: %(default)s.",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="GARVIS repository root. Default: current directory.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=_default_home() / "evidence" / "research.json",
        help="Hash-chained evidence ledger path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_home() / "hypercube" / "latest_research_cycle.json",
        help="Verified result JSON path.",
    )
    return parser


def _configure_xai_compatible(model: str) -> None:
    if not model.casefold().startswith("compatible/"):
        return
    xai_key = os.getenv("XAI_API_KEY", "").strip()
    if not os.getenv("GARVIS_COMPAT_API_KEY") and xai_key:
        os.environ["GARVIS_COMPAT_API_KEY"] = xai_key
    if not os.getenv("GARVIS_COMPAT_BASE_URL") and xai_key:
        os.environ["GARVIS_COMPAT_BASE_URL"] = "https://api.x.ai/v1"


async def _main_async(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    query = " ".join(args.query).strip()
    _configure_xai_compatible(args.model)

    try:
        result = await run_bridge(
            query=query,
            repository_root=args.repository.expanduser().resolve(),
            model=args.model,
            ledger_path=args.ledger.expanduser(),
            output_path=args.output.expanduser(),
        )
    except (BridgeError, ValueError, RuntimeError) as exc:
        print(f"GARVIS research-hypercube error: {exc}", file=sys.stderr)
        return 2

    summary = {
        "OWNER": result["owner"],
        "QUERY": result["query"],
        "MODEL": result["model"],
        "SOURCES": result["source_count"],
        "PRIMARY_SOURCES": result["primary_source_count"],
        "SNAPSHOT_VALIDATION": result["snapshot_validation"],
        "MATH_VERIFICATION": (
            "PASS" if result["math_verification_passed"] else "FAIL"
        ),
        "EVIDENCE_GATE": "PASS" if result["evidence_gate_passed"] else "FAIL",
        "USABLE_FOR_MATH": result["usable_for_mathematical_followup"],
        "USABLE_FOR_PATCH": result["usable_to_justify_repository_patch"],
        "OUTPUT": str(args.output.expanduser()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return asyncio.run(_main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
