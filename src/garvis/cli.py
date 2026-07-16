"""Command-line interface for the GARVIS response spine."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from .assistant import DEFAULT_MODEL, GarvisAssistant, GarvisResponseError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garvis",
        description="Run GARVIS as a direct conversational assistant.",
    )
    parser.add_argument("prompt", nargs="*", help="Question or request for GARVIS.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive conversation. This is the default when no prompt is supplied.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GARVIS_MODEL", DEFAULT_MODEL),
        help="OpenAI model name. Defaults to GARVIS_MODEL or %(default)s.",
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Conversation session identifier. Default: %(default)s.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite path for persistent conversation memory.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable persistent conversation memory for this run.",
    )
    return parser


def _check_configuration() -> Optional[str]:
    if not os.getenv("OPENAI_API_KEY"):
        return (
            "OPENAI_API_KEY is not set. Add it to your environment before running GARVIS; "
            "do not place API keys in the repository."
        )
    return None


def _print_reply(text: str, requires_approval: bool, approval_reason: Optional[str]) -> None:
    print(text)
    if requires_approval:
        print("\n[Execution status: approval required before any outside-world action.]")
        if approval_reason:
            print(f"[Reason: {approval_reason}]")


async def _run_interactive(assistant: GarvisAssistant, session_id: str) -> int:
    print("GARVIS online. Type /exit to end the session.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit"}:
            return 0

        try:
            reply = await assistant.respond(prompt, session_id=session_id)
        except Exception as exc:  # CLI boundary: return a clear error instead of a traceback.
            print(f"GARVIS error: {exc}", file=sys.stderr)
            continue

        print("GARVIS:")
        _print_reply(reply.text, reply.requires_approval, reply.approval_reason)


async def _run(args: argparse.Namespace) -> int:
    configuration_error = _check_configuration()
    if configuration_error:
        print(f"GARVIS configuration error: {configuration_error}", file=sys.stderr)
        return 2

    assistant = GarvisAssistant(
        model=args.model,
        persist_memory=not args.no_memory,
        session_db=args.db,
    )

    prompt = " ".join(args.prompt).strip()
    if prompt:
        try:
            reply = await assistant.respond(prompt, session_id=args.session)
        except Exception as exc:  # CLI boundary: return a clear error instead of a traceback.
            print(f"GARVIS error: {exc}", file=sys.stderr)
            return 1

        _print_reply(reply.text, reply.requires_approval, reply.approval_reason)
        return 0

    return await _run_interactive(assistant, args.session)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
