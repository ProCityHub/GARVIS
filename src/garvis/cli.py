"""Command-line interface for the GARVIS response spine."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .lattice_cycle_cli import run_lattice_cycle_file

if TYPE_CHECKING:
    from .assistant import GarvisAssistant
    from .capability_runtime import CapabilityAwareRuntime

DEFAULT_REMOTE_MODEL = "gpt-5.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garvis",
        description="Run GARVIS locally by default, with an explicit remote fallback.",
    )
    parser.add_argument("prompt", nargs="*", help="Question or request for GARVIS.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive conversation. This is the default when no prompt is supplied.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use the legacy remote provider runtime instead of the local GGUF runtime.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GARVIS_MODEL", DEFAULT_REMOTE_MODEL),
        help="Remote model name used only with --remote. Default: %(default)s.",
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
    parser.add_argument(
        "--lattice-cycle",
        type=Path,
        default=None,
        metavar="JSON_PATH",
        help=(
            "Run the local deterministic lattice cognitive cycle "
            "from an explicit JSON evidence file."
        ),
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=0,
        help="Non-negative heartbeat cycle number. Default: %(default)s.",
    )
    parser.add_argument(
        "--external-action",
        action="store_true",
        help=(
            "Mark the evaluated proposal as external. This requests "
            "human review and never performs the action."
        ),
    )
    return parser


def _run_local_lattice_cycle(args: argparse.Namespace) -> int:
    if args.prompt or args.interactive:
        print(
            "GARVIS lattice-cycle error: --lattice-cycle cannot be "
            "combined with a conversation prompt or --interactive.",
            file=sys.stderr,
        )
        return 2

    try:
        summary = run_lattice_cycle_file(
            path=args.lattice_cycle,
            cycle=args.cycle,
            external_action=args.external_action,
        )
    except ValueError as exc:
        print(
            f"GARVIS lattice-cycle error: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _build_local_runtime() -> CapabilityAwareRuntime:
    from .capability_runtime import CapabilityAwareRuntime
    from .local_language_runtime import LocalLanguageRuntime, LocalRuntimeConfig

    local = LocalLanguageRuntime(LocalRuntimeConfig.from_environment(Path.cwd()))
    return CapabilityAwareRuntime(local)


def _configure_local_memory(args: argparse.Namespace) -> None:
    os.environ["GARVIS_MEMORY_ENABLED"] = "0" if args.no_memory else "1"
    if args.db is not None:
        os.environ["GARVIS_MEMORY_DB"] = str(args.db.expanduser())


def _run_local_interactive(runtime: CapabilityAwareRuntime) -> int:
    print("GARVIS local is active. Type /exit to end the session.")
    while True:
        try:
            prompt = input("Adrien: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit"}:
            return 0

        try:
            reply = runtime.respond(prompt)
        except Exception as exc:
            print(f"GARVIS local error: {exc}", file=sys.stderr)
            continue

        print("GARVIS:")
        print(reply)


def _run_local(args: argparse.Namespace) -> int:
    _configure_local_memory(args)

    try:
        runtime = _build_local_runtime()
    except Exception as exc:
        print(f"GARVIS local configuration error: {exc}", file=sys.stderr)
        return 2

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        return _run_local_interactive(runtime)

    try:
        print(runtime.respond(prompt))
    except Exception as exc:
        print(f"GARVIS local error: {exc}", file=sys.stderr)
        return 1
    return 0


def _check_configuration(model: str | None = None) -> str | None:
    from .anthropic_backend import is_anthropic_model

    resolved = model or os.getenv("GARVIS_MODEL", DEFAULT_REMOTE_MODEL)
    if is_anthropic_model(resolved):
        if not os.getenv("ANTHROPIC_API_KEY"):
            return (
                "ANTHROPIC_API_KEY is not set. Add it to your environment before "
                "running GARVIS on an Anthropic model; do not place API keys in "
                "the repository."
            )
        return None
    if resolved.strip().lower().startswith("litellm/"):
        return None  # provider-specific keys; litellm reports its own errors
    if not os.getenv("OPENAI_API_KEY"):
        return (
            "OPENAI_API_KEY is not set. Add it to your environment before running GARVIS; "
            "do not place API keys in the repository."
        )
    return None


def _print_reply(text: str, requires_approval: bool, approval_reason: str | None) -> None:
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
    if args.lattice_cycle is not None:
        return _run_local_lattice_cycle(args)
    if not args.remote:
        return _run_local(args)

    from .assistant import GarvisAssistant

    configuration_error = _check_configuration(args.model)
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
