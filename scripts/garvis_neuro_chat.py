#!/usr/bin/env python3
"""Interactive GARVIS chat using bounded neurocognitive recall."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from garvis.assistant import GarvisAssistant
from garvis.neurocognitive import NeurocognitiveEngine


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run GARVIS with the hypercube neurocognitive memory spine."
    )
    result.add_argument("--session", default="neuro-0.2")
    result.add_argument("--db", type=Path, default=None)
    result.add_argument("--model", default=os.getenv("GARVIS_MODEL"))
    result.add_argument(
        "--repository-grounding",
        action="store_true",
        help="Attach repository snapshots when the current request asks about code.",
    )
    return result


def read_user_message() -> str:
    first = input("\nAdrien: ")
    command = first.strip().casefold()

    if command != "/paste":
        return first.strip()

    print("Paste the complete message now.")
    print("Type /send on a new line when the message is complete.")
    print("Type /cancel on a new line to discard it.")

    lines: list[str] = []
    while True:
        line = input()
        command = line.strip().casefold()
        if command == "/send":
            return "\n".join(lines).strip()
        if command == "/cancel":
            return ""
        lines.append(line)


async def run() -> int:
    args = parser().parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    engine = NeurocognitiveEngine(db_path=args.db)
    assistant = GarvisAssistant(
        model=args.model,
        persist_memory=False,
        repository_root=Path.cwd(),
    )

    print("GARVIS neurocognitive chat is active.")
    print("0.0 archive → 0.6 recall → 1.0 response → 1.6 consolidation")
    print("One-line message: type normally and press Enter.")
    print("Multiline message: type /paste, paste everything, then type /send.")
    print("Type /exit to close.")

    while True:
        try:
            user_text = read_user_message()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_text:
            continue
        if user_text.casefold() in {"/exit", "/quit"}:
            return 0

        cycle = engine.prepare(user_text, session_id=args.session)
        model_input = (
            "[Operating instruction]\n"
            "Answer the current sensory input directly. Recalled memory is supporting "
            "context, not a separate request. Do not recite repository evidence, system "
            "architecture, or prior answers unless the current input asks for them.\n\n"
            f"{cycle.model_context}\n\n"
            "[Current sensory input]\n"
            f"{cycle.raw_input}"
        )

        try:
            reply = await assistant.respond(
                model_input,
                session_id=f"{args.session}-bounded",
                ground_repository=args.repository_grounding,
            )
        except Exception as exc:
            engine.feedback(
                session_id=args.session,
                intended="produce a bounded GARVIS response",
                observed=str(exc),
                error_signal=type(exc).__name__,
            )
            print(f"GARVIS error: {exc}", file=sys.stderr)
            continue

        print(f"\nGARVIS: {reply.text}")
        engine.consolidate(cycle=cycle, assistant_text=reply.text)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
