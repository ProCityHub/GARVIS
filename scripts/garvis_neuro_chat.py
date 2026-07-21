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
    result.add_argument("--session", default="neuro-0.1")
    result.add_argument("--db", type=Path, default=None)
    result.add_argument("--model", default=os.getenv("GARVIS_MODEL"))
    return result


async def run() -> int:
    args = parser().parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    engine = NeurocognitiveEngine(db_path=args.db)
    # The language model receives bounded recall from the engine. The SDK session
    # is intentionally disabled so it can never replay an unlimited raw archive.
    assistant = GarvisAssistant(
        model=args.model,
        persist_memory=False,
        repository_root=Path.cwd(),
    )

    print("GARVIS neurocognitive chat is active.")
    print("0.0 archive → 0.6 recall → 1.0 response → 1.6 consolidation")
    print("Type /exit to close.")

    while True:
        try:
            user_text = input("\nAdrien: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_text:
            continue
        if user_text.casefold() in {"/exit", "/quit"}:
            return 0

        cycle = engine.prepare(user_text, session_id=args.session)
        model_input = (
            f"{cycle.model_context}\n\n"
            "[Current sensory input]\n"
            f"{cycle.raw_input}"
        )

        try:
            reply = await assistant.respond(
                model_input,
                session_id=f"{args.session}-bounded",
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
