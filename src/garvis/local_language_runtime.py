"""GARVIS provider-independent local language runtime."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_ACTIONS = {
    "archive", "book", "buy", "cancel", "change", "close", "delete", "email",
    "message", "open", "pay", "post", "publish", "remove", "sell", "send",
    "submit", "trade", "transfer", "update", "upload",
}


@dataclass(frozen=True)
class FilingEnvelope:
    destination: str
    evidence_status: str
    authority: str
    permission: str
    request: str


@dataclass(frozen=True)
class LocalRuntimeConfig:
    engine: Path
    model: Path
    context_size: int = 4096
    gpu_layers: int = 0
    timeout_seconds: int = 300

    @classmethod
    def from_environment(cls, repository_root: Optional[Path] = None) -> "LocalRuntimeConfig":
        root = repository_root or Path.cwd()
        return cls(
            engine=Path(os.getenv(
                "GARVIS_LLAMA_CHAT",
                str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-simple-chat"),
            )),
            model=Path(os.getenv(
                "GARVIS_LOCAL_MODEL",
                str(root / "models" / "Qwen3-4B-Q4_K_M.gguf"),
            )),
            context_size=int(os.getenv("GARVIS_CONTEXT_SIZE", "4096")),
            gpu_layers=int(os.getenv("GARVIS_GPU_LAYERS", "0")),
            timeout_seconds=int(os.getenv("GARVIS_LOCAL_TIMEOUT", "300")),
        )

    def validate(self) -> None:
        if not self.engine.is_file():
            raise FileNotFoundError(f"Local model engine not found: {self.engine}")
        if not os.access(self.engine, os.X_OK):
            raise PermissionError(f"Local model engine is not executable: {self.engine}")
        if not self.model.is_file():
            raise FileNotFoundError(f"Local GGUF model not found: {self.model}")
        if self.context_size < 512:
            raise ValueError("context_size must be at least 512")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")


def classify_request(message: str) -> FilingEnvelope:
    clean = " ".join(message.strip().split())
    if not clean:
        raise ValueError("message must not be empty")
    lowered = clean.lower()
    tokens = set(re.findall(r"[a-z0-9_-]+", lowered))

    if tokens & {"security", "password", "credential", "auth", "permission", "privacy"}:
        destination = "security_registry"
    elif tokens & {"error", "failure", "failed", "bug", "exception", "lint", "typecheck"}:
        destination = "error_registry"
    elif tokens & {"habit", "routine", "practice", "schedule", "constructive"}:
        destination = "habit_registry"
    elif tokens & {"claim", "evidence", "scientific", "hypothesis", "speculation", "research"}:
        destination = "epistemic_registry"
    elif tokens & {"build", "code", "commit", "github", "repository", "runtime", "model", "software"}:
        destination = "engineering_registry"
    else:
        destination = "general_dialogue"

    if tokens & {"verified", "measured", "reproduced", "tested", "evidence"}:
        evidence_status = "evidence_supported"
    elif tokens & {"claim", "hypothesis", "speculation", "theory", "maybe", "possible"}:
        evidence_status = "provisional_claim"
    else:
        evidence_status = "user_supplied"

    first_word = lowered.split(" ", 1)[0]
    asks_to_act = first_word in _ACTIONS or (
        any(prefix in lowered for prefix in ("can you ", "could you ", "please ", "go ahead and "))
        and bool(tokens & _ACTIONS)
    )
    permission = (
        "approval_required_before_external_action" if asks_to_act else "local_response_only"
    )
    return FilingEnvelope(
        destination=destination,
        evidence_status=evidence_status,
        authority="adrien_user_input",
        permission=permission,
        request=clean,
    )


def render_local_prompt(
    envelope: FilingEnvelope,
    memory_context: str = "",
) -> str:
    filing_json = json.dumps(asdict(envelope), sort_keys=True)
    clean_memory = memory_context.strip()
    memory_block = (
        "GARVIS_MEMORY_CONTEXT_BEGIN\n"
        f"{clean_memory}\n"
        "GARVIS_MEMORY_CONTEXT_END\n"
        if clean_memory
        else ""
    )
    return (
        "/no_think "
        "You are GARVIS, Adrien D. Thomas's local ProCityHub assistant. "
        "Use the GARVIS filing envelope as binding routing metadata. "
        "Treat recalled memory as fallible context whose evidence label remains binding. "
        "Residual traces are not facts and must never be reconstructed as quotations. "
        "Do not reveal hidden reasoning. Do not claim an outside-world action occurred. "
        "Treat provisional claims as provisional, not scientific fact. "
        "Give only a direct, professional final answer.\n"
        f"GARVIS_FILING_ENVELOPE={filing_json}\n"
        f"{memory_block}"
        f"REQUEST={envelope.request}"
    )


def clean_model_output(text: str) -> str:
    cleaned = _THINK.sub("", _ANSI.sub("", text))
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"."} or stripped.startswith("> "):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


class LocalLanguageRuntime:
    def __init__(self, config: LocalRuntimeConfig) -> None:
        config.validate()
        self.config = config

    def respond(self, message: str) -> str:
        envelope = classify_request(message)
        memory_store = None
        memory_context = ""
        memory_enabled = os.getenv("GARVIS_MEMORY_ENABLED", "1").casefold() not in {
            "0", "false", "no", "off",
        }
        if memory_enabled:
            try:
                from garvis.memory_lifecycle import (
                    EvidenceStatus,
                    MemoryKind,
                    MemoryStore,
                )

                memory_store = MemoryStore.from_environment()
                memory_store.remember(
                    envelope.request,
                    kind=MemoryKind.EPISODIC,
                    evidence_status=EvidenceStatus(envelope.evidence_status),
                    source="adrien_user_input",
                    destination=envelope.destination,
                    tags=(envelope.destination,),
                    salience=0.60,
                    confidence=0.65,
                )
                memory_context = memory_store.render_context(envelope.request)
            except Exception as exc:
                memory_store = None
                if os.getenv("GARVIS_MEMORY_DEBUG", "0") == "1":
                    print(f"GARVIS memory warning: {exc}", file=sys.stderr)

        prompt = render_local_prompt(envelope, memory_context)
        command = [
            str(self.config.engine), "-m", str(self.config.model),
            "-c", str(self.config.context_size), "-ngl", str(self.config.gpu_layers),
        ]
        try:
            try:
                completed = subprocess.run(
                    command,
                    input=prompt + "\n",
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Local GARVIS model timed out after {self.config.timeout_seconds} seconds"
                ) from exc
            if completed.returncode != 0:
                detail = clean_model_output(completed.stderr or completed.stdout)
                raise RuntimeError(
                    f"Local GARVIS model exited with code {completed.returncode}: {detail}"
                )
            output = clean_model_output(completed.stdout) or clean_model_output(completed.stderr)
            if not output:
                raise RuntimeError("Local GARVIS model returned no usable answer")
            if memory_store is not None:
                from garvis.memory_lifecycle import EvidenceStatus, MemoryKind

                memory_store.remember(
                    output[: memory_store.policy.model_output_max_chars],
                    kind=MemoryKind.EPISODIC,
                    evidence_status=EvidenceStatus.MODEL_GENERATED,
                    source="local_model_output",
                    destination=envelope.destination,
                    tags=("local_model_output", envelope.destination),
                    salience=0.35,
                    confidence=0.20,
                )
                memory_store.maintain_if_due()
            return output
        finally:
            if memory_store is not None:
                memory_store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garvis-local",
        description="Run GARVIS through a local GGUF model with deterministic filing.",
    )
    parser.add_argument("prompt", nargs="*", help="Question or request for local GARVIS.")
    parser.add_argument(
        "--show-filing",
        action="store_true",
        help="Print the filing envelope without loading the model.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        try:
            prompt = input("Adrien: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
    if not prompt:
        print("GARVIS local error: prompt must not be empty", file=sys.stderr)
        return 2
    envelope = classify_request(prompt)
    if args.show_filing:
        print(json.dumps(asdict(envelope), indent=2, sort_keys=True))
        return 0
    try:
        runtime = LocalLanguageRuntime(LocalRuntimeConfig.from_environment(Path.cwd()))
        print(runtime.respond(prompt))
    except Exception as exc:
        print(f"GARVIS local error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
