#!/usr/bin/env python3
"""
GARVIS RESILIENCE MODULE — DIRECTIVE-010 candidate
===================================================
Fixes three defects observed 2026-07-12 (session crash, screenshot on record):

  DEFECT 1: 429 rate-limit killed the process (retry-after was 10s).
            FIX: call_with_retry() — exponential backoff, honors Retry-After.

  DEFECT 2: conversation memory lived only in RAM; process death = amnesia.
            FIX: SessionLedger — append-only JSONL on disk, written after
            EVERY turn, reloaded on startup. The conversation joins the
            chain of custody. "Nothing forgotten" now includes the talk.

  DEFECT 3: full history resent every turn; context grew unbounded until
            it hit the TPM limit (215k used, 251k requested, 400k cap).
            FIX: build_context() — hard cap on turns sent to the model;
            older turns stay in the ledger (recallable), not in the prompt.

Constitutional note: local-first, no new capabilities, no network beyond
the existing LLM call, no autonomous action. Pure reliability organ.

WIRING (three lines in garvis chat loop):
    from garvis_resilience import SessionLedger, call_with_retry, build_context
    ledger = SessionLedger()                       # at startup (auto-resumes)
    ...
    ledger.append("user", user_text)               # after reading input
    messages = build_context(system_prompt, ledger)
    reply = call_with_retry(client, model, messages)
    ledger.append("assistant", reply)              # before printing
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
LEDGER_DIR = Path(os.environ.get("GARVIS_HOME", str(Path.home()))) / ".garvis_sessions"
MAX_CONTEXT_TURNS = 30          # newest N turns sent to the model
MAX_RETRIES = 5
BASE_BACKOFF_S = 5.0            # doubles each retry: 5, 10, 20, 40, 80


# ----------------------------------------------------------------------
# DEFECT 2 FIX — the session ledger
# ----------------------------------------------------------------------
class SessionLedger:
    """Append-only JSONL conversation ledger. One file per day, auto-resume.

    Every turn is flushed to disk the moment it happens. Kill the process,
    pull the battery, hit the rate limit — the conversation survives.
    """

    def __init__(self, session_name: str | None = None):
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        name = session_name or datetime.now().strftime("session_%Y%m%d")
        self.path = LEDGER_DIR / f"{name}.jsonl"
        self.turns: list[dict] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.turns.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # tolerate a torn final line from a crash
            if self.turns:
                print(f"[ledger] resumed {len(self.turns)} turns from {self.path.name}")

    def append(self, role: str, content: str) -> None:
        turn = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
        }
        self.turns.append(turn)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())          # survive power loss, not just crash

    def recall(self, keyword: str, limit: int = 5) -> list[dict]:
        """Deterministic search over the full history (not just the window)."""
        kw = keyword.lower()
        hits = [t for t in self.turns if kw in t["content"].lower()]
        return hits[-limit:]


# ----------------------------------------------------------------------
# DEFECT 3 FIX — bounded context
# ----------------------------------------------------------------------
def build_context(system_prompt: str, ledger: SessionLedger,
                  max_turns: int = MAX_CONTEXT_TURNS) -> list[dict]:
    """System prompt + newest N turns only. History beyond the window
    lives in the ledger and is reachable via ledger.recall()."""
    recent = ledger.turns[-max_turns:]
    msgs = [{"role": "system", "content": system_prompt}]
    for t in recent:
        if t["role"] in ("user", "assistant"):
            msgs.append({"role": t["role"], "content": t["content"]})
    return msgs


# ----------------------------------------------------------------------
# DEFECT 1 FIX — retry with backoff
# ----------------------------------------------------------------------
def call_with_retry(client, model: str, messages: list[dict],
                    max_retries: int = MAX_RETRIES) -> str:
    """Wraps the chat call. A 429 or transient network error becomes a
    wait, not a death. Raises only after max_retries genuine failures."""
    delay = BASE_BACKOFF_S
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content
        except Exception as e:                      # noqa: BLE001
            last_err = e
            msg = str(e)
            retriable = ("429" in msg or "rate" in msg.lower()
                         or "timeout" in msg.lower() or "connection" in msg.lower()
                         or "503" in msg or "502" in msg)
            if not retriable or attempt == max_retries:
                raise
            # honor an explicit retry-after if the message contains one
            wait = delay
            for token in msg.replace(",", " ").split():
                if token.endswith("s.") or token.endswith("s"):
                    try:
                        hinted = float(token.rstrip("s."))
                        if 0 < hinted < 300:
                            wait = hinted + 1.0
                            break
                    except ValueError:
                        pass
            print(f"[retry {attempt}/{max_retries}] transient error; "
                  f"waiting {wait:.0f}s — session preserved.")
            time.sleep(wait)
            delay *= 2
    raise last_err  # pragma: no cover


# ----------------------------------------------------------------------
# SELF-TEST (no network): ledger write/resume + context bounding
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== garvis_resilience self-test ===")
    test = SessionLedger("selftest_tmp")
    n0 = len(test.turns)
    test.append("user", "test turn A")
    test.append("assistant", "test reply B")
    reloaded = SessionLedger("selftest_tmp")
    assert len(reloaded.turns) == n0 + 2, "resume failed"
    ctx = build_context("SYS", reloaded, max_turns=1)
    assert ctx[0]["role"] == "system" and len(ctx) == 2, "context cap failed"
    hits = reloaded.recall("turn A")
    assert hits, "recall failed"
    (LEDGER_DIR / "selftest_tmp.jsonl").unlink()
    print("PASS: ledger persists, resumes, caps context, recalls history.")
    print("Wire it into the chat loop per the header instructions.")
