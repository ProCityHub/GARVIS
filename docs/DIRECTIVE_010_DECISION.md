# DIRECTIVE-010 Decision Record — Session Resilience Organ

- Decision date: 2026-07-18
- Implementation branch: `directive-010-session-resilience`
- Authority: Adrien D. Thomas
- Status: proposed for pull-request audit; not merged

## Motivation

On 2026-07-12, a GARVIS request encountered an OpenAI `429` rate-limit
response. The exception was not contained at the chat boundary. The process
ended and the active conversation state was lost.

This event is evidence of an implementation defect only. It is not evidence
of consciousness, autonomous agency, or a physical memory law.

## Screenshot record

1. **Crash screenshot — 2026-07-12, external audit record.** Shows the
   unhandled `429`/rate-limit failure and the interrupted GARVIS session.
2. **Validation screenshot — phone self-test record.** Shows
   `garvis_resilience self-test`, followed by:
   `PASS: ledger persists, resumes, caps context, recalls history.` The test
   was rerun successfully during Directive-010 integration on 2026-07-19.

The screenshots remain in Adrien's audit conversation record. They are cited
here but are not silently copied, renamed, or represented as repository files.

## Defect 1 — transient errors terminated the session

Observed behavior:

- A `429` response propagated through the runtime.
- Retry behavior did not preserve a stable conversational process.
- Adrien had to restart the interface manually.

Fix:

- `call_with_retry()` applies bounded exponential backoff.
- `Retry-After` information is honored when supplied.
- Retryable `429`, timeout, connection, `502`, and `503` failures are
  distinguished from permanent failures.
- The original ledger remains on disk throughout retry handling.

## Defect 2 — conversation survival depended on the live process

Observed behavior:

- Useful context existed in process memory and the SDK session database.
- Process death could interrupt the current conversational chain.
- Recovery required manual database inspection and reset.

Fix:

- `SessionLedger` writes append-only JSONL records to local storage.
- Adrien's input is flushed immediately after it is read.
- The completed assistant reply is flushed before it is printed.
- Starting the same named session reloads the prior ledger and prints
  `[ledger] resumed`.

## Defect 3 — full history was repeatedly resent

Observed behavior:

- The historical Termux wrapper prepended its complete constitutional context
  to every user message.
- Those repeated blocks were then persisted as ordinary conversation turns.
- One session reached 226 saved messages and 2,425,328 stored characters.
- The next request attempted roughly 474,000–486,000 tokens and failed against
  the applicable token-throughput limit.

Fix:

- Wrapper context is separated from Adrien's current message.
- Only the current message is appended to the durable ledger.
- `build_context()` sends only a bounded recent-turn window to the model.
- Older turns remain on disk and available through ledger recall.
- The canonical Termux script opens one resilient interactive runtime rather
  than spawning a new persistent SDK session for every line.

## Five integration points

1. Construct `SessionLedger` at GARVIS startup; an existing session resumes.
2. Append the user input immediately after it is read.
3. Build model messages through the bounded `build_context()` function.
4. Call the model through `call_with_retry()`.
5. Append the assistant reply before displaying it.

## Validation

The standalone module self-test passed:

- ledger persists,
- session resumes,
- context is capped,
- history can be recalled.

Automated integration tests additionally verify:

- repeated wrapper context is not stored in the ledger,
- user input survives a simulated model failure,
- assistant output is stored before the response returns,
- normal and wrapped prompts are separated correctly.

## Acceptance test

1. Start GARVIS with session `adrien-main`.
2. Say one line and receive a reply.
3. Kill the GARVIS process without using `/exit`.
4. Restart GARVIS with the same session.
5. Observe `[ledger] resumed`.
6. Confirm the previous line remains available in the ledger.
7. Confirm no repository commit, push, merge, deployment, or external action
   occurs through this resilience organ.

## Boundary

Directive-010 is a reliability organ only. It adds no autonomous authority,
internet capability, shell authority, merge authority, or scientific claim
authority. Adrien remains the sole merge and deployment authority.
