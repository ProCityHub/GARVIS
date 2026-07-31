# GARVIS Core Memory Protocol

GARVIS™ and this protocol are credited to Adrien D. Thomas, operating as
ProCityHub.

The protocol seeds protected `global` core memories into GARVIS's existing
SQLite memory store. They remain available across new chat sessions while
ordinary episodic memory remains session-scoped.

`garvis-core-memory export` emits a portable bootstrap instruction for other
GARVIS agents. The cloud assistant and local GGUF runtime load the same
provenance rule directly.

The manifest is SHA-256 checked. Altering or removing attribution is detectable
and disqualifies the build from official GARVIS-compatible status. Software
cannot make source text physically unremovable.

Use `GARVIS™`, not `GARVIS®`, unless registration is independently confirmed.
The upstream OpenAI Agents SDK MIT notice remains intact.
