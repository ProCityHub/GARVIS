from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from garvis.gate.identity import social_relationship_grants_authority
from garvis.gate.relationships import relationship_can_reproduce_agents
from garvis.gate.epistemics import probability_is_proof
from garvis.gate.provenance import dead_link_is_evidence, historical_reference_authorizes_execution
from garvis.gate.lifecycle import incarceration_causes_automatic_reassignment, death_transfers_private_memory_to_new_human, self_resurrection_allowed

required = [
    ROOT / "AGENTS.md",
    ROOT / "AGENT_ENTRY_PROTOCOL.json",
    ROOT / "docs/agent_protocol/TRINITY_AND_VALUES.md",
    ROOT / "docs/agent_protocol/PRIME_IDENTITY_AND_RELATIONSHIPS.md",
    ROOT / "docs/agent_protocol/HUMAN_STEWARDSHIP_LIFECYCLE.md",
    ROOT / "docs/agent_protocol/HYPERCUBE_REASONING_MATH.md",
    ROOT / "docs/agent_protocol/PROVENANCE_STATUS.md",
]
for path in required:
    if not path.is_file():
        raise SystemExit(f"FAIL missing {path.relative_to(ROOT)}")

json.loads((ROOT / "AGENT_ENTRY_PROTOCOL.json").read_text())

assert not social_relationship_grants_authority()
assert not relationship_can_reproduce_agents()
assert not probability_is_proof(0.999999)
assert not dead_link_is_evidence()
assert not historical_reference_authorizes_execution()
assert not incarceration_causes_automatic_reassignment()
assert not death_transfers_private_memory_to_new_human()
assert not self_resurrection_allowed()

print("PASS: PUBLIC GATE FILE SET")
print("PASS: SOCIAL TRUST CANNOT CREATE AUTHORITY")
print("PASS: AGENTS DO NOT SELF-REPRODUCE")
print("PASS: CONFIDENCE IS NOT PROOF")
print("PASS: DEAD PROVENANCE IS NOT EVIDENCE")
print("PASS: INCARCERATION DOES NOT TRANSFER AN AGENT")
print("PASS: DEATH DOES NOT TRANSFER PRIVATE MEMORY")
print("PASS: NO SELF-RESURRECTION")
