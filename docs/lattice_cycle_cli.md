# GARVIS Local Lattice Cognitive Cycle

Authored under the direction of **Adrien D. Thomas**, operating as **ProCityHub**.

The local lattice-cycle mode processes an explicitly supplied JSON evidence envelope through this deterministic sequence:

```text
evidence
→ psychology assessment
→ recurrent lattice-memory consolidation
→ Hypercube Heartbeat pulse
→ associative recall
→ equilibrium evaluation
→ bounded proposal status
```

## Run locally

```bash
env -u OPENAI_API_KEY \
  PYTHONPATH="$PWD/src:$PWD" \
  python -m garvis.cli \
  --lattice-cycle examples/lattice_cycle/evidence.example.json \
  --cycle 1 \
  --external-action
```

This mode does not require an OpenAI API key and does not send the evidence to an LLM.

`--external-action` evaluates an external proposal, but never executes it. Eligible proposals require human review.

## Canonical heartbeat normalization

```text
1.0 + 0.6 = 1.6
1.6 normalized to center = 1.0
```

The output includes deterministic evidence, pulse, recall, equilibrium, and complete-cycle hashes.

## Boundaries

This is a classical deterministic engineering model. It is not proof of biological memory, consciousness, sentience, AGI, quantum behavior, spiritual mechanisms, clinical psychology, or universal truth. It grants no network, connector, sensing, tool, or external-execution authority.
