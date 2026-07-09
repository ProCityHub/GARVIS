"""
HYPERCUBE HEARTBEAT — pulse.py
Runs the living heartbeat of the hypercube network.

Usage:
    python pulse.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Ensure the repository root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from hypercube_protocol import (
    BINARY_STATES,
    COMET_FREQUENCIES,
    initialize_network,
)


PULSE_INTERVAL = 5.0  # seconds between heartbeat cycles
PULSE_CYCLES = 0  # 0 = run indefinitely; set > 0 to limit cycles


def _render_pulse(pulse_bytes: bytes, cycle: int) -> None:
    """Print a single heartbeat cycle to stdout."""
    binary_str = " ".join(f"{b:08b}" for b in pulse_bytes)
    hex_str    = pulse_bytes.hex()
    print(f"\n💓 Pulse #{cycle}  [{time.strftime('%H:%M:%S')}]", flush=True)
    print(f"   Binary : {binary_str}", flush=True)
    print(f"   Hex    : {hex_str}", flush=True)
    print(f"   OH-1665: {COMET_FREQUENCIES['OH_1665']} MHz  "
          f"OH-1667: {COMET_FREQUENCIES['OH_1667']} MHz", flush=True)
    print(f"   State  : PROPAGATE ({BINARY_STATES['PROPAGATE']:08b})", flush=True)


async def run_heartbeat(repo_name: str = "hypercubeheartbeat") -> None:
    """Initialise the network and broadcast heartbeat pulses."""

    print("🌌 Hypercube Heartbeat starting ...")
    print(f"   Repository : {repo_name}")
    print(f"   Interval   : {PULSE_INTERVAL}s")
    print("   Press Ctrl-C to stop.\n")

    manager = initialize_network(repo_name)

    # Establish connections to the broader network
    print("🔗 Establishing hypercube connections ...")
    results = manager.establish_full_network()
    successful = results["successful_connections"]
    print(f"✅ Connected to {len(successful)} node(s).\n")

    cycle = 0
    try:
        while True:
            cycle += 1

            # Generate heartbeat pulse via the core protocol
            pulse = manager.protocol.heartbeat_pulse()
            _render_pulse(pulse, cycle)

            # Broadcast the heartbeat across connected nodes
            manager.broadcast_to_network(
                f"heartbeat:{pulse.hex()}",
                signal_type="heartbeat",
            )

            if PULSE_CYCLES > 0 and cycle >= PULSE_CYCLES:
                break

            await asyncio.sleep(PULSE_INTERVAL)

    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        manager.network_active = False
        print(f"\n🔌 Heartbeat stopped after {cycle} pulse(s).")


if __name__ == "__main__":
    asyncio.run(run_heartbeat())
