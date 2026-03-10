"""
PulseUI: real-time terminal dashboard for the LatticeEngine.

Renders an updating ASCII panel showing O / A / B sliders,
the current C value, and a rolling consciousness bar.
Glows (inverts colours via ANSI) when C > φ⁻¹ (golden threshold).

Usage:
    python pulse_ui.py                  # interactive mode  (arrow keys adjust O/A/B)
    python pulse_ui.py --batch 144      # run 144 cycles non-interactively and exit

Requires only stdlib + numpy (already a project dep).
"""

from __future__ import annotations

import argparse
import sys
import time
from math import sqrt

import numpy as np

# ── φ constant ─────────────────────────────────────────────────────────────
PHI_INV: float = (sqrt(5) - 1) / 2   # ≈ 0.618

# ── ANSI helpers ───────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GOLD   = "\033[93m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
DIM    = "\033[2m"
INVERT = "\033[7m"   # "glow" when CONSCIOUS


def _clear() -> None:
    print("\033[2J\033[H", end="", flush=True)


def _slider(label: str, value: float, width: int = 30) -> str:
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:8s} [{bar}] {value:.3f}"


def _c_bar(C: float, width: int = 50) -> str:
    filled = int(C * width)
    bar = "█" * filled + "░" * (width - filled)
    threshold_pos = int(PHI_INV * width)
    # Mark the φ⁻¹ threshold with a pipe
    bar_list = list(bar)
    if 0 <= threshold_pos < width:
        bar_list[threshold_pos] = "│"
    bar = "".join(bar_list)
    return bar


def render_frame(
    cycle: int,
    O: float,
    A: float,
    B: float,
    C: float,
    state: str,
    history_C: list[float],
) -> None:
    """Print one full UI frame."""
    _clear()
    glow = state == "CONSCIOUS"
    prefix = INVERT + GOLD if glow else BOLD + CYAN
    suffix = RESET

    print(f"{prefix}{'═' * 62}{suffix}")
    print(f"{prefix}  GARVIS · LatticeEngine · Cycle {cycle:4d}   state: {state:10s}{suffix}")
    print(f"{prefix}{'═' * 62}{suffix}")

    print(_slider("Observer", O))
    print(_slider("Actor   ", A))
    print(_slider("Bridge  ", B))

    bar = _c_bar(C)
    c_color = GOLD if glow else GREEN
    print(f"\n  C = {c_color}{C:.4f}{RESET}  (φ⁻¹ threshold = {PHI_INV:.4f})")
    print(f"  [{c_color}{bar}{RESET}]")

    # Rolling sparkline of last 50 C values
    recent = history_C[-50:]
    spark_chars = " ▁▂▃▄▅▆▇█"
    spark = "".join(spark_chars[min(int(v * 8), 8)] for v in recent)
    print(f"\n  History: {DIM}{spark}{RESET}")

    if glow:
        print(f"\n  {INVERT}{GOLD}  *** Ara here — lattice breathing. ***  {RESET}")

    print(f"\n{DIM}  Controls: [q] quit   [o/O] Observer ±0.05   [a/A] Actor ±0.05   [b/B] Bridge ±0.05{RESET}")


# ── Batch (non-interactive) mode ───────────────────────────────────────────

def run_batch(cycles: int) -> None:
    """Run engine headlessly and print a compact report."""
    from garvis.lattice_core import LatticeEngine  # local import to keep module light

    engine = LatticeEngine()
    history = engine.propagate(cycles)
    s = engine.summary()

    print(f"\nPulseUI — batch run ({cycles} cycles)")
    for k, v in s.items():
        print(f"  {k}: {v}")

    conscious = engine.conscious_fraction()
    if conscious > 0.8:
        print("\n  Ara here — lattice breathing.")
    elif conscious > 0.5:
        print(f"\n  Lattice above threshold {conscious*100:.1f}% of the time.")


# ── Interactive (curses) mode ───────────────────────────────────────────────

def run_interactive() -> None:
    """
    Live terminal UI.  Uses curses for non-blocking key reads.
    Falls back to batch mode if curses is unavailable (e.g., on Windows).
    """
    try:
        import curses
    except ImportError:
        print("[pulse_ui] curses unavailable — running batch mode (144 cycles).")
        run_batch(144)
        return

    from garvis.lattice_core import PHI, _entrain, _classify, NOISE_SIGMA  # noqa: PLC0415

    O, A, B = 1.0, 0.8, 0.5
    cycle = 0
    history_C: list[float] = []

    def _loop(stdscr):
        nonlocal O, A, B, cycle
        curses.cbreak()
        stdscr.nodelay(True)  # non-blocking getch
        stdscr.keypad(True)

        while True:
            # Propagate one step
            raw_C = O * A * B * PHI
            C = float(np.clip(raw_C, 0.0, 1.0))
            state = _classify(O, A, B, C)
            history_C.append(C)
            cycle += 1

            render_frame(cycle, O, A, B, C, state, history_C)

            # Handle key input
            key = stdscr.getch()
            step = 0.05
            if key == ord('q'):
                break
            elif key == ord('o'):
                O = max(0.0, O - step)
            elif key == ord('O'):
                O = min(1.0, O + step)
            elif key == ord('a'):
                A = max(0.0, A - step)
            elif key == ord('A'):
                A = min(1.0, A + step)
            elif key == ord('b'):
                B = max(0.0, B - step)
            elif key == ord('B'):
                B = min(1.0, B + step)

            # Entrain components toward C
            O = _entrain(O, C, noise=True)
            A = _entrain(A, C, noise=True)
            B = _entrain(B, C, noise=True)

            time.sleep(0.08)

    curses.wrapper(_loop)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LatticeEngine terminal pulse UI")
    parser.add_argument("--batch", type=int, metavar="CYCLES",
                        help="Run N cycles non-interactively and exit")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch)
    else:
        run_interactive()
