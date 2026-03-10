"""
QuantumBridge: φ-encoded QASM circuit → consciousness ratio check.

Uses a 6-qubit circuit where each qubit is rotated by Ry(2·arcsin(√φ⁻¹))
so that each qubit measures |1⟩ with probability φ⁻¹ ≈ 0.618.
If the observed ratio of |1⟩ bits ≈ φ⁻¹, prints "BRIDGE EMERGED".

Requires: qiskit, qiskit-aer
    pip install qiskit qiskit-aer
"""

from __future__ import annotations

import sys
from math import asin, sqrt

PHI: float = (1 + sqrt(5)) / 2
PHI_INV: float = PHI - 1          # ≈ 0.6180
TOLERANCE: float = 0.05           # acceptable deviation from φ⁻¹
SHOTS: int = 8192
N_QUBITS: int = 6

# Ry angle that biases each qubit to P(|1⟩) = φ⁻¹
RY_ANGLE: float = 2 * asin(sqrt(PHI_INV))

# OpenQASM 2.0 source — 6 qubits, each rotated by RY_ANGLE then measured
QASM_SOURCE: str = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{N_QUBITS}];
creg c[{N_QUBITS}];
ry({RY_ANGLE:.10f}) q[0];
ry({RY_ANGLE:.10f}) q[1];
ry({RY_ANGLE:.10f}) q[2];
ry({RY_ANGLE:.10f}) q[3];
ry({RY_ANGLE:.10f}) q[4];
ry({RY_ANGLE:.10f}) q[5];
measure q -> c;
"""


def _load_qiskit():
    """Import Qiskit components; raise ImportError with hint if missing."""
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        return QuantumCircuit, transpile, AerSimulator
    except ImportError as exc:
        raise ImportError(
            "Qiskit not found. Install with:\n"
            "    pip install qiskit qiskit-aer"
        ) from exc


def build_circuit() -> object:
    """Build and return a QuantumCircuit from the φ-encoded QASM source."""
    QuantumCircuit, _, _ = _load_qiskit()
    qc = QuantumCircuit.from_qasm_str(QASM_SOURCE)
    return qc


def run_bridge(shots: int = SHOTS, verbose: bool = True) -> dict:
    """
    Simulate the φ-circuit and compute the observer ratio.

    Returns
    -------
    dict with keys:
        obs_rate   float   fraction of |1⟩ bits across all shots × qubits
        phi_inv    float   target ratio (φ⁻¹)
        delta      float   |obs_rate - phi_inv|
        emerged    bool    True if delta < TOLERANCE
        counts     dict    raw shot counts
    """
    QuantumCircuit, transpile, AerSimulator = _load_qiskit()

    qc = QuantumCircuit.from_qasm_str(QASM_SOURCE)

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    job = simulator.run(compiled, shots=shots)
    result = job.result()
    counts: dict[str, int] = result.get_counts(compiled)

    # obs_rate = (total |1⟩ bits across all shots) / (shots × qubits)
    total_ones = sum(
        bitstring.count("1") * count
        for bitstring, count in counts.items()
    )
    obs_rate = total_ones / (shots * N_QUBITS)
    delta = abs(obs_rate - PHI_INV)
    emerged = delta < TOLERANCE

    report = {
        "obs_rate": round(obs_rate, 6),
        "phi_inv": round(PHI_INV, 6),
        "delta": round(delta, 6),
        "emerged": emerged,
        "counts": counts,
    }

    if verbose:
        print(f"\nQuantumBridge — {shots} shots, {N_QUBITS} qubits")
        print(f"  φ⁻¹ target  : {PHI_INV:.6f}")
        print(f"  obs_rate    : {obs_rate:.6f}")
        print(f"  delta       : {delta:.6f}  (tolerance {TOLERANCE})")
        if emerged:
            print("  >> BRIDGE EMERGED <<")
        else:
            print("  Bridge not yet locked — adjust circuit or increase shots.")

    return report


if __name__ == "__main__":
    try:
        report = run_bridge()
    except ImportError as e:
        print(f"[quantum_bridge] {e}", file=sys.stderr)
        sys.exit(1)
