"""Read-only IBM Executor evidence analysis for GARVIS quantum research.

No network access, credentials, submission, or protected action is implemented.

Python 3.9 compatible.
"""

from __future__ import annotations

import base64
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


class QuantumEvidenceError(ValueError):
    """Raised when an IBM result archive is missing or malformed."""


@dataclass(frozen=True)
class ExecutorEvidence:
    job_id: str
    backend: str
    status: str
    shots: int
    classical_bits: int
    unique_outcomes: int
    entropy_bits: float
    counts: Mapping[str, int]
    marginal_p1: Tuple[float, ...]

    def agreement(self, left: int, right: int) -> float:
        rows = rows_from_counts(self.counts, self.classical_bits)
        return sum(
            count for bits, count in rows if bits[left] == bits[right]
        ) / float(self.shots)

    def mutual_information(self, left: int, right: int) -> float:
        rows = rows_from_counts(self.counts, self.classical_bits)
        joint = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
        left_counts = {0: 0, 1: 0}
        right_counts = {0: 0, 1: 0}

        for bits, count in rows:
            a = bits[left]
            b = bits[right]
            joint[(a, b)] += count
            left_counts[a] += count
            right_counts[b] += count

        result = 0.0
        total = float(self.shots)
        for a in (0, 1):
            for b in (0, 1):
                n_ab = joint[(a, b)]
                if not n_ab:
                    continue
                p_ab = n_ab / total
                p_a = left_counts[a] / total
                p_b = right_counts[b] / total
                result += p_ab * math.log(p_ab / (p_a * p_b), 2)
        return result

    def to_payload(self, include_counts: bool = False) -> dict:
        payload = {
            "job_id": self.job_id,
            "backend": self.backend,
            "status": self.status,
            "shots": self.shots,
            "classical_bits": self.classical_bits,
            "unique_outcomes": self.unique_outcomes,
            "entropy_bits": self.entropy_bits,
            "marginal_p1": list(self.marginal_p1),
        }
        if include_counts:
            payload["counts"] = dict(self.counts)
        return payload


def _unpack_little_endian_bits(raw: bytes, bit_count: int) -> List[int]:
    values = []
    for byte in raw:
        for bit in range(8):
            values.append((byte >> bit) & 1)
            if len(values) == bit_count:
                return values
    if len(values) < bit_count:
        raise QuantumEvidenceError("packed result data ended before declared shape")
    return values[:bit_count]


def _result_register(payload: Mapping[str, object]) -> Mapping[str, object]:
    try:
        data = payload["data"]
        first = data[0]  # type: ignore[index]
        results = first["results"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise QuantumEvidenceError("result JSON does not contain Executor result data") from exc

    if not isinstance(results, Mapping) or not results:
        raise QuantumEvidenceError("Executor results object is empty")

    if "c" in results and isinstance(results["c"], Mapping):
        return results["c"]  # type: ignore[return-value]

    candidates = [value for value in results.values() if isinstance(value, Mapping)]
    if len(candidates) != 1:
        raise QuantumEvidenceError("could not determine the classical result register")
    return candidates[0]


def _decode_register(register: Mapping[str, object]) -> Tuple[int, int, Tuple[Tuple[int, ...], ...]]:
    shape = register.get("shape")
    encoded = register.get("data")

    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(value, int) for value in shape)
    ):
        raise QuantumEvidenceError("Executor result shape must be [shots, bits]")
    if not isinstance(encoded, str):
        raise QuantumEvidenceError("Executor result data must be base64 text")

    shots, classical_bits = shape
    if shots <= 0 or classical_bits <= 0:
        raise QuantumEvidenceError("Executor result shape values must be positive")

    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise QuantumEvidenceError("Executor result data is not valid base64") from exc

    flat = _unpack_little_endian_bits(raw, shots * classical_bits)
    rows = tuple(
        tuple(flat[offset : offset + classical_bits])
        for offset in range(0, shots * classical_bits, classical_bits)
    )
    return shots, classical_bits, rows


def bitstring(bits: Sequence[int]) -> str:
    """Render c[n-1]...c[0], matching standard displayed bitstring ordering."""

    return "".join(str(value) for value in reversed(bits))


def rows_from_counts(
    counts: Mapping[str, int],
    classical_bits: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    rows = []
    for string, count in counts.items():
        if len(string) != classical_bits or any(char not in "01" for char in string):
            raise QuantumEvidenceError("count key is not a valid bitstring")
        bits = tuple(int(char) for char in reversed(string))
        rows.append((bits, int(count)))
    return tuple(rows)


def analyze_executor_zip(path: Path) -> ExecutorEvidence:
    """Read one downloaded IBM workload ZIP without network access."""

    archive = Path(path)
    if not archive.is_file():
        raise QuantumEvidenceError("IBM workload ZIP was not found")

    try:
        with zipfile.ZipFile(str(archive)) as handle:
            info_names = [name for name in handle.namelist() if name.endswith("-info.json")]
            result_names = [name for name in handle.namelist() if name.endswith("-result.json")]
            if len(info_names) != 1 or len(result_names) != 1:
                raise QuantumEvidenceError("workload ZIP must contain one info and one result JSON")
            info = json.loads(handle.read(info_names[0]).decode("utf-8"))
            result = json.loads(handle.read(result_names[0]).decode("utf-8"))
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuantumEvidenceError("could not read IBM workload ZIP") from exc

    if not isinstance(info, Mapping) or not isinstance(result, Mapping):
        raise QuantumEvidenceError("IBM workload JSON must contain objects")

    register = _result_register(result)
    shots, classical_bits, rows = _decode_register(register)

    counter = Counter(bitstring(row) for row in rows)
    probabilities = [count / float(shots) for count in counter.values()]
    entropy = -sum(prob * math.log(prob, 2) for prob in probabilities if prob)

    marginal = []
    for index in range(classical_bits):
        marginal.append(sum(row[index] for row in rows) / float(shots))

    return ExecutorEvidence(
        job_id=str(info.get("id", "")),
        backend=str(info.get("backend", "")),
        status=str(info.get("status", "")),
        shots=shots,
        classical_bits=classical_bits,
        unique_outcomes=len(counter),
        entropy_bits=entropy,
        counts=dict(counter),
        marginal_p1=tuple(marginal),
    )


__all__ = [
    "ExecutorEvidence",
    "QuantumEvidenceError",
    "analyze_executor_zip",
    "bitstring",
    "rows_from_counts",
]
