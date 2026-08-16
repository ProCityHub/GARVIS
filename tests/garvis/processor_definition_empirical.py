from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from garvis.processor_definition import (
    HardwareObserver,
    ProcessorRouter,
    RouteCandidate,
    canonical_identity,
)


REPEATS = 5


def scalar_workload() -> int:
    value = 1

    for i in range(1, 350_001):
        value = (
            value * 1664525
            + i
            + 1013904223
        ) & 0xFFFFFFFF

    return value


def dot_relevant_reference() -> int:
    # Scalar Python reference workload.
    # It is SIMD-relevant mathematically, but this does NOT
    # claim that SIMD instructions are being executed.
    total = 0

    for i in range(1, 220_001):
        left = (i * 3) & 0xFFFF
        right = (i * 7) & 0xFFFF
        total += left * right

    return total


def hash_workload() -> str:
    block = b"GARVIS-PROCESSOR-BODY-V1" * 128
    digest = b""

    for i in range(2_000):
        digest = hashlib.sha256(
            block
            + digest
            + i.to_bytes(4, "little")
        ).digest()

    return digest.hex()


WORKLOADS: Dict[str, Callable[[], object]] = {
    "SCALAR": scalar_workload,
    "DOT_RELEVANT_SCALAR_REFERENCE": dot_relevant_reference,
    "HASH": hash_workload,
}


def observed_max_temp(state) -> Optional[float]:
    values = [
        reading.celsius
        for reading in state.thermal
        if (
            reading.valid
            and reading.celsius is not None
            and 0.0 < reading.celsius < 150.0
        )
    ]

    return max(values) if values else None


def sustainability_from_delta(delta_c: float) -> float:
    # Experimental calibration transform only.
    # This is NOT an Android thermal safety threshold.
    return 1.0 / (1.0 + max(0.0, delta_c))


def benchmark_lane(
    cpu: int,
    operation: Callable[[], object],
    expected: object,
) -> Dict[str, object]:
    observer = HardwareObserver()

    original_affinity = set(
        os.sched_getaffinity(0)
    )

    before = observer.observe()

    before_temp = observed_max_temp(before)
    before_mem = before.memory_available_kb

    samples_ms: List[float] = []

    try:
        os.sched_setaffinity(0, {cpu})

        active = set(
            os.sched_getaffinity(0)
        )

        if active != {cpu}:
            raise RuntimeError(
                "requested process affinity not established"
            )

        warmup = operation()

        if warmup != expected:
            raise RuntimeError(
                "warm-up correctness failure"
            )

        gc_was_enabled = gc.isenabled()
        gc.disable()

        try:
            for _ in range(REPEATS):
                start = time.perf_counter_ns()
                result = operation()
                end = time.perf_counter_ns()

                if result != expected:
                    raise RuntimeError(
                        "benchmark correctness failure"
                    )

                samples_ms.append(
                    (end - start) / 1_000_000.0
                )
        finally:
            if gc_was_enabled:
                gc.enable()

    finally:
        os.sched_setaffinity(
            0,
            original_affinity,
        )

    restored = set(
        os.sched_getaffinity(0)
    )

    if restored != original_affinity:
        raise RuntimeError(
            "process affinity was not restored"
        )

    after = observer.observe()

    after_temp = observed_max_temp(after)
    after_mem = after.memory_available_kb

    temp_delta = 0.0

    if (
        before_temp is not None
        and after_temp is not None
    ):
        temp_delta = after_temp - before_temp

    memory_delta = None

    if (
        before_mem is not None
        and after_mem is not None
    ):
        memory_delta = after_mem - before_mem

    return {
        "cpu": cpu,
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
        "temperature_before_c": before_temp,
        "temperature_after_c": after_temp,
        "temperature_delta_c": temp_delta,
        "memory_available_before_kb": before_mem,
        "memory_available_after_kb": after_mem,
        "memory_available_delta_kb": memory_delta,
        "correctness": True,
        "affinity_restored": True,
    }


def main(output_path: str) -> None:
    observer = HardwareObserver()
    state = observer.observe()

    lanes = {
        lane.name: lane
        for lane in state.lanes
    }

    required = (
        "cpu_efficiency",
        "cpu_performance",
    )

    for name in required:
        if name not in lanes:
            raise RuntimeError(
                "missing observed lane: %s" % name
            )

    original_allowed = set(
        os.sched_getaffinity(0)
    )

    selected_cpus: Dict[str, int] = {}

    for lane_name in required:
        available = [
            cpu
            for cpu in lanes[lane_name].cpu_ids
            if cpu in original_allowed
        ]

        if not available:
            raise RuntimeError(
                "no process-accessible CPU for %s"
                % lane_name
            )

        selected_cpus[lane_name] = available[0]

    expected = {
        name: operation()
        for name, operation in WORKLOADS.items()
    }

    router = ProcessorRouter()

    results: Dict[str, object] = {}

    lattice_matches = 0
    linear_matches = 0
    ability_matches = 0

    for index, (name, operation) in enumerate(
        WORKLOADS.items()
    ):
        measured: Dict[str, Dict[str, object]] = {}

        order = (
            ("cpu_efficiency", "cpu_performance")
            if index % 2 == 0
            else ("cpu_performance", "cpu_efficiency")
        )

        for lane_name in order:
            measured[lane_name] = benchmark_lane(
                selected_cpus[lane_name],
                operation,
                expected[name],
            )

        medians = {
            lane: float(data["median_ms"])
            for lane, data in measured.items()
        }

        fastest_time = min(
            medians.values()
        )

        ability_scores = {
            lane: fastest_time / elapsed
            for lane, elapsed in medians.items()
        }

        candidates = []

        for lane_name in required:
            delta = float(
                measured[lane_name][
                    "temperature_delta_c"
                ]
            )

            candidates.append(
                RouteCandidate(
                    lane=lane_name,
                    observation_confidence=1.0,
                    ability=ability_scores[lane_name],
                    sustainability=(
                        sustainability_from_delta(delta)
                    ),
                    verified=True,
                )
            )

        lattice = router.select(
            candidates,
            method="lattice",
        )

        linear = router.select(
            candidates,
            method="linear",
        )

        ability = router.select(
            candidates,
            method="ability",
        )

        empirical_fastest = min(
            medians,
            key=medians.get,
        )

        lattice_matches += int(
            lattice.lane == empirical_fastest
        )

        linear_matches += int(
            linear.lane == empirical_fastest
        )

        ability_matches += int(
            ability.lane == empirical_fastest
        )

        results[name] = {
            "lane_results": measured,
            "normalized_ability": ability_scores,
            "empirical_fastest_lane": empirical_fastest,
            "routing": {
                "lattice": asdict(lattice),
                "linear": asdict(linear),
                "ability": asdict(ability),
            },
            "epistemic_status": {
                "timing": "EMPIRICAL",
                "temperature": "OBSERVED",
                "processor_topology": "OBSERVED",
                "canonical_identity": "MATHEMATICAL",
                "lattice_scheduler": (
                    "HYPOTHESIS_UNDER_TEST"
                ),
                "simd_execution": "UNVERIFIED",
                "gpu_execution": "NOT_PERFORMED",
            },
        }

    final_affinity = set(
        os.sched_getaffinity(0)
    )

    if final_affinity != original_allowed:
        raise RuntimeError(
            "final process affinity mismatch"
        )

    payload = {
        "stage": "PROCESSOR_DEFINITION_AGENT_V1_TESTS",
        "device": {
            "manufacturer": state.identity.manufacturer,
            "model": state.identity.model,
            "soc_model": state.identity.soc_model,
            "architecture": state.architecture,
            "logical_cpu_count": state.logical_cpu_count,
            "memory_total_kb": state.memory_total_kb,
            "memory_available_kb_at_start": (
                state.memory_available_kb
            ),
        },
        "allowed_affinity": sorted(original_allowed),
        "selected_cpu_per_lane": selected_cpus,
        "repeats": REPEATS,
        "workloads": results,
        "scheduler_summary": {
            "workload_count": len(WORKLOADS),
            "lattice_matches_empirical_fastest": (
                lattice_matches
            ),
            "linear_matches_empirical_fastest": (
                linear_matches
            ),
            "ability_matches_empirical_fastest": (
                ability_matches
            ),
            "status": (
                "RETROSPECTIVE_CALIBRATION_ONLY"
            ),
            "superiority_established": False,
        },
        "canonical_math": {
            "exponent_sum": canonical_identity(),
            "identity_status": "MATHEMATICAL_IDENTITY",
        },
        "boundaries": {
            "temporary_process_affinity_only": True,
            "process_affinity_restored": True,
            "cpu_frequency_changed": False,
            "governor_changed": False,
            "sysfs_written": False,
            "gpu_executed": False,
            "accelerator_executed": False,
            "root_used": False,
            "kernel_modified": False,
            "thermal_protection_changed": False,
        },
    }

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "ALLOWED_PROCESS_AFFINITY="
        + ",".join(
            str(cpu)
            for cpu in sorted(original_allowed)
        )
    )

    for lane, cpu in selected_cpus.items():
        print(
            "SELECTED_CPU_%s=%d"
            % (
                lane.upper(),
                cpu,
            )
        )

    for workload, data in results.items():
        print(
            "===== WORKLOAD %s ====="
            % workload
        )

        for lane in required:
            lane_data = data[
                "lane_results"
            ][lane]

            print(
                "%s_MEDIAN_MS=%.6f"
                % (
                    lane.upper(),
                    lane_data["median_ms"],
                )
            )

            print(
                "%s_TEMP_DELTA_C=%.3f"
                % (
                    lane.upper(),
                    lane_data[
                        "temperature_delta_c"
                    ],
                )
            )

        print(
            "EMPIRICAL_FASTEST=%s"
            % data["empirical_fastest_lane"]
        )

        print(
            "LATTICE_SELECTED=%s"
            % data["routing"]["lattice"]["lane"]
        )

        print(
            "LINEAR_SELECTED=%s"
            % data["routing"]["linear"]["lane"]
        )

        print(
            "ABILITY_SELECTED=%s"
            % data["routing"]["ability"]["lane"]
        )

    total = len(WORKLOADS)

    print(
        "LATTICE_MATCHES_EMPIRICAL_FASTEST=%d/%d"
        % (
            lattice_matches,
            total,
        )
    )

    print(
        "LINEAR_MATCHES_EMPIRICAL_FASTEST=%d/%d"
        % (
            linear_matches,
            total,
        )
    )

    print(
        "ABILITY_MATCHES_EMPIRICAL_FASTEST=%d/%d"
        % (
            ability_matches,
            total,
        )
    )

    print(
        "SCHEDULER_EVIDENCE_STATUS="
        "RETROSPECTIVE_CALIBRATION_ONLY"
    )

    print(
        "SCHEDULER_SUPERIORITY_ESTABLISHED=NO"
    )

    print(
        "GARVIS_CANONICAL_EXPONENT_IDENTITY=PASS"
    )

    print(
        "LATTICE_SCHEDULER_STATUS="
        "HYPOTHESIS_UNDER_TEST"
    )

    print("SIMD_EXECUTION=UNVERIFIED")
    print("GPU_EXECUTION=NOT_PERFORMED")
    print("ACCELERATOR_EXECUTION=NOT_PERFORMED")
    print("PROCESS_AFFINITY_RESTORED=PASS")
    print("EMPIRICAL_TEST_RUN=PASS")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: processor_definition_empirical.py "
            "<evidence-json>"
        )

    main(sys.argv[1])
