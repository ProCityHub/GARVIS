from __future__ import annotations

import json
import math

import pytest

from garvis.processor_definition import (
    BodyState,
    CpuCore,
    DeviceIdentity,
    EvidenceLedger,
    EvidenceRecord,
    HardwareObserver,
    ProcessorRouter,
    RouteCandidate,
    body_profile_json,
    canonical_identity,
    classify_processor_lanes,
    lattice_score,
    maximum_observed_temperature,
    memory_budget_kb,
    parse_thermal_reading,
    benchmark_callable,
)


def core(
    cpu,
    capacity,
    max_khz,
):
    return CpuCore(
        cpu=cpu,
        online=True,
        capacity=capacity,
        min_khz=384000,
        max_khz=max_khz,
        current_khz=1000000,
        governor="walt",
    )


def test_canonical_identity_is_exact_within_float_tolerance():
    assert math.isclose(
        canonical_identity(),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_lattice_unity_state_is_one():
    assert math.isclose(
        lattice_score(1.0, 1.0, 1.0),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 1.0),
    ],
)
def test_lattice_rejects_non_positive_domain(values):
    with pytest.raises(ValueError):
        lattice_score(*values)


def test_phone_topology_forms_distinct_cpu_lanes():
    cores = [
        *[
            core(i, 765, 3532800)
            for i in range(6)
        ],
        core(6, 1024, 4473600),
        core(7, 1024, 4473600),
    ]

    lanes = classify_processor_lanes(
        cores,
        features=("asimd", "asimddp", "i8mm", "bf16"),
        egl="adreno",
        vulkan="adreno",
    )

    by_name = {
        lane.name: lane
        for lane in lanes
    }

    assert by_name["cpu_efficiency"].cpu_ids == (
        0, 1, 2, 3, 4, 5
    )

    assert by_name["cpu_performance"].cpu_ids == (
        6, 7
    )

    assert by_name["simd"].observed is True
    assert by_name["simd"].execution_verified is False

    assert by_name["gpu"].observed is True
    assert by_name["gpu"].execution_verified is False

    assert (
        by_name["accelerator_unverified"].observed
        is False
    )


def test_router_compares_lattice_and_baselines():
    candidates = [
        RouteCandidate(
            lane="cpu_efficiency",
            observation_confidence=1.0,
            ability=0.70,
            sustainability=0.95,
        ),
        RouteCandidate(
            lane="cpu_performance",
            observation_confidence=1.0,
            ability=0.95,
            sustainability=0.65,
        ),
    ]

    router = ProcessorRouter()

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

    assert lattice.lane in {
        "cpu_efficiency",
        "cpu_performance",
    }

    assert linear.lane in {
        "cpu_efficiency",
        "cpu_performance",
    }

    assert ability.lane == "cpu_performance"


def test_router_refuses_unverified_candidate():
    router = ProcessorRouter()

    with pytest.raises(ValueError):
        router.select(
            [
                RouteCandidate(
                    lane="gpu",
                    observation_confidence=1.0,
                    ability=1.0,
                    sustainability=1.0,
                    verified=False,
                )
            ]
        )


def test_thermal_parser_preserves_real_and_rejects_sentinel():
    cpu = parse_thermal_reading(
        "thermal_zone1",
        "cpu",
        "52500",
    )

    bad = parse_thermal_reading(
        "thermal_zone64",
        "mmw0",
        "-273000",
    )

    assert cpu.valid is True
    assert cpu.celsius == pytest.approx(52.5)

    assert bad.valid is False
    assert bad.celsius is None

    assert maximum_observed_temperature(
        [cpu, bad]
    ) == pytest.approx(52.5)


def test_memory_budget_is_explicit_and_bounded():
    assert memory_budget_kb(
        total_kb=10_000,
        available_kb=8_000,
        fraction=0.25,
        reserve_kb=2_000,
    ) == 1_500

    assert memory_budget_kb(
        total_kb=1_000,
        available_kb=None,
        fraction=0.5,
        reserve_kb=2_000,
    ) == 0


def test_benchmark_record_is_empirical():
    timer_values = iter(
        [10.0, 10.005]
    )

    record = benchmark_callable(
        workload_id="vector-test",
        lane="cpu_performance",
        operation=lambda: 4 + 4,
        correctness=lambda result: result == 8,
        software_revision="test",
        provenance="unit-test",
        iterations=3,
        timer=lambda: next(timer_values),
        timestamp="2026-08-16T00:00:00+00:00",
    )

    assert record.success is True
    assert record.correctness is True
    assert record.elapsed_ms == pytest.approx(5.0)
    assert record.epistemic_status == "EMPIRICAL"


def test_evidence_ledger_serializes_records():
    ledger = EvidenceLedger()

    ledger.append(
        EvidenceRecord(
            workload_id="matrix-1",
            lane="cpu_efficiency",
            elapsed_ms=2.5,
            success=True,
            correctness=True,
            software_revision="abc",
            provenance="fixture",
            timestamp="2026-08-16T00:00:00+00:00",
        )
    )

    payload = json.loads(
        ledger.to_json()
    )

    assert payload[0]["workload_id"] == "matrix-1"
    assert payload[0]["epistemic_status"] == "EMPIRICAL"


def test_body_profile_keeps_epistemic_boundary():
    state = BodyState(
        identity=DeviceIdentity(
            manufacturer="samsung",
            model="SM-S938W",
            device="pa3q",
            soc_manufacturer="QTI",
            soc_model="SM8750",
        ),
        architecture="aarch64",
        abis="arm64-v8a",
        logical_cpu_count=8,
        features=("asimd", "i8mm", "bf16"),
        cores=(),
        memory_total_kb=11379972,
        memory_available_kb=8000000,
        egl="adreno",
        vulkan="adreno",
        gpu_driver="example",
        thermal=(),
        lanes=(),
    )

    payload = json.loads(
        body_profile_json(
            state,
            EvidenceLedger(),
        )
    )

    assert (
        payload["canonical_math"]["identity_status"]
        == "MATHEMATICAL_IDENTITY"
    )

    assert math.isclose(
        payload["canonical_math"]["exponent_sum"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert (
        payload["processor_lattice_scheduler"]["status"]
        == "HYPOTHESIS_UNDER_TEST"
    )

    assert (
        payload["physical_quantum_processor_claim"]
        == "NOT_ESTABLISHED"
    )


def test_hardware_observer_builds_observed_phone_state():
    paths = [
        "/sys/devices/system/cpu/cpu0",
        "/sys/devices/system/cpu/cpu1",
        "/sys/devices/system/cpu/cpu6",
        "/sys/devices/system/cpu/cpu7",
    ]

    values = {
        "/proc/cpuinfo": (
            "Features : fp asimd asimddp i8mm bf16\n"
        ),
        "/proc/meminfo": (
            "MemTotal:       11379972 kB\n"
            "MemAvailable:    8000000 kB\n"
        ),
        "/sys/devices/system/cpu/cpu0/cpu_capacity": "765",
        "/sys/devices/system/cpu/cpu1/cpu_capacity": "765",
        "/sys/devices/system/cpu/cpu6/cpu_capacity": "1024",
        "/sys/devices/system/cpu/cpu7/cpu_capacity": "1024",
        "/sys/devices/system/cpu/cpu0/online": "1",
        "/sys/devices/system/cpu/cpu1/online": "1",
        "/sys/devices/system/cpu/cpu6/online": "1",
        "/sys/devices/system/cpu/cpu7/online": "1",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq": "3532800",
        "/sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq": "3532800",
        "/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq": "4473600",
        "/sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq": "4473600",
        "/sys/class/thermal/thermal_zone1/type": "cpu",
        "/sys/class/thermal/thermal_zone1/temp": "52500",
    }

    props = {
        "ro.product.manufacturer": "samsung",
        "ro.product.model": "SM-S938W",
        "ro.product.device": "pa3q",
        "ro.soc.manufacturer": "QTI",
        "ro.soc.model": "SM8750",
        "ro.product.cpu.abilist": "arm64-v8a",
        "ro.hardware.egl": "adreno",
        "ro.hardware.vulkan": "adreno",
        "ro.gfx.driver.0": "samsung-driver",
    }

    def fake_read(path):
        return values.get(path)

    def fake_glob(pattern):
        if "cpu[0-9]" in pattern:
            return paths

        if "thermal_zone" in pattern:
            return [
                "/sys/class/thermal/thermal_zone1"
            ]

        return []

    observer = HardwareObserver(
        read_text=fake_read,
        glob_paths=fake_glob,
        prop_reader=lambda key: props.get(key),
        cpu_count_reader=lambda: 8,
        machine_reader=lambda: "aarch64",
    )

    state = observer.observe()

    assert state.identity.model == "SM-S938W"
    assert state.identity.soc_model == "SM8750"
    assert state.architecture == "aarch64"
    assert state.logical_cpu_count == 8
    assert state.memory_total_kb == 11379972
    assert state.memory_available_kb == 8000000
    assert "asimd" in state.features
    assert "i8mm" in state.features
    assert "bf16" in state.features

    by_name = {
        lane.name: lane
        for lane in state.lanes
    }

    assert by_name["cpu_efficiency"].cpu_ids == (0, 1)
    assert by_name["cpu_performance"].cpu_ids == (6, 7)
    assert by_name["simd"].observed is True
    assert by_name["gpu"].observed is True

    assert (
        maximum_observed_temperature(
            state.thermal
        )
        == pytest.approx(52.5)
    )

def test_observation_does_not_imply_execution_verification():
    cores = [
        *[
            core(i, 765, 3532800)
            for i in range(6)
        ],
        core(6, 1024, 4473600),
        core(7, 1024, 4473600),
    ]

    lanes = classify_processor_lanes(
        cores,
        features=("asimd", "asimddp", "i8mm", "bf16"),
        egl="adreno",
        vulkan="adreno",
    )

    cpu_lanes = [
        lane
        for lane in lanes
        if lane.kind == "CPU"
    ]

    assert cpu_lanes

    assert all(
        lane.observed is True
        for lane in cpu_lanes
    )

    assert all(
        lane.execution_verified is False
        for lane in cpu_lanes
    )
