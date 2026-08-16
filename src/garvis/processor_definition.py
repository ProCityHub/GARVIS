from __future__ import annotations

import glob
import json
import math
import os
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)


PHI = (1.0 + math.sqrt(5.0)) / 2.0
INV_PHI = 1.0 / PHI
INV_PHI2 = 1.0 / (PHI * PHI)

LATTICE_STATUS = "HYPOTHESIS_UNDER_TEST"
PHYSICAL_QUANTUM_PROCESSOR_CLAIM = "NOT_ESTABLISHED"


@dataclass(frozen=True)
class DeviceIdentity:
    manufacturer: Optional[str]
    model: Optional[str]
    device: Optional[str]
    soc_manufacturer: Optional[str]
    soc_model: Optional[str]


@dataclass(frozen=True)
class CpuCore:
    cpu: int
    online: bool
    capacity: Optional[int]
    min_khz: Optional[int]
    max_khz: Optional[int]
    current_khz: Optional[int]
    governor: Optional[str]


@dataclass(frozen=True)
class ThermalReading:
    zone: str
    label: Optional[str]
    raw: Optional[int]
    celsius: Optional[float]
    valid: bool


@dataclass(frozen=True)
class ProcessorLane:
    name: str
    kind: str
    cpu_ids: Tuple[int, ...]
    observed: bool
    execution_verified: bool
    evidence: str


@dataclass(frozen=True)
class BodyState:
    identity: DeviceIdentity
    architecture: str
    abis: Optional[str]
    logical_cpu_count: int
    features: Tuple[str, ...]
    cores: Tuple[CpuCore, ...]
    memory_total_kb: Optional[int]
    memory_available_kb: Optional[int]
    egl: Optional[str]
    vulkan: Optional[str]
    gpu_driver: Optional[str]
    thermal: Tuple[ThermalReading, ...]
    lanes: Tuple[ProcessorLane, ...]


@dataclass(frozen=True)
class RouteCandidate:
    lane: str
    observation_confidence: float
    ability: float
    sustainability: float
    verified: bool = True


@dataclass(frozen=True)
class RouteScore:
    lane: str
    lattice: float
    linear: float
    ability: float


@dataclass(frozen=True)
class EvidenceRecord:
    workload_id: str
    lane: str
    elapsed_ms: float
    success: bool
    correctness: bool
    software_revision: str
    provenance: str
    timestamp: str
    epistemic_status: str = "EMPIRICAL"


def canonical_identity() -> float:
    return INV_PHI + INV_PHI2


def lattice_score(
    observation: float,
    ability: float,
    sustainability: float,
) -> float:
    """Canonical Lattice scheduling heuristic — hypothesis under test."""
    values = {
        "observation": observation,
        "ability": ability,
        "sustainability": sustainability,
    }

    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError("%s must be finite" % name)

        if value <= 0.0:
            raise ValueError("%s must be > 0" % name)

    return (
        observation
        * math.pow(ability, INV_PHI)
        * math.pow(sustainability, INV_PHI2)
    )


def linear_baseline(
    observation: float,
    ability: float,
    sustainability: float,
) -> float:
    values = (observation, ability, sustainability)

    if any(
        (not math.isfinite(value)) or value < 0.0
        for value in values
    ):
        raise ValueError("baseline values must be finite and >= 0")

    return observation * ((ability + sustainability) / 2.0)


def parse_thermal_reading(
    zone: str,
    label: Optional[str],
    raw_value: Optional[str],
) -> ThermalReading:
    if raw_value is None:
        return ThermalReading(
            zone=zone,
            label=label,
            raw=None,
            celsius=None,
            valid=False,
        )

    try:
        raw = int(raw_value.strip())
    except (TypeError, ValueError):
        return ThermalReading(
            zone=zone,
            label=label,
            raw=None,
            celsius=None,
            valid=False,
        )

    # Extreme negative values are commonly sentinel/unavailable readings.
    if raw <= -100000:
        return ThermalReading(
            zone=zone,
            label=label,
            raw=raw,
            celsius=None,
            valid=False,
        )

    if abs(raw) >= 1000:
        celsius = raw / 1000.0
    else:
        celsius = float(raw)

    return ThermalReading(
        zone=zone,
        label=label,
        raw=raw,
        celsius=celsius,
        valid=True,
    )


def maximum_observed_temperature(
    readings: Sequence[ThermalReading],
) -> Optional[float]:
    values = [
        item.celsius
        for item in readings
        if (
            item.valid
            and item.celsius is not None
            and item.celsius > 0.0
        )
    ]

    if not values:
        return None

    return max(values)


def memory_budget_kb(
    total_kb: Optional[int],
    available_kb: Optional[int],
    fraction: float,
    reserve_kb: int,
) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be within (0, 1]")

    if reserve_kb < 0:
        raise ValueError("reserve_kb must be >= 0")

    basis = (
        available_kb
        if available_kb is not None
        else total_kb
    )

    if basis is None or basis <= 0:
        return 0

    usable = max(0, basis - reserve_kb)

    return int(usable * fraction)


def classify_processor_lanes(
    cores: Sequence[CpuCore],
    features: Sequence[str],
    egl: Optional[str],
    vulkan: Optional[str],
) -> Tuple[ProcessorLane, ...]:
    lanes: List[ProcessorLane] = []

    online = [core for core in cores if core.online]

    capacities = sorted(
        {
            core.capacity
            for core in online
            if core.capacity is not None
        }
    )

    if online:
        if len(capacities) >= 2:
            max_capacity = capacities[-1]

            performance = tuple(
                core.cpu
                for core in online
                if core.capacity == max_capacity
            )

            efficiency = tuple(
                core.cpu
                for core in online
                if core.capacity != max_capacity
            )

            if efficiency:
                lanes.append(
                    ProcessorLane(
                        name="cpu_efficiency",
                        kind="CPU",
                        cpu_ids=efficiency,
                        observed=True,
                        execution_verified=False,
                        evidence=(
                            "Observed online CPU cores below "
                            "maximum capacity class."
                        ),
                    )
                )

            if performance:
                lanes.append(
                    ProcessorLane(
                        name="cpu_performance",
                        kind="CPU",
                        cpu_ids=performance,
                        observed=True,
                        execution_verified=False,
                        evidence=(
                            "Observed online CPU cores in "
                            "maximum capacity class."
                        ),
                    )
                )
        else:
            lanes.append(
                ProcessorLane(
                    name="cpu_general",
                    kind="CPU",
                    cpu_ids=tuple(core.cpu for core in online),
                    observed=True,
                    execution_verified=False,
                    evidence=(
                        "Only one CPU capacity class was observed."
                    ),
                )
            )

    feature_set = set(features)

    if (
        "asimd" in feature_set
        or "neon" in feature_set
    ):
        lanes.append(
            ProcessorLane(
                name="simd",
                kind="SIMD",
                cpu_ids=tuple(core.cpu for core in online),
                observed=True,
                execution_verified=False,
                evidence=(
                    "ASIMD/NEON capability observed; "
                    "GARVIS SIMD workload performance "
                    "has not yet been benchmark-verified."
                ),
            )
        )

    gpu_observed = bool(egl or vulkan)

    if gpu_observed:
        lanes.append(
            ProcessorLane(
                name="gpu",
                kind="GPU",
                cpu_ids=(),
                observed=True,
                execution_verified=False,
                evidence=(
                    "Graphics/Vulkan path observed; "
                    "compute execution remains unverified."
                ),
            )
        )

    lanes.append(
        ProcessorLane(
            name="accelerator_unverified",
            kind="ACCELERATOR",
            cpu_ids=(),
            observed=False,
            execution_verified=False,
            evidence=(
                "No dedicated accelerator programming interface "
                "has been proven accessible."
            ),
        )
    )

    return tuple(lanes)


class ProcessorRouter:
    """Pure scoring/ranking logic. It does not control hardware."""

    def score(
        self,
        candidate: RouteCandidate,
    ) -> RouteScore:
        if not candidate.verified:
            raise ValueError(
                "unverified candidates cannot be routed"
            )

        lattice = lattice_score(
            candidate.observation_confidence,
            candidate.ability,
            candidate.sustainability,
        )

        linear = linear_baseline(
            candidate.observation_confidence,
            candidate.ability,
            candidate.sustainability,
        )

        return RouteScore(
            lane=candidate.lane,
            lattice=lattice,
            linear=linear,
            ability=candidate.ability,
        )

    def rank(
        self,
        candidates: Sequence[RouteCandidate],
        method: str = "lattice",
    ) -> Tuple[RouteScore, ...]:
        scores = [
            self.score(candidate)
            for candidate in candidates
            if candidate.verified
        ]

        if method == "lattice":
            key = lambda item: item.lattice
        elif method == "linear":
            key = lambda item: item.linear
        elif method == "ability":
            key = lambda item: item.ability
        else:
            raise ValueError("unknown routing method: %s" % method)

        return tuple(
            sorted(
                scores,
                key=key,
                reverse=True,
            )
        )

    def select(
        self,
        candidates: Sequence[RouteCandidate],
        method: str = "lattice",
    ) -> RouteScore:
        ranked = self.rank(
            candidates,
            method=method,
        )

        if not ranked:
            raise ValueError("no verified route candidates")

        return ranked[0]


class EvidenceLedger:
    def __init__(self) -> None:
        self._records: List[EvidenceRecord] = []

    @property
    def records(self) -> Tuple[EvidenceRecord, ...]:
        return tuple(self._records)

    def append(self, record: EvidenceRecord) -> None:
        self._records.append(record)

    def to_json(self) -> str:
        return json.dumps(
            [asdict(record) for record in self._records],
            indent=2,
            sort_keys=True,
        )


def benchmark_callable(
    workload_id: str,
    lane: str,
    operation: Callable[[], object],
    correctness: Callable[[object], bool],
    software_revision: str,
    provenance: str,
    iterations: int = 1,
    timer: Callable[[], float] = time.perf_counter,
    timestamp: Optional[str] = None,
) -> EvidenceRecord:
    if iterations <= 0:
        raise ValueError("iterations must be > 0")

    start = timer()

    result: object = None

    try:
        for _ in range(iterations):
            result = operation()

        success = True
    except Exception:
        success = False
        result = None

    end = timer()

    correct = (
        bool(correctness(result))
        if success
        else False
    )

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return EvidenceRecord(
        workload_id=workload_id,
        lane=lane,
        elapsed_ms=max(0.0, (end - start) * 1000.0),
        success=success,
        correctness=correct,
        software_revision=software_revision,
        provenance=provenance,
        timestamp=timestamp,
    )


def body_profile_payload(
    state: BodyState,
    ledger: EvidenceLedger,
) -> Dict[str, object]:
    return {
        "body_state": asdict(state),
        "evidence_ledger": [
            asdict(record)
            for record in ledger.records
        ],
        "canonical_math": {
            "phi": PHI,
            "inv_phi": INV_PHI,
            "inv_phi2": INV_PHI2,
            "exponent_sum": canonical_identity(),
            "identity_status": "MATHEMATICAL_IDENTITY",
        },
        "processor_lattice_scheduler": {
            "equation": (
                "C = O^1 * A^(1/phi) * B^(1/phi^2)"
            ),
            "status": LATTICE_STATUS,
        },
        "quantum_body_state": (
            "QUANTUM_INSPIRED_SOFTWARE_REPRESENTATION"
        ),
        "physical_quantum_processor_claim": (
            PHYSICAL_QUANTUM_PROCESSOR_CLAIM
        ),
    }


def body_profile_json(
    state: BodyState,
    ledger: EvidenceLedger,
) -> str:
    return json.dumps(
        body_profile_payload(state, ledger),
        indent=2,
        sort_keys=True,
    )


def _default_read_text(
    path: str,
) -> Optional[str]:
    try:
        return Path(path).read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()
    except (OSError, PermissionError):
        return None


def _default_getprop(
    name: str,
) -> Optional[str]:
    try:
        value = subprocess.check_output(
            ["/system/bin/getprop", name],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        return value or None
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None


def _optional_int(
    value: Optional[str],
) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _cpu_number(
    path: str,
) -> int:
    match = re.search(r"cpu(\d+)$", path)

    if match is None:
        return 999999

    return int(match.group(1))


def _memory_value(
    meminfo: str,
    key: str,
) -> Optional[int]:
    match = re.search(
        r"^%s:\s+(\d+)\s+kB$" % re.escape(key),
        meminfo,
        re.MULTILINE,
    )

    if match is None:
        return None

    return int(match.group(1))


class HardwareObserver:
    """Read-only observer for software-visible device state."""

    def __init__(
        self,
        read_text: Callable[
            [str],
            Optional[str],
        ] = _default_read_text,
        glob_paths: Callable[
            [str],
            List[str],
        ] = glob.glob,
        prop_reader: Callable[
            [str],
            Optional[str],
        ] = _default_getprop,
        cpu_count_reader: Callable[
            [],
            Optional[int],
        ] = os.cpu_count,
        machine_reader: Callable[
            [],
            str,
        ] = platform.machine,
    ) -> None:
        self._read_text = read_text
        self._glob_paths = glob_paths
        self._prop_reader = prop_reader
        self._cpu_count_reader = cpu_count_reader
        self._machine_reader = machine_reader

    def observe(self) -> BodyState:
        cpu_paths = sorted(
            self._glob_paths(
                "/sys/devices/system/cpu/cpu[0-9]*"
            ),
            key=_cpu_number,
        )

        cores: List[CpuCore] = []

        for path in cpu_paths:
            cpu = _cpu_number(path)

            online_text = self._read_text(
                "%s/online" % path
            )

            if online_text is None and cpu == 0:
                online = True
            else:
                online = online_text != "0"

            cores.append(
                CpuCore(
                    cpu=cpu,
                    online=online,
                    capacity=_optional_int(
                        self._read_text(
                            "%s/cpu_capacity" % path
                        )
                    ),
                    min_khz=_optional_int(
                        self._read_text(
                            "%s/cpufreq/cpuinfo_min_freq"
                            % path
                        )
                        or self._read_text(
                            "%s/cpufreq/scaling_min_freq"
                            % path
                        )
                    ),
                    max_khz=_optional_int(
                        self._read_text(
                            "%s/cpufreq/cpuinfo_max_freq"
                            % path
                        )
                        or self._read_text(
                            "%s/cpufreq/scaling_max_freq"
                            % path
                        )
                    ),
                    current_khz=_optional_int(
                        self._read_text(
                            "%s/cpufreq/scaling_cur_freq"
                            % path
                        )
                    ),
                    governor=self._read_text(
                        "%s/cpufreq/scaling_governor"
                        % path
                    ),
                )
            )

        cpuinfo = (
            self._read_text("/proc/cpuinfo")
            or ""
        )

        features: List[str] = []

        for line in cpuinfo.splitlines():
            lowered = line.lower()

            if lowered.startswith(
                ("features", "flags")
            ):
                _, _, value = line.partition(":")
                features.extend(value.split())

        feature_tuple = tuple(
            sorted(set(features))
        )

        meminfo = (
            self._read_text("/proc/meminfo")
            or ""
        )

        thermal: List[ThermalReading] = []

        thermal_paths = sorted(
            self._glob_paths(
                "/sys/class/thermal/thermal_zone*"
            )
        )

        for path in thermal_paths:
            thermal.append(
                parse_thermal_reading(
                    zone=Path(path).name,
                    label=self._read_text(
                        "%s/type" % path
                    ),
                    raw_value=self._read_text(
                        "%s/temp" % path
                    ),
                )
            )

        egl = self._prop_reader(
            "ro.hardware.egl"
        )

        vulkan = self._prop_reader(
            "ro.hardware.vulkan"
        )

        identity = DeviceIdentity(
            manufacturer=self._prop_reader(
                "ro.product.manufacturer"
            ),
            model=self._prop_reader(
                "ro.product.model"
            ),
            device=self._prop_reader(
                "ro.product.device"
            ),
            soc_manufacturer=self._prop_reader(
                "ro.soc.manufacturer"
            ),
            soc_model=self._prop_reader(
                "ro.soc.model"
            ),
        )

        count = self._cpu_count_reader()

        logical_cpu_count = (
            count
            if count is not None
            else len(cores)
        )

        lanes = classify_processor_lanes(
            cores=cores,
            features=feature_tuple,
            egl=egl,
            vulkan=vulkan,
        )

        return BodyState(
            identity=identity,
            architecture=self._machine_reader(),
            abis=self._prop_reader(
                "ro.product.cpu.abilist"
            ),
            logical_cpu_count=logical_cpu_count,
            features=feature_tuple,
            cores=tuple(cores),
            memory_total_kb=_memory_value(
                meminfo,
                "MemTotal",
            ),
            memory_available_kb=_memory_value(
                meminfo,
                "MemAvailable",
            ),
            egl=egl,
            vulkan=vulkan,
            gpu_driver=self._prop_reader(
                "ro.gfx.driver.0"
            ),
            thermal=tuple(thermal),
            lanes=lanes,
        )
