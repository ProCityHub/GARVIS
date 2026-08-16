"""Genesis X Double-Slit Brain Test Model

Creator / conceptual architect: Adrien D. Thomas / ProCityHub

Purpose
-------
A falsification-oriented test harness for comparing:

1. Standard two-path quantum-mechanical visibility

       V_QM = 2 O sqrt(A B) / (A + B)

2. Genesis X / Canonical Lattice interference-discriminator hypothesis

       V_phi = O (A^alpha B^beta + A^beta B^alpha) / (A + B)

   where

       phi   = (1 + sqrt(5)) / 2
       alpha = 1 / phi
       beta  = 1 / phi^2
       alpha + beta = 1

The Canonical Lattice Law remains:

       C = O^1 A^(1/phi) B^(1/phi^2)

Status
------
- Canonical Lattice Law: HYPOTHESIS UNDER TEST
- Interference discriminator: HYPOTHESIS UNDER TEST
- Empirical status of this discriminator: NOT_EVALUATED until authentic
  asymmetric-path data are tested.
- This software does not establish AGI, consciousness, or new physics.

Engineering boundary
--------------------
The OAB / Identity-Cylinder terms in this file are an engineering representation
for organizing prediction -> observation -> error. They are not physical claims
about the double-slit apparatus itself.

Run
---
    python genesis_x_double_slit_brain.py

CSV evaluation
--------------
Expected columns:

    A,B,O,delta_theta,measured_intensity

Then:

    python genesis_x_double_slit_brain.py data.csv
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


CREATOR = "Adrien D. Thomas"
ORGANIZATION = "ProCityHub"
SYSTEM = "GARVIS / Genesis X"
MODEL_NAME = "Double-Slit OAB Falsification Brain"

PHI = (1.0 + math.sqrt(5.0)) / 2.0
ALPHA = 1.0 / PHI
BETA = 1.0 / (PHI * PHI)

EPS = 1e-12


class EmpiricalStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    DIRECTIONAL_SUPPORT_QM = "DIRECTIONAL_SUPPORT_QM"
    DIRECTIONAL_SUPPORT_PHI = "DIRECTIONAL_SUPPORT_PHI"
    INDETERMINATE = "INDETERMINATE"


class DiscriminationClass(str, Enum):
    NON_DISCRIMINATING_SYMMETRIC = "NON_DISCRIMINATING_SYMMETRIC"
    DISCRIMINATING_ASYMMETRIC = "DISCRIMINATING_ASYMMETRIC"


@dataclass(frozen=True)
class EpistemicBoundary:
    canonical_lattice_law: str = "HYPOTHESIS_UNDER_TEST"
    interference_discriminator: str = "HYPOTHESIS_UNDER_TEST"
    discriminator_empirical_status: str = "NOT_EVALUATED"
    agi: str = "NOT_ESTABLISHED"
    consciousness: str = "NOT_ESTABLISHED"
    new_physics: str = "NOT_ESTABLISHED"
    note: str = (
        "Mathematical consistency and software PASS do not establish physical truth."
    )


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def validate_inputs(A: float, B: float, O: float) -> tuple[float, float, float]:
    A = _finite_nonnegative("A", A)
    B = _finite_nonnegative("B", B)
    O = _finite_nonnegative("O", O)

    if O > 1.0 + EPS:
        raise ValueError("O must be in [0, 1] for visibility interpretation")

    if A + B <= EPS:
        raise ValueError("A + B must be > 0")

    return A, B, min(O, 1.0)


def exponent_identity_error() -> float:
    """Numerical error in alpha + beta = 1."""
    return abs((ALPHA + BETA) - 1.0)


def canonical_lattice_score(O: float, A: float, B: float) -> float:
    """Canonical Lattice Law: C = O^1 A^alpha B^beta."""
    A, B, O = validate_inputs(A, B, O)

    if A <= EPS or B <= EPS or O <= EPS:
        return 0.0

    return O * (A ** ALPHA) * (B ** BETA)


def qm_visibility(A: float, B: float, O: float = 1.0) -> float:
    """
    Standard two-path visibility with coherence factor O:

        V_QM = 2 O sqrt(A B) / (A + B)
    """
    A, B, O = validate_inputs(A, B, O)
    return (2.0 * O * math.sqrt(A * B)) / (A + B)


def phi_visibility(A: float, B: float, O: float = 1.0) -> float:
    """
    Genesis X interference discriminator hypothesis:

        V_phi = O(A^alpha B^beta + A^beta B^alpha)/(A+B)
    """
    A, B, O = validate_inputs(A, B, O)

    term_1 = (A ** ALPHA) * (B ** BETA) if A > 0 and B > 0 else 0.0
    term_2 = (A ** BETA) * (B ** ALPHA) if A > 0 and B > 0 else 0.0

    return O * (term_1 + term_2) / (A + B)


def phi_visibility_cosh_form(A: float, B: float, O: float = 1.0) -> float:
    """
    Equivalent form for A,B > 0:

        V_phi =
        [2 O sqrt(A B)/(A+B)]
        cosh(((alpha-beta)/2) ln(A/B))
    """
    A, B, O = validate_inputs(A, B, O)

    if A <= EPS or B <= EPS:
        return 0.0

    base = qm_visibility(A, B, O)
    factor = math.cosh(((ALPHA - BETA) / 2.0) * math.log(A / B))
    return base * factor


def classify_path_pair(
    A: float,
    B: float,
    *,
    relative_tolerance: float = 1e-9,
) -> DiscriminationClass:
    """
    Equal/symmetric path intensities cannot discriminate V_QM from V_phi.
    Unequal intensities can.
    """
    A = _finite_nonnegative("A", A)
    B = _finite_nonnegative("B", B)

    scale = max(abs(A), abs(B), 1.0)
    if abs(A - B) <= relative_tolerance * scale:
        return DiscriminationClass.NON_DISCRIMINATING_SYMMETRIC

    return DiscriminationClass.DISCRIMINATING_ASYMMETRIC


def predicted_intensity_qm(
    A: float,
    B: float,
    O: float,
    delta_theta: float,
) -> float:
    """
    Standard two-path intensity:

        I_QM = A + B + 2 O sqrt(A B) cos(delta_theta)
    """
    A, B, O = validate_inputs(A, B, O)
    delta_theta = float(delta_theta)

    return (
        A
        + B
        + 2.0 * O * math.sqrt(A * B) * math.cos(delta_theta)
    )


def phi_cross_amplitude(A: float, B: float) -> float:
    """G_phi(A,B) = 0.5(A^alpha B^beta + A^beta B^alpha)."""
    A = _finite_nonnegative("A", A)
    B = _finite_nonnegative("B", B)

    if A <= EPS or B <= EPS:
        return 0.0

    return 0.5 * (
        (A ** ALPHA) * (B ** BETA)
        + (A ** BETA) * (B ** ALPHA)
    )


def predicted_intensity_phi(
    A: float,
    B: float,
    O: float,
    delta_theta: float,
) -> float:
    """
    Genesis X discriminator hypothesis:

        I_phi = A + B + 2 O G_phi(A,B) cos(delta_theta)

    where

        G_phi = 0.5(A^alpha B^beta + A^beta B^alpha)
    """
    A, B, O = validate_inputs(A, B, O)
    delta_theta = float(delta_theta)

    return (
        A
        + B
        + 2.0 * O * phi_cross_amplitude(A, B) * math.cos(delta_theta)
    )


def measured_visibility(i_max: float, i_min: float) -> float:
    """Visibility from observed extrema."""
    i_max = _finite_nonnegative("i_max", i_max)
    i_min = _finite_nonnegative("i_min", i_min)

    denominator = i_max + i_min
    if denominator <= EPS:
        raise ValueError("i_max + i_min must be > 0")

    return (i_max - i_min) / denominator


@dataclass(frozen=True)
class DoubleSlitObservation:
    A: float
    B: float
    O: float
    delta_theta: float
    measured_intensity: float
    label: str = ""

    def validated(self) -> "DoubleSlitObservation":
        A, B, O = validate_inputs(self.A, self.B, self.O)
        measured = _finite_nonnegative(
            "measured_intensity",
            self.measured_intensity,
        )

        return DoubleSlitObservation(
            A=A,
            B=B,
            O=O,
            delta_theta=float(self.delta_theta),
            measured_intensity=measured,
            label=self.label,
        )


@dataclass(frozen=True)
class PredictionResidual:
    label: str
    A: float
    B: float
    O: float
    delta_theta: float
    measured_intensity: float

    qm_prediction: float
    phi_prediction: float

    qm_residual: float
    phi_residual: float

    path_class: DiscriminationClass

    @property
    def qm_abs_error(self) -> float:
        return abs(self.qm_residual)

    @property
    def phi_abs_error(self) -> float:
        return abs(self.phi_residual)


def evaluate_observation(observation: DoubleSlitObservation) -> PredictionResidual:
    observation = observation.validated()

    qm = predicted_intensity_qm(
        observation.A,
        observation.B,
        observation.O,
        observation.delta_theta,
    )

    phi = predicted_intensity_phi(
        observation.A,
        observation.B,
        observation.O,
        observation.delta_theta,
    )

    return PredictionResidual(
        label=observation.label,
        A=observation.A,
        B=observation.B,
        O=observation.O,
        delta_theta=observation.delta_theta,
        measured_intensity=observation.measured_intensity,
        qm_prediction=qm,
        phi_prediction=phi,
        qm_residual=observation.measured_intensity - qm,
        phi_residual=observation.measured_intensity - phi,
        path_class=classify_path_pair(observation.A, observation.B),
    )


@dataclass(frozen=True)
class ModelScore:
    n_total: int
    n_asymmetric: int

    qm_mae: float | None
    phi_mae: float | None

    qm_rmse: float | None
    phi_rmse: float | None

    delta_mae_phi_minus_qm: float | None
    delta_rmse_phi_minus_qm: float | None

    status: EmpiricalStatus

    note: str


def _rmse(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def score_dataset(
    observations: Iterable[DoubleSlitObservation],
    *,
    asymmetric_only: bool = True,
    tie_tolerance: float = 1e-12,
) -> tuple[ModelScore, list[PredictionResidual]]:
    """
    Compare out-of-sample-style residuals.

    By default, only asymmetric A != B records contribute to discrimination
    statistics because symmetric records cannot distinguish the models.
    """
    residuals = [evaluate_observation(item) for item in observations]

    usable = [
        row
        for row in residuals
        if (
            not asymmetric_only
            or row.path_class is DiscriminationClass.DISCRIMINATING_ASYMMETRIC
        )
    ]

    if not usable:
        return (
            ModelScore(
                n_total=len(residuals),
                n_asymmetric=0,
                qm_mae=None,
                phi_mae=None,
                qm_rmse=None,
                phi_rmse=None,
                delta_mae_phi_minus_qm=None,
                delta_rmse_phi_minus_qm=None,
                status=EmpiricalStatus.NOT_EVALUATED,
                note="No asymmetric observations available for discrimination.",
            ),
            residuals,
        )

    qm_errors = [row.qm_residual for row in usable]
    phi_errors = [row.phi_residual for row in usable]

    qm_mae = statistics.fmean(abs(value) for value in qm_errors)
    phi_mae = statistics.fmean(abs(value) for value in phi_errors)

    qm_rmse = _rmse(qm_errors)
    phi_rmse = _rmse(phi_errors)

    delta_mae = phi_mae - qm_mae
    delta_rmse = phi_rmse - qm_rmse

    if (
        abs(delta_mae) <= tie_tolerance
        and abs(delta_rmse) <= tie_tolerance
    ):
        status = EmpiricalStatus.INDETERMINATE
        note = "Residual metrics are tied within tolerance."

    elif delta_mae < 0.0 and delta_rmse < 0.0:
        status = EmpiricalStatus.DIRECTIONAL_SUPPORT_PHI
        note = (
            "V_phi has smaller MAE and RMSE on this dataset. "
            "This is directional evidence only, not proof."
        )

    elif delta_mae > 0.0 and delta_rmse > 0.0:
        status = EmpiricalStatus.DIRECTIONAL_SUPPORT_QM
        note = (
            "Standard QM has smaller MAE and RMSE on this dataset. "
            "This is directional evidence against the V_phi discriminator."
        )

    else:
        status = EmpiricalStatus.INDETERMINATE
        note = "MAE and RMSE disagree; result is mixed."

    return (
        ModelScore(
            n_total=len(residuals),
            n_asymmetric=len(usable),
            qm_mae=qm_mae,
            phi_mae=phi_mae,
            qm_rmse=qm_rmse,
            phi_rmse=phi_rmse,
            delta_mae_phi_minus_qm=delta_mae,
            delta_rmse_phi_minus_qm=delta_rmse,
            status=status,
            note=note,
        ),
        residuals,
    )


@dataclass(frozen=True)
class OABDoubleSlitCoordinate:
    """
    OAB engineering coordinate for one double-slit observation.

    observer_axis:
        observed intensity normalized by baseline A+B

    actor_axis:
        phase-control coordinate cos(delta_theta)

    bridge_axis:
        asymmetry coordinate log(A/B), or 0 for unavailable ratio

    zero_reference:
        prediction == observation in selected model space
    """
    observer_axis: float
    actor_axis: float
    bridge_axis: float
    zero_reference: float = 0.0


def oab_coordinate(observation: DoubleSlitObservation) -> OABDoubleSlitCoordinate:
    observation = observation.validated()

    baseline = observation.A + observation.B
    observer_axis = observation.measured_intensity / baseline
    actor_axis = math.cos(observation.delta_theta)

    if observation.A > EPS and observation.B > EPS:
        bridge_axis = math.log(observation.A / observation.B)
    else:
        bridge_axis = 0.0

    return OABDoubleSlitCoordinate(
        observer_axis=observer_axis,
        actor_axis=actor_axis,
        bridge_axis=bridge_axis,
    )


@dataclass(frozen=True)
class IdentityCylinderPoint:
    step: int
    z_t: float
    theta_t: float
    tau_t: int
    radius_qm: float
    radius_phi: float
    preferred_model_at_step: str


@dataclass
class DoubleSlitBrain:
    """
    Stateful test brain.

    The "Identity Cylinder" here is an engineering ledger of prediction error
    across observations.

    radius_qm  = |measured - I_QM|
    radius_phi = |measured - I_phi|

    It does not represent consciousness or ontological identity.
    """

    points: list[IdentityCylinderPoint] = field(default_factory=list)
    residuals: list[PredictionResidual] = field(default_factory=list)

    def observe(self, observation: DoubleSlitObservation) -> IdentityCylinderPoint:
        residual = evaluate_observation(observation)
        self.residuals.append(residual)

        step = len(self.points) + 1

        error_vector_x = residual.qm_residual
        error_vector_y = residual.phi_residual

        theta = math.atan2(error_vector_y, error_vector_x)

        if residual.qm_abs_error < residual.phi_abs_error:
            preferred = "QM"
        elif residual.phi_abs_error < residual.qm_abs_error:
            preferred = "PHI"
        else:
            preferred = "TIE"

        point = IdentityCylinderPoint(
            step=step,
            z_t=float(step),
            theta_t=theta,
            tau_t=step,
            radius_qm=residual.qm_abs_error,
            radius_phi=residual.phi_abs_error,
            preferred_model_at_step=preferred,
        )

        self.points.append(point)
        return point

    def score(self, *, asymmetric_only: bool = True) -> ModelScore:
        observations = [
            DoubleSlitObservation(
                A=row.A,
                B=row.B,
                O=row.O,
                delta_theta=row.delta_theta,
                measured_intensity=row.measured_intensity,
                label=row.label,
            )
            for row in self.residuals
        ]

        score, _ = score_dataset(
            observations,
            asymmetric_only=asymmetric_only,
        )
        return score

    def summary(self) -> dict:
        return {
            "creator": CREATOR,
            "organization": ORGANIZATION,
            "system": SYSTEM,
            "model": MODEL_NAME,
            "steps": len(self.points),
            "score_asymmetric_only": asdict(self.score()),
            "epistemic_boundary": asdict(EpistemicBoundary()),
        }


def load_csv(path: str | Path) -> list[DoubleSlitObservation]:
    path = Path(path)
    rows: list[DoubleSlitObservation] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {
            "A",
            "B",
            "O",
            "delta_theta",
            "measured_intensity",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "CSV missing columns: " + ", ".join(sorted(missing))
            )

        for index, row in enumerate(reader, start=1):
            rows.append(
                DoubleSlitObservation(
                    A=float(row["A"]),
                    B=float(row["B"]),
                    O=float(row["O"]),
                    delta_theta=float(row["delta_theta"]),
                    measured_intensity=float(row["measured_intensity"]),
                    label=row.get("label", "") or f"row-{index}",
                )
            )

    return rows


def synthetic_qm_dataset() -> list[DoubleSlitObservation]:
    """
    Synthetic control generated from standard QM.

    This is software-validation data only.
    It is NOT physical experimental evidence.
    """
    cases = [
        (1.0, 1.0, 0.90),
        (1.0, 0.70, 0.90),
        (1.0, 0.40, 0.90),
        (1.0, 0.20, 0.90),
        (0.30, 1.0, 0.75),
    ]

    phases = [
        0.0,
        math.pi / 4.0,
        math.pi / 2.0,
        3.0 * math.pi / 4.0,
        math.pi,
    ]

    rows: list[DoubleSlitObservation] = []

    for case_index, (A, B, O) in enumerate(cases):
        for phase_index, phase in enumerate(phases):
            rows.append(
                DoubleSlitObservation(
                    A=A,
                    B=B,
                    O=O,
                    delta_theta=phase,
                    measured_intensity=predicted_intensity_qm(
                        A,
                        B,
                        O,
                        phase,
                    ),
                    label=f"synthetic-qm-{case_index}-{phase_index}",
                )
            )

    return rows


def synthetic_phi_dataset() -> list[DoubleSlitObservation]:
    """
    Synthetic control generated from the V_phi hypothesis.

    This is software-validation data only.
    It is NOT physical experimental evidence.
    """
    cases = [
        (1.0, 1.0, 0.90),
        (1.0, 0.70, 0.90),
        (1.0, 0.40, 0.90),
        (1.0, 0.20, 0.90),
        (0.30, 1.0, 0.75),
    ]

    phases = [
        0.0,
        math.pi / 4.0,
        math.pi / 2.0,
        3.0 * math.pi / 4.0,
        math.pi,
    ]

    rows: list[DoubleSlitObservation] = []

    for case_index, (A, B, O) in enumerate(cases):
        for phase_index, phase in enumerate(phases):
            rows.append(
                DoubleSlitObservation(
                    A=A,
                    B=B,
                    O=O,
                    delta_theta=phase,
                    measured_intensity=predicted_intensity_phi(
                        A,
                        B,
                        O,
                        phase,
                    ),
                    label=f"synthetic-phi-{case_index}-{phase_index}",
                )
            )

    return rows


def self_test() -> dict:
    # Mathematical identity
    assert exponent_identity_error() < 1e-12

    # Visibility bounds and symmetric equality
    for O in (0.0, 0.25, 0.5, 1.0):
        v_qm = qm_visibility(1.0, 1.0, O)
        v_phi = phi_visibility(1.0, 1.0, O)

        assert abs(v_qm - O) < 1e-12
        assert abs(v_phi - O) < 1e-12
        assert abs(v_qm - v_phi) < 1e-12

    # Cosh form equivalence
    for A, B, O in (
        (1.0, 0.8, 1.0),
        (1.0, 0.4, 0.9),
        (0.2, 1.0, 0.7),
    ):
        direct = phi_visibility(A, B, O)
        cosh_form = phi_visibility_cosh_form(A, B, O)
        assert abs(direct - cosh_form) < 1e-12

    # For equal coherence and positive unequal paths, this hypothesis predicts
    # visibility at least as large as the standard-QM expression.
    for A, B in (
        (1.0, 0.8),
        (1.0, 0.4),
        (1.0, 0.1),
        (0.2, 1.0),
    ):
        assert phi_visibility(A, B, 1.0) + 1e-12 >= qm_visibility(A, B, 1.0)

    # Symmetric data cannot discriminate.
    assert (
        classify_path_pair(1.0, 1.0)
        is DiscriminationClass.NON_DISCRIMINATING_SYMMETRIC
    )

    # Asymmetric data can discriminate.
    assert (
        classify_path_pair(1.0, 0.4)
        is DiscriminationClass.DISCRIMINATING_ASYMMETRIC
    )

    # Synthetic QM control should directionally select QM.
    qm_score, _ = score_dataset(synthetic_qm_dataset())
    assert qm_score.status is EmpiricalStatus.DIRECTIONAL_SUPPORT_QM

    # Synthetic phi control should directionally select phi.
    phi_score, _ = score_dataset(synthetic_phi_dataset())
    assert phi_score.status is EmpiricalStatus.DIRECTIONAL_SUPPORT_PHI

    # OAB / cylinder software sanity check.
    brain = DoubleSlitBrain()

    for observation in synthetic_qm_dataset()[:8]:
        brain.observe(observation)

    assert len(brain.points) == 8

    return {
        "SELF_TEST": "PASS",
        "CREATOR": CREATOR,
        "SYSTEM": SYSTEM,
        "PHI": PHI,
        "ALPHA": ALPHA,
        "BETA": BETA,
        "ALPHA_PLUS_BETA": ALPHA + BETA,
        "CANONICAL_LATTICE_LAW": "C=O^1*A^(1/phi)*B^(1/phi^2)",
        "INTERFERENCE_DISCRIMINATOR": (
            "V_phi=O*(A^alpha*B^beta + A^beta*B^alpha)/(A+B)"
        ),
        "QM_BASELINE": "V_QM=2*O*sqrt(A*B)/(A+B)",
        "SYMMETRIC_PATHS": "NON_DISCRIMINATING",
        "ASYMMETRIC_PATHS": "DISCRIMINATING",
        "EMPIRICAL_STATUS": "NOT_EVALUATED",
        "EPISTEMIC_BOUNDARY": asdict(EpistemicBoundary()),
    }


def run_csv(path: str | Path) -> dict:
    observations = load_csv(path)

    brain = DoubleSlitBrain()

    for observation in observations:
        brain.observe(observation)

    score = brain.score(asymmetric_only=True)

    return {
        "file": str(path),
        "score": asdict(score),
        "brain": brain.summary(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0

    result = run_csv(argv[0])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
