"""Close-cycle cadence and a candidate retry delay.

Fibonacci backoff is reported next to an exponential control. It is not
adopted: the pre-registered comparison against exponential delay is a later
phase. Permanent and settings failures stop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

FIBONACCI_CADENCE = (3, 5, 8, 13, 21, 34, 55, 89, 144)
FIB_MAX = 144


def fibonacci_number(count: int) -> int:
    """Return F(count), capped at 144. F(1)=1, F(2)=1, F(3)=2."""

    if count <= 0:
        return 0
    if count <= 2:
        return 1
    first, second = 1, 1
    for _ in range(3, count + 1):
        first, second = second, first + second
    return min(second, FIB_MAX)


@dataclass(frozen=True)
class ScheduleAdvice:
    action: str
    delay_units: int
    reason: str
    exponential_delay_units: int
    fibonacci_adopted: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def recommend(
    *,
    event: str,
    fail_count: int,
    fail_class: str,
    cycle_entries: int,
    fib_target: int,
) -> ScheduleAdvice:
    """Recommend wait, close-cycle, or stop. ``retry`` is not used for successes."""

    failures = max(0, int(fail_count))
    exponential = min(2 ** failures, FIB_MAX) if failures else 0
    candidate = fibonacci_number(failures)
    if fail_class in {"permanent", "settings"}:
        return ScheduleAdvice(
            action="stop",
            delay_units=0,
            reason=f"{fail_class} failure stops the cycle; it is not retried",
            exponential_delay_units=exponential,
        )
    at_cadence = (
        fib_target in FIBONACCI_CADENCE and cycle_entries >= fib_target
    )
    if at_cadence:
        return ScheduleAdvice(
            action="close-cycle",
            delay_units=0,
            reason="Fibonacci cadence reached; consolidate and score",
            exponential_delay_units=exponential,
        )
    if event in {"fail", "timeout"} and fail_class == "transient":
        return ScheduleAdvice(
            action="wait",
            delay_units=candidate,
            reason=(
                "candidate Fibonacci delay recorded beside exponential delay; "
                "Fibonacci backoff is not adopted"
            ),
            exponential_delay_units=exponential,
        )
    return ScheduleAdvice(
        action="wait",
        delay_units=0,
        reason="cycle remains open; Fibonacci cadence has not been reached",
        exponential_delay_units=exponential,
    )
