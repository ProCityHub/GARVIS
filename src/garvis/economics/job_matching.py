"""Deterministic job matching for ProCityHub candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateProfile:
    skills: tuple[str, ...]
    minimum_hourly_rate: float
    maximum_distance_km: float

    def __post_init__(self) -> None:
        if self.minimum_hourly_rate < 0.0:
            raise ValueError("minimum_hourly_rate must not be negative")
        if self.maximum_distance_km < 0.0:
            raise ValueError("maximum_distance_km must not be negative")


@dataclass(frozen=True)
class JobPosting:
    title: str
    required_skills: tuple[str, ...]
    hourly_rate: float
    distance_km: float
    verified_employer: bool


@dataclass(frozen=True)
class JobMatch:
    score: float
    skill_match: float
    pay_match: float
    distance_match: float
    verified_employer: bool

    @property
    def recommendation(self) -> str:
        if not self.verified_employer:
            return "verify_employer"
        if self.score >= 0.75:
            return "strong_match"
        if self.score >= 0.5:
            return "possible_match"
        return "weak_match"


def match_job(candidate: CandidateProfile, posting: JobPosting) -> JobMatch:
    candidate_skills = {skill.casefold().strip() for skill in candidate.skills if skill.strip()}
    required_skills = {skill.casefold().strip() for skill in posting.required_skills if skill.strip()}
    if not required_skills:
        skill_match = 1.0
    else:
        skill_match = len(candidate_skills & required_skills) / len(required_skills)

    if candidate.minimum_hourly_rate <= 0.0:
        pay_match = 1.0
    else:
        pay_match = min(1.0, max(0.0, posting.hourly_rate / candidate.minimum_hourly_rate))

    if candidate.maximum_distance_km <= 0.0:
        distance_match = 1.0 if posting.distance_km <= 0.0 else 0.0
    else:
        distance_match = max(0.0, 1.0 - (posting.distance_km / candidate.maximum_distance_km))

    verification_factor = 1.0 if posting.verified_employer else 0.5
    score = (
        (skill_match * 0.5) + (pay_match * 0.3) + (distance_match * 0.2)
    ) * verification_factor

    return JobMatch(
        score=max(0.0, min(1.0, score)),
        skill_match=skill_match,
        pay_match=pay_match,
        distance_match=distance_match,
        verified_employer=posting.verified_employer,
    )
