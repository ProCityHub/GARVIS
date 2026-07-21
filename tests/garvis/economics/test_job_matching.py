from garvis.economics import CandidateProfile, JobPosting, match_job


def test_verified_drywall_job_is_strong_match() -> None:
    candidate = CandidateProfile(
        skills=("drywall", "quality control", "interior systems"),
        minimum_hourly_rate=35.0,
        maximum_distance_km=50.0,
    )
    posting = JobPosting(
        title="Drywall quality-control mechanic",
        required_skills=("drywall", "quality control"),
        hourly_rate=42.0,
        distance_km=10.0,
        verified_employer=True,
    )

    result = match_job(candidate, posting)

    assert result.recommendation == "strong_match"
    assert result.skill_match == 1.0


def test_unverified_employer_requires_verification() -> None:
    candidate = CandidateProfile(("drywall",), 30.0, 25.0)
    posting = JobPosting("Remote estimator", ("drywall",), 40.0, 0.0, False)

    assert match_job(candidate, posting).recommendation == "verify_employer"
