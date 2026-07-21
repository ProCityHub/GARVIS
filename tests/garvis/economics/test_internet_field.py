import pytest

from garvis.economics.internet_field import InternetPolicy, validate_url


def test_allowlist_rejects_unapproved_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("garvis.economics.internet_field._assert_public_host", lambda _: None)
    policy = InternetPolicy(allowed_domains=("example.com",))
    with pytest.raises(ValueError, match="allowlist"):
        validate_url("https://unapproved.example.net/path", policy)


def test_allowlist_accepts_subdomain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("garvis.economics.internet_field._assert_public_host", lambda _: None)
    policy = InternetPolicy(allowed_domains=("example.com",))
    assert validate_url("https://docs.example.com/path", policy).startswith("https://")
