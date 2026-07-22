import pytest

from garvis.internet_research import ResearchError, _validate_public_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/private",
        "http://localhost/private",
        "file:///sdcard/Download/private.txt",
        "http://user:password@example.com/",
    ],
)
def test_blocked_urls(url: str) -> None:
    with pytest.raises(ResearchError):
        _validate_public_url(url)
