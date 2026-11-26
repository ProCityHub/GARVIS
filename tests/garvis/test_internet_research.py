import pytest

from garvis.internet_research import ResearchError, _host_is_or_subdomain, _validate_public_url


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


def test_official_asyncio_source_is_query_driven(monkeypatch) -> None:
    from garvis.internet_research import InternetResearchClient, ResearchPolicy

    client = InternetResearchClient(
        policy=ResearchPolicy(
            max_results=1,
            max_pages=0,
        )
    )

    report = client.research("Research the current official Python asyncio documentation")

    assert len(report.sources) == 1
    assert report.sources[0].url == "https://docs.python.org/3/library/asyncio.html"
    assert report.sources[0].domain == "docs.python.org"
    assert "official_catalog" in report.provider


def test_zero_source_report_is_refused(monkeypatch) -> None:
    from garvis.internet_research import InternetResearchClient

    client = InternetResearchClient()

    monkeypatch.setattr(
        client,
        "_official_sources",
        lambda _query: [],
    )
    monkeypatch.setattr(
        client,
        "_duckduckgo",
        lambda _query: [],
    )
    monkeypatch.setattr(
        client,
        "_duckduckgo_lite",
        lambda _query: [],
    )
    monkeypatch.setattr(
        client,
        "_wikipedia",
        lambda _query: [],
    )
    monkeypatch.setattr(
        client,
        "_github",
        lambda _query: [],
    )
    monkeypatch.setattr(
        client,
        "_arxiv",
        lambda _query: [],
    )

    with pytest.raises(
        ResearchError,
        match="No public results available",
    ):
        client.research("a query with no available source")


def test_duckduckgo_lite_ignores_non_result_links(monkeypatch) -> None:
    import garvis.internet_research as research_module
    from garvis.internet_research import InternetResearchClient, ResearchPolicy

    client = InternetResearchClient(
        policy=ResearchPolicy(
            max_results=5,
            max_pages=0,
        )
    )

    html = """
    <html>
      <body>
        <a href="https://navigation.example/help">Navigation</a>
        <a class="result-link" href="https://result-one.example/page">
          Result One
        </a>
        <a href="https://result-two.example/page" class="result-link">
          Result Two
        </a>
      </body>
    </html>
    """

    monkeypatch.setattr(
        research_module,
        "_validate_public_url",
        lambda _url: None,
    )
    monkeypatch.setattr(
        client,
        "_get",
        lambda url: (
            html.encode("utf-8"),
            "text/html",
            url,
        ),
    )

    results = client._duckduckgo_lite("test query")

    assert [item.url for item in results] == [
        "https://result-one.example/page",
        "https://result-two.example/page",
    ]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://duckduckgo.com/html", True),
        ("https://lite.duckduckgo.com/lite/", True),
        ("https://sub.lite.duckduckgo.com/path", True),
        ("https://evilduckduckgo.com/", False),
        ("https://notduckduckgo.com/", False),
        ("https://duckduckgo.com.evil.example/path", False),
        ("https://DUCKDUCKGO.COM/search", True),
        ("https://DuckDuckGo.COM:443/search", True),
        ("https://lite.duckduckgo.com:8443/lite/", True),
        ("https://duckduckgo.com./", True),
        ("https://duckduckgo.com.:8443/x", True),
        ("https://duckduckgo.com@evil.example/path", False),
        ("https://user:pass@evilduckduckgo.com/", False),
        ("https://evil.example@duckduckgo.com/path", True),
        ("https://user:password@lite.duckduckgo.com:443/lite/", True),
        ("https://duckduckgo.com%2e.evil.example/", False),
        ("", False),
    ],
)
def test_host_match_rejects_lookalikes(url: str, expected: bool) -> None:
    assert _host_is_or_subdomain(url, "duckduckgo.com") is expected


def test_duckduckgo_lite_keeps_lookalikes_and_drops_real_hosts(monkeypatch) -> None:
    import garvis.internet_research as research_module
    from garvis.internet_research import InternetResearchClient, ResearchPolicy

    client = InternetResearchClient(policy=ResearchPolicy(max_results=8, max_pages=0))
    html = """
    <html><body>
      <a class="result-link" href="https://evilduckduckgo.com/phish">Lookalike</a>
      <a class="result-link" href="https://lite.duckduckgo.com/settings">Subdomain</a>
      <a class="result-link" href="https://DUCKDUCKGO.COM:443/html">Upper port</a>
      <a class="result-link" href="https://duckduckgo.com./">Trailing dot</a>
      <a class="result-link" href="https://duckduckgo.com@kept.example/page">Userinfo trick</a>
      <a class="result-link" href="https://user:pw@html.duckduckgo.com/">Real subdomain</a>
      <a class="result-link" href="https://docs.example/guide">Real result</a>
    </body></html>
    """
    monkeypatch.setattr(research_module, "_validate_public_url", lambda _url: None)
    monkeypatch.setattr(
        client,
        "_get",
        lambda url: (html.encode("utf-8"), "text/html", url),
    )

    assert [item.url for item in client._duckduckgo_lite("test query")] == [
        "https://evilduckduckgo.com/phish",
        "https://duckduckgo.com@kept.example/page",
        "https://docs.example/guide",
    ]


def test_github_non_object_payload_returns_empty(monkeypatch) -> None:
    from garvis.internet_research import InternetResearchClient

    client = InternetResearchClient()

    monkeypatch.setattr(
        client,
        "_get",
        lambda url: (
            b'["unexpected", "payload"]',
            "application/json",
            url,
        ),
    )

    assert client._github("python autonomous agent") == []
