"""Bounded, read-only internet field for GARVIS research."""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

DEFAULT_DOMAINS: tuple[str, ...] = (
    "jobbank.gc.ca",
    "canada.ca",
    "bankofcanada.ca",
    "ciro.ca",
    "securities-administrators.ca",
    "developer.bitcoin.org",
    "bitcoin.org",
    "github.com",
    "raw.githubusercontent.com",
    "upwork.com",
)


@dataclass(frozen=True)
class InternetPolicy:
    allowed_domains: tuple[str, ...] = DEFAULT_DOMAINS
    timeout_seconds: float = 12.0
    max_response_bytes: int = 300_000
    max_text_chars: int = 14_000
    max_links: int = 30


class _PageParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.text: list[str] = []
        self.links: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[Tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(urljoin(self.base_url, href))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            cleaned = " ".join(data.split())
            if cleaned:
                self.text.append(cleaned)


def load_policy() -> InternetPolicy:
    policy_path = Path(os.getenv("GARVIS_INTERNET_POLICY", "config/garvis_internet_field.json"))
    if not policy_path.is_file():
        return InternetPolicy()
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    return InternetPolicy(
        allowed_domains=tuple(payload.get("allowed_domains", DEFAULT_DOMAINS)),
        timeout_seconds=float(payload.get("timeout_seconds", 12.0)),
        max_response_bytes=int(payload.get("max_response_bytes", 300_000)),
        max_text_chars=int(payload.get("max_text_chars", 14_000)),
        max_links=int(payload.get("max_links", 30)),
    )


def _domain_allowed(hostname: str, policy: InternetPolicy) -> bool:
    host = hostname.casefold().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in policy.allowed_domains)


def _assert_public_host(hostname: str) -> None:
    addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("host did not resolve")
    for item in addresses:
        address = ipaddress.ip_address(item[4][0])
        if not address.is_global:
            raise ValueError("private, loopback, link-local, or reserved network targets are blocked")


def validate_url(url: str, policy: Optional[InternetPolicy] = None) -> str:
    active = policy or load_policy()
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http and https URLs are allowed")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL must contain a normal public hostname and no credentials")
    if not _domain_allowed(parsed.hostname, active):
        raise ValueError(f"domain is not in GARVIS allowlist: {parsed.hostname}")
    _assert_public_host(parsed.hostname)
    return parsed.geturl()


def read_url(url: str, policy: Optional[InternetPolicy] = None) -> dict[str, Any]:
    """Fetch one allowlisted page with GET only and return text plus same-policy links."""

    active = policy or load_policy()
    checked = validate_url(url, active)
    request = Request(
        checked,
        headers={"User-Agent": "GARVIS-Research/1.0 (+read-only; ProCityHub)"},
        method="GET",
    )
    with urlopen(request, timeout=active.timeout_seconds) as response:
        final_url = validate_url(response.geturl(), active)
        content_type = response.headers.get_content_type()
        if not (
            content_type.startswith("text/")
            or content_type in {"application/json", "application/xml", "application/xhtml+xml"}
        ):
            raise ValueError(f"unsupported content type: {content_type}")
        raw = response.read(active.max_response_bytes + 1)
        if len(raw) > active.max_response_bytes:
            raise ValueError("response exceeded GARVIS size limit")
        charset = response.headers.get_content_charset() or "utf-8"
        body = raw.decode(charset, errors="replace")

    if "html" in content_type:
        parser = _PageParser(final_url)
        parser.feed(body)
        text = "\n".join(parser.text)[: active.max_text_chars]
        links: list[str] = []
        seen: set[str] = set()
        for candidate in parser.links:
            try:
                validated = validate_url(candidate, active)
            except (ValueError, OSError, socket.gaierror):
                continue
            if validated not in seen:
                seen.add(validated)
                links.append(validated)
            if len(links) >= active.max_links:
                break
    else:
        text = body[: active.max_text_chars]
        links = []

    return {
        "source_url": final_url,
        "content_type": content_type,
        "text": text,
        "links": links,
        "notice": "Read-only source retrieval. No form submission, login, purchase, or transaction occurred.",
    }
