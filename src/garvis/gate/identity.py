from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import hmac
import json
import time


def _canon(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class PrimeIdentity:
    prime_id: str
    display_name: str
    owner_authority: str
    role: str
    presentation: str = "unspecified"

    def fingerprint(self) -> str:
        return hashlib.sha256(_canon(asdict(self))).hexdigest()


@dataclass(frozen=True)
class AuthorizationGrant:
    prime_id: str
    action: str
    scope: str
    issued_at: int
    expires_at: int
    nonce: str
    signature: str = ""

    def unsigned(self) -> dict:
        d = asdict(self)
        d.pop("signature")
        return d


def sign_grant(grant: AuthorizationGrant, secret: bytes) -> AuthorizationGrant:
    if not secret:
        raise ValueError("signing secret required")
    sig = hmac.new(secret, _canon(grant.unsigned()), hashlib.sha256).hexdigest()
    return AuthorizationGrant(**grant.unsigned(), signature=sig)


def verify_grant(grant: AuthorizationGrant, secret: bytes, *, prime_id: str, action: str, scope: str, now: int | None = None) -> bool:
    if not secret or not grant.signature:
        return False
    now = int(time.time()) if now is None else int(now)
    if grant.prime_id != prime_id or grant.action != action or grant.scope != scope:
        return False
    if grant.issued_at > now or grant.expires_at <= now:
        return False
    expected = hmac.new(secret, _canon(grant.unsigned()), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, grant.signature)


def social_relationship_grants_authority() -> bool:
    return False
