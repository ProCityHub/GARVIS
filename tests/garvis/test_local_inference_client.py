from __future__ import annotations

from garvis.local_inference_client import completion_payload


def test_completion_payload_is_bounded_and_cacheable() -> None:
    payload = completion_payload(
        "heartbeat",
        n_predict=96,
    )

    assert payload["prompt"] == "heartbeat"
    assert payload["n_predict"] == 96
    assert payload["temperature"] == 0.0
    assert payload["cache_prompt"] is True
    assert payload["timings"] is True


def test_completion_payload_never_requests_zero_tokens() -> None:
    payload = completion_payload(
        "heartbeat",
        n_predict=0,
    )

    assert payload["n_predict"] == 1
