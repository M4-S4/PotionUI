import json

import httpx
import pytest

from src.features.llm.harness.errors import (
    AuthFailed,
    ConnectionFailed,
    ContextOverflow,
    InvalidRequest,
    Overloaded,
    QuotaExhausted,
    RateLimited,
    ServerFailed,
    WireFailure,
)
from tests.features.llm.harness.adapters.openrouter_support import ADAPTER


def failure(status, error=None, headers=None, body=None):
    payload = json.dumps({"error": error}).encode() if body is None else body
    return WireFailure(status=status, body=payload, headers=headers or {})


CLASSIFIED = [
    ("bad request", 400, {"code": 400, "message": "Invalid param"}, InvalidRequest, False),
    ("no key", 401, {"code": 401, "message": "No auth credentials found"}, AuthFailed, False),
    ("no credits", 402, {"code": 402, "message": "Insufficient credits"}, QuotaExhausted, False),
    ("moderation", 403, {"code": 403, "message": "Input flagged", "metadata": {"reasons": ["violence"]}}, AuthFailed, False),
    ("timeout", 408, {"code": 408, "message": "Request timed out"}, ConnectionFailed, True),
    ("rate limit", 429, {"code": 429, "message": "Rate limited"}, RateLimited, True),
    ("bad gateway", 502, {"code": 502, "message": "Provider returned error"}, ServerFailed, True),
    ("no provider", 503, {"code": 503, "message": "No available provider"}, Overloaded, True),
]


class TestClassification:
    @pytest.mark.parametrize("label, status, error, expected, retryable", CLASSIFIED, ids=[row[0] for row in CLASSIFIED])
    def test_http_failures(self, label, status, error, expected, retryable):
        result = ADAPTER.classify_error(failure(status, error))

        assert type(result) is expected
        assert result.status == status
        assert result.retryable is retryable
        assert result.message.startswith(error["message"])

    def test_a_429_stays_a_rate_limit_even_when_the_message_mentions_credits(self):
        result = ADAPTER.classify_error(failure(429, {"code": 429, "message": "Free tier credits per minute exceeded"}))

        assert type(result) is RateLimited

    def test_the_retry_after_header_is_kept(self):
        result = ADAPTER.classify_error(failure(429, {"code": 429, "message": "Slow"}, {"Retry-After": "7"}))

        assert result.retry_after == 7.0

    def test_the_provider_metadata_is_kept_and_named_in_the_message(self):
        error = {
            "code": 502,
            "message": "Provider returned error",
            "metadata": {"provider_name": "Anthropic", "raw": {"type": "overloaded_error"}},
        }

        result = ADAPTER.classify_error(failure(502, error))

        assert result.message == "Provider returned error (Anthropic)"
        assert result.detail["metadata"] == error["metadata"]
        assert "overloaded_error" in result.detail["body"]

    def test_a_context_overflow_is_still_recognised(self):
        result = ADAPTER.classify_error(failure(400, {"code": 400, "message": "This model's maximum context length is 8192 tokens"}))

        assert type(result) is ContextOverflow

    def test_a_body_that_is_not_json_still_gives_a_typed_error(self):
        result = ADAPTER.classify_error(failure(502, body=b"<html>Bad gateway</html>"))

        assert type(result) is ServerFailed
        assert "detail" not in result.detail.get("metadata", {})

    def test_a_transport_failure_is_a_connection_failure(self):
        result = ADAPTER.classify_error(WireFailure(status=None, cause=httpx.ConnectError("refused")))

        assert type(result) is ConnectionFailed

    def test_an_in_stream_failure_keeps_its_flag(self):
        result = ADAPTER.classify_error(
            WireFailure(status=502, body=json.dumps({"error": {"code": 502, "message": "Gone"}}).encode(), in_stream=True)
        )

        assert type(result) is ServerFailed
        assert result.detail["in_stream"] is True
