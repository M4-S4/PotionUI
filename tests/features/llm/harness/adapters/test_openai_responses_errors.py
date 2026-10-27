import json

import httpx
import pytest

from src.features.llm.harness.errors import (
    AuthFailed,
    ConnectionFailed,
    ContextOverflow,
    InvalidRequest,
    OpaqueViolation,
    Overloaded,
    QuotaExhausted,
    RateLimited,
    ServerFailed,
    WireFailure,
)
from tests.features.llm.harness.adapters.openai_responses_support import ADAPTER


def failure(status, error=None, headers=None, in_stream=False, body=None):
    payload = json.dumps({"error": error}).encode("utf-8") if body is None else body
    return WireFailure(status=status, body=payload, headers=headers or {}, in_stream=in_stream)


CLASSIFIED = [
    ("invalid encrypted content", 400, {"code": "invalid_encrypted_content", "message": "Encrypted content organization_id did not match the target organization."}, OpaqueViolation),
    ("undecryptable content", 400, {"code": "invalid_encrypted_content", "message": "The encrypted content could not be decrypted or parsed."}, OpaqueViolation),
    ("orphaned reasoning", 400, {"message": "Item 'rs_1' of type 'reasoning' was provided without its required following item."}, OpaqueViolation),
    ("orphaned call", 400, {"message": "Item 'fc_1' of type 'function_call' was provided without its required 'reasoning' item: 'rs_1'."}, OpaqueViolation),
    ("context length", 400, {"code": "context_length_exceeded", "message": "This model's maximum context length is 400000 tokens."}, ContextOverflow),
    ("context window", 400, {"message": "Your input exceeds the context window of this model."}, ContextOverflow),
    ("quota", 429, {"code": "insufficient_quota", "type": "insufficient_quota", "message": "You exceeded your current quota."}, QuotaExhausted),
    ("quota by message", 429, {"message": "You exceeded your current quota, please check your plan."}, QuotaExhausted),
    ("rate limit", 429, {"code": "rate_limit_exceeded", "message": "Rate limit reached."}, RateLimited),
    ("rate limit without a body code", 429, {"message": "Too many requests"}, RateLimited),
    ("bad key", 401, {"code": "invalid_api_key", "message": "Incorrect API key provided."}, AuthFailed),
    ("forbidden", 403, {"message": "Country, region, or territory not supported"}, AuthFailed),
    ("overloaded", 503, {"message": "The engine is currently overloaded."}, Overloaded),
    ("overloaded by code", 500, {"code": "server_is_overloaded", "message": "Overloaded"}, Overloaded),
    ("server error", 500, {"code": "server_error", "message": "The server had an error."}, ServerFailed),
    ("bad gateway", 502, {"message": "Bad gateway"}, ServerFailed),
    ("plain bad request", 400, {"code": "invalid_value", "message": "Unknown parameter."}, InvalidRequest),
    ("encrypted content on a model that cannot", 400, {"message": "Encrypted content is not supported with this model: gpt-4o."}, InvalidRequest),
    ("not found", 404, {"message": "The model does not exist."}, InvalidRequest),
]


STREAM_CODES = [
    ("rate_limit_exceeded", RateLimited),
    ("insufficient_quota", QuotaExhausted),
    ("context_length_exceeded", ContextOverflow),
    ("invalid_api_key", AuthFailed),
    ("invalid_organization", AuthFailed),
    ("account_deactivated", AuthFailed),
    ("server_is_overloaded", Overloaded),
    ("overloaded", Overloaded),
    ("slow_down", Overloaded),
    ("server_error", ServerFailed),
    ("internal_error", ServerFailed),
    ("invalid_prompt", InvalidRequest),
    ("", InvalidRequest),
]


class TestClassification:
    @pytest.mark.parametrize("code, expected", STREAM_CODES)
    def test_failures_inside_the_stream_are_sorted_by_their_code(self, code, expected):
        result = ADAPTER.classify_error(failure(None, {"code": code, "message": "Stream failure"}, in_stream=True))

        assert type(result) is expected
        assert result.status is None
        assert result.detail["in_stream"] is True

    @pytest.mark.parametrize("label, status, error, expected", CLASSIFIED, ids=[row[0] for row in CLASSIFIED])
    def test_http_failures(self, label, status, error, expected):
        result = ADAPTER.classify_error(failure(status, error))

        assert type(result) is expected
        assert result.status == status
        assert result.message == error["message"]

    def test_the_retry_after_header_is_kept(self):
        result = ADAPTER.classify_error(failure(429, {"code": "rate_limit_exceeded", "message": "Slow"}, {"Retry-After": "7"}))

        assert result.retry_after == 7.0
        assert result.retryable is True

    def test_quota_is_not_retryable_even_with_a_retry_after(self):
        result = ADAPTER.classify_error(failure(429, {"code": "insufficient_quota", "message": "Out"}, {"retry-after": "3"}))

        assert result.retryable is False

    def test_the_request_id_and_error_fields_are_kept_for_diagnostics(self):
        error = {"code": "invalid_value", "type": "invalid_request_error", "param": "input", "message": "Bad"}

        result = ADAPTER.classify_error(failure(400, error, {"X-Request-Id": "req_9"}))

        assert result.code == "invalid_value"
        assert result.detail["request_id"] == "req_9"
        assert '"param": "input"' in result.detail["body"]

    def test_a_body_that_is_not_json_still_gives_a_typed_error(self):
        result = ADAPTER.classify_error(failure(502, body=b"<html>Bad gateway</html>"))

        assert type(result) is ServerFailed
        assert "Bad gateway" in result.message

    def test_a_string_error_is_used_as_the_message(self):
        result = ADAPTER.classify_error(failure(400, body=b'{"error": "nope"}'))

        assert type(result) is InvalidRequest
        assert result.message == "nope"

    def test_an_empty_body_falls_back_to_the_status(self):
        result = ADAPTER.classify_error(failure(500, body=b""))

        assert type(result) is ServerFailed
        assert "500" in result.message

    def test_a_transport_failure_is_a_connection_failure(self):
        cause = httpx.ConnectError("refused")

        result = ADAPTER.classify_error(WireFailure(status=None, cause=cause))

        assert type(result) is ConnectionFailed
        assert result.retryable is True
        assert "refused" in result.message

