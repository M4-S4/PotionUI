import functools
import json

import httpx
import pytest

from src.features.llm.harness.errors import (
    AuthFailed,
    ConnectionFailed,
    ContextOverflow,
    HarnessError,
    InvalidRequest,
    OpaqueViolation,
    Overloaded,
    QuotaExhausted,
    RateLimited,
    ServerFailed,
    WireFailure,
)
from src.features.llm.harness.profile import ProfileSource
from src.features.llm.model_listing import ModelListingError
from tests.features.llm.harness.adapters.openai_responses_support import ADAPTER, make_config

_REAL_ASYNC_CLIENT = httpx.AsyncClient


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


class TestClassification:
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
        assert result.detail == {"code": "invalid_value", "type": "invalid_request_error", "param": "input", "request_id": "req_9"}

    def test_a_failure_inside_the_stream_is_marked_and_has_no_status(self):
        result = ADAPTER.classify_error(failure(None, {"code": "rate_limit_exceeded", "message": "Slow"}, in_stream=True))

        assert type(result) is RateLimited
        assert result.status is None
        assert result.detail["in_stream"] is True

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

    def test_every_result_is_a_harness_error(self):
        assert all(isinstance(ADAPTER.classify_error(failure(status, {"message": "x"})), HarnessError) for status in (400, 401, 404, 429, 500, 503))


@pytest.fixture
def http(monkeypatch):
    calls = []

    def install(handler):
        def recording(request):
            calls.append(request)
            return handler(request)

        monkeypatch.setattr(
            httpx, "AsyncClient",
            functools.partial(_REAL_ASYNC_CLIENT, transport=httpx.MockTransport(recording)),
        )
        return calls

    return install


class TestListing:
    async def test_models_are_listed_sorted_with_their_owner(self, http):
        calls = http(lambda request: httpx.Response(200, json={"data": [
            {"id": "o-mini", "owned_by": "system"},
            {"id": "gpt-5.6", "owned_by": "openai"},
            {"id": "plain"},
            {"nope": 1},
        ]}))

        models = await ADAPTER.list_models("https://api.openai.com/v1/", "test-key-a")

        assert str(calls[0].url) == "https://api.openai.com/v1/models"
        assert calls[0].headers["authorization"] == "Bearer test-key-a"
        assert [model.id for model in models] == ["gpt-5.6", "o-mini", "plain"]
        assert models[0].details == {"owned_by": "openai"}
        assert models[2].details is None

    async def test_listing_without_a_key_sends_no_authorization(self, http):
        calls = http(lambda request: httpx.Response(200, json={"data": []}))

        await ADAPTER.list_models("https://api.openai.com/v1")

        assert "authorization" not in calls[0].headers

    async def test_a_reply_without_a_list_is_reported(self, http):
        http(lambda request: httpx.Response(200, json={"object": "error"}))

        with pytest.raises(ModelListingError):
            await ADAPTER.list_models("https://api.openai.com/v1", "test-key-a")

    async def test_rejected_credentials_are_reported(self, http):
        http(lambda request: httpx.Response(401, json={}))

        with pytest.raises(ModelListingError, match="rejected the credentials"):
            await ADAPTER.list_models("https://api.openai.com/v1", "bad-key")


class TestDiscovery:
    async def test_discovery_returns_a_discovered_layer(self, http):
        http(lambda request: httpx.Response(200, json={"data": [{"id": "gpt-5.6", "owned_by": "openai"}]}))

        layer = await ADAPTER.discover(make_config())

        assert layer.source is ProfileSource.DISCOVERED
        assert dict(layer.fields) == {}

    async def test_a_context_window_the_server_reports_is_picked_up(self, http):
        http(lambda request: httpx.Response(200, json={"data": [
            {"id": "other", "context_window": 1000},
            {"id": "gpt-5.6", "context_length": 400000},
        ]}))

        layer = await ADAPTER.discover(make_config())

        assert dict(layer.fields) == {"context_window": 400000}

    async def test_discovery_failures_surface_to_the_resolver(self, http):
        http(lambda request: httpx.Response(500, json={}))

        with pytest.raises(ModelListingError):
            await ADAPTER.discover(make_config())
