import httpx

from backend import http_client
from src.plugin_api.providers import client_ssl_context


def test_new_client_uses_shared_tls_context(monkeypatch):
    seen = []
    real = httpx.AsyncClient

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", recording)
    http_client.new_client()

    assert seen[0]["verify"] is client_ssl_context()


def test_new_client_keeps_transport_override(monkeypatch):
    seen = []
    real = httpx.AsyncClient

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    monkeypatch.setattr(httpx, "AsyncClient", recording)
    http_client.set_transport_override(transport)
    http_client.new_client()

    assert seen[0]["transport"] is transport
    assert seen[0]["timeout"] == http_client.REQUEST_TIMEOUT_SECONDS
