import json
from typing import Any, Callable, List, Optional

import httpx

REAL_CLIENT = httpx.AsyncClient


class Peer:
    def __init__(self, responders: List[Callable[[httpx.Request], httpx.Response]]):
        self.requests: List[httpx.Request] = []
        self.timeouts: List[Any] = []
        self._responders = list(responders)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responders[min(len(self.requests), len(self._responders)) - 1](request)

    @property
    def body(self) -> Any:
        return json.loads(self.requests[-1].content)


def install_peer(monkeypatch, *responders: Callable[[httpx.Request], httpx.Response]) -> Peer:
    peer = Peer(list(responders))

    def factory(*args, **kwargs):
        peer.timeouts.append(kwargs.get("timeout"))
        kwargs["transport"] = httpx.MockTransport(peer.handle)
        return REAL_CLIENT(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return peer


def chunked(chunks: List[bytes], status: int = 200, headers: Optional[dict] = None) -> Callable[[httpx.Request], httpx.Response]:
    async def body():
        for chunk in chunks:
            yield chunk

    return lambda request: httpx.Response(status, content=body(), headers=headers)
