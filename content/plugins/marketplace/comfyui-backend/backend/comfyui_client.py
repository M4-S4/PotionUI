from contextlib import asynccontextmanager
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

import aiohttp

from src.plugin_api.providers import aiohttp_connector, certificate_failure_message, is_certificate_error


class ComfyUiCertificateError(aiohttp.ClientError):
    pass


class ComfyUiClient:
    def __init__(self, base_url: str, timeout_seconds: Optional[float] = None):
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._session: Any = None

    async def __aenter__(self) -> "ComfyUiClient":
        options: Dict[str, Any] = {"connector": aiohttp_connector()}
        if self._timeout_seconds is not None:
            options["timeout"] = aiohttp.ClientTimeout(total=self._timeout_seconds)
        self._session = await aiohttp.ClientSession(**options).__aenter__()
        return self

    async def __aexit__(self, *exc_info) -> bool:
        session, self._session = self._session, None
        if session is not None:
            await session.__aexit__(*exc_info)
        return False

    def get(self, path: str, request_timeout: Optional[float] = None):
        return self._request("get", path, request_timeout)

    def post(self, path: str, request_timeout: Optional[float] = None, **kwargs: Any):
        return self._request("post", path, request_timeout, **kwargs)

    @asynccontextmanager
    async def _request(self, method: str, path: str, request_timeout: Optional[float], **kwargs: Any):
        send = getattr(self._session, method)
        try:
            async with send(f"{self._base_url}{path}", **self._options(request_timeout), **kwargs) as resp:
                yield resp
        except aiohttp.ClientError as e:
            if is_certificate_error(e):
                raise ComfyUiCertificateError(certificate_failure_message(urlparse(self._base_url).netloc)) from e
            raise

    @staticmethod
    def _options(request_timeout: Optional[float]) -> Dict[str, Any]:
        if request_timeout is None:
            return {}
        return {"timeout": aiohttp.ClientTimeout(total=request_timeout)}

    async def get_json(self, path: str) -> Any:
        async with self.get(path) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def object_info(self) -> Dict[str, Any]:
        return await self.get_json("/object_info")

    async def folders(self) -> List[str]:
        return await self.get_json("/models")

    async def model_names(self, folder: str) -> List[str]:
        return await self.get_json(f"/models/{folder}")

    async def models(self, folder: str) -> List[Dict[str, Any]]:
        try:
            return await self.get_json(f"/experiment/models/{folder}")
        except (aiohttp.ClientError, TimeoutError):
            return [{"name": name} for name in await self.model_names(folder)]
