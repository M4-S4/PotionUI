import asyncio
import logging
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import aiohttp

from src.features.cloud.http import parse_retry_after
from src.features.providers.base_provider import (
    MarketplaceProviderBase,
    ProviderConnectionError,
    ProviderRateLimitError,
)
from src.platform.http.tls import aiohttp_connector
from src.platform.plugins.hooks import HookContext

logger = logging.getLogger(__name__)

DEFAULT_RETRY_AFTER_SECONDS = 60.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


def retry_after_from_headers(headers: Mapping[str, str], default: float = DEFAULT_RETRY_AFTER_SECONDS) -> float:
    seconds = parse_retry_after(headers.get("Retry-After"))
    return default if seconds is None else seconds


class HttpMarketplaceProvider(MarketplaceProviderBase):
    BASE_URL: str = ""
    DOWNLOAD_HOST: str = ""
    DEFAULT_RATE_LIMIT_DELAY: float = 1.0
    EXTRA_HEADERS: Mapping[str, str] = {}

    def __init__(self) -> None:
        self._api_key: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_delay: float = self.DEFAULT_RATE_LIMIT_DELAY
        self._last_request_time: float = 0
        self._headers: Dict[str, str] = {}
        self._initialized: bool = False

    async def initialize(self, settings: Dict[str, Any]) -> bool:
        self._api_key = settings.get("api_key") or None
        try:
            self._rate_limit_delay = float(settings.get("rate_limit_delay", self.DEFAULT_RATE_LIMIT_DELAY))
        except (TypeError, ValueError):
            self._rate_limit_delay = self.DEFAULT_RATE_LIMIT_DELAY

        self._headers = {"User-Agent": "PotionUI/1.0", **self.EXTRA_HEADERS}
        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"

        self._initialized = True
        logger.info("%s provider initialized (API key: %s)", self.provider_name, "configured" if self._api_key else "not configured")
        return True

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._initialized = False
        logger.info("%s provider shutdown", self.provider_name)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers, connector=aiohttp_connector())
        return self._session

    async def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.monotonic()

    async def _get_json(
        self,
        url: str,
        *,
        what: str,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> Tuple[int, Any]:
        options: Dict[str, Any] = {"timeout": aiohttp.ClientTimeout(total=timeout)}
        if params is not None:
            options["params"] = params
        if headers is not None:
            options["headers"] = headers
        session = await self._get_session()
        try:
            async with session.get(url, **options) as response:
                if response.status == 429:
                    raise ProviderRateLimitError(
                        f"{self.provider_name} rate limit exceeded",
                        retry_after=retry_after_from_headers(response.headers),
                    )
                if response.status == 200:
                    return 200, await response.json()
                try:
                    return response.status, await response.text()
                except Exception:
                    return response.status, ""
        except asyncio.TimeoutError:
            raise ProviderConnectionError(f"Timeout {what}")
        except aiohttp.ClientError as error:
            raise ProviderConnectionError(f"Connection error: {error}")

    def matches_download_url(self, url: str) -> bool:
        return bool(self.DOWNLOAD_HOST) and self.DOWNLOAD_HOST in url


def register_provider(
    context: HookContext, provider_id: str, label: str, load: Callable[[], type]
) -> HookContext:
    try:
        provider_class = load()
        providers = context.data.get("providers", {})
        providers[provider_id] = provider_class
        context.data["providers"] = providers
        logger.info("%s provider registered successfully", label)
    except ImportError as error:
        logger.error("Failed to import the %s provider: %s", label, error)
    except Exception as error:
        logger.error("Failed to register the %s provider: %s", label, error)
    return context
