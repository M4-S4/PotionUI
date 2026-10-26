from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Mapping, Optional, Type

from src.features.llm.exceptions import LLMException


class HarnessError(LLMException):
    kind: ClassVar[str] = "error"
    retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        code: Optional[str] = None,
        retry_after: Optional[float] = None,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.retry_after = retry_after
        self.detail = dict(detail or {})

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "status": self.status,
            "code": self.code,
            "retry_after": self.retry_after,
            "detail": self.detail,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "HarnessError":
        cls = ERROR_KINDS.get(data.get("kind"), HarnessError)
        return cls(
            data.get("message", ""),
            status=data.get("status"),
            code=data.get("code"),
            retry_after=data.get("retry_after"),
            detail=data.get("detail"),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HarnessError) and type(other) is type(self) and other.to_json() == self.to_json()

    __hash__ = Exception.__hash__


class RateLimited(HarnessError):
    kind = "rate_limited"
    retryable = True


class QuotaExhausted(HarnessError):
    kind = "quota_exhausted"


class Overloaded(HarnessError):
    kind = "overloaded"
    retryable = True


class ServerFailed(HarnessError):
    kind = "server_error"
    retryable = True


class ConnectionFailed(HarnessError):
    kind = "connection"
    retryable = True


class ContextOverflow(HarnessError):
    kind = "context_overflow"


class AuthFailed(HarnessError):
    kind = "auth"


class InvalidRequest(HarnessError):
    kind = "invalid_request"


class OpaqueViolation(HarnessError):
    kind = "opaque_violation"


class MalformedStream(HarnessError):
    kind = "malformed_stream"


ERROR_KINDS: Dict[str, Type[HarnessError]] = {
    cls.kind: cls
    for cls in (
        HarnessError,
        RateLimited,
        QuotaExhausted,
        Overloaded,
        ServerFailed,
        ConnectionFailed,
        ContextOverflow,
        AuthFailed,
        InvalidRequest,
        OpaqueViolation,
        MalformedStream,
    )
}


@dataclass(frozen=True)
class WireFailure:
    status: Optional[int]
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)
    in_stream: bool = False
    cause: Optional[BaseException] = None

    def header(self, name: str) -> Optional[str]:
        wanted = name.lower()
        return next((value for key, value in self.headers.items() if key.lower() == wanted), None)

    def retry_after_seconds(self) -> Optional[float]:
        value = self.header("retry-after")
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None
